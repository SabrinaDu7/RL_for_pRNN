"""Is the observation IN the hidden state, and the readout losing it?

The experiment: freeze a checkpoint, collect (h, target) along the probe
walk in every training room, fit a FRESH linear head - the exact functional
form of the network's own `outlayer` (Linear hidden→n_tiles·C, no squash) -
on held-out episodes, and score both heads on the published test protocol
with the same metrics `surprisal_timing.measure` pins. If the fresh head
beats the network's own, the representation carries more than the trained
readout extracts - there is room in the readout; if it matches, the ceiling
is the representation itself and readout work is pointless.

Splits are disjoint by env seed and walker seed (`walk_episodes`): train
seed_base 1000, early-stop validation 2000, test 100 = THE published
protocol, so the own-readout column reproduces the committed numbers.

ANALYSIS ONLY. Run:
    uv run python -m curious_george.evaluation.readout_probe --ckpt <run>/predictiveNet_state.pt
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int

from curious_george.evaluation.error_decomposition import per_tile_errors
from curious_george.evaluation.surprisal_timing import (
    EPISODE_STEPS,
    LANDMARK_CLASS_IDS,
    walk_episodes,
)
from curious_george.models.device import eval_mode

TRAIN_EPISODES_PER_ROOM = 8
VAL_EPISODES_PER_ROOM = 2
TEST_EPISODES_PER_ROOM = 2  # the published protocol's count


@dataclass(frozen=True)
class ReadoutMetrics:
    """The committed comparison axes, one row per readout."""

    landmark_shown: float  # nats/tile
    landmark_masked: float
    background: float
    recall_shown: float
    recall_masked: float
    miss_shown: float
    miss_masked: float

    def row(self, name: str) -> str:
        return (
            f"{name:<14} shown {self.landmark_shown:.3f} / masked "
            f"{self.landmark_masked:.3f} nats | bg {self.background:.3f} | "
            f"recall {self.recall_shown:.3f}/{self.recall_masked:.3f} | "
            f"miss {self.miss_shown:.3f}/{self.miss_masked:.3f}"
        )


def score(
    logits: Float[torch.Tensor, "N X"],
    targets_px: Float[torch.Tensor, "N 147"],
    shown: Bool[np.ndarray, " N"],
) -> ReadoutMetrics:
    per, classes = per_tile_errors(logits, targets_px, ce=True)
    lm_ids = torch.tensor(sorted(LANDMARK_CLASS_IDS))
    is_lm = torch.isin(classes, lm_ids)
    argmax = logits.reshape(logits.shape[0], classes.shape[1], -1).argmax(-1)
    correct = argmax == classes
    sh = torch.as_tensor(shown)
    lm_rows = is_lm.any(-1)

    def mean_over(err_mask, row_mask):
        sel = err_mask & row_mask.unsqueeze(-1)
        return float(per[sel].mean())

    def frame_stats(row_mask):
        rows = torch.where(lm_rows & row_mask)[0]
        rec = torch.stack([correct[t][is_lm[t]].float().mean() for t in rows])
        mis = torch.stack([
            (~torch.isin(argmax[t], lm_ids).any()).float() for t in rows
        ])
        return float(rec.mean()), float(mis.mean())

    rec_s, miss_s = frame_stats(sh)
    rec_m, miss_m = frame_stats(~sh)
    return ReadoutMetrics(
        landmark_shown=mean_over(is_lm, sh),
        landmark_masked=mean_over(is_lm, ~sh),
        background=mean_over(~is_lm, sh | ~sh),
        recall_shown=rec_s, recall_masked=rec_m,
        miss_shown=miss_s, miss_masked=miss_m,
    )


def collect(
    *, adapter, env, layouts, episodes_per_room: int, seed_base: int, walker_seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """(h, own-readout logits, target pixels, shown) over one walk, stacked."""
    hs, preds, targets = [], [], []
    shown_one = np.resize(np.asarray(adapter.pN.pRNN.inMask, dtype=bool), EPISODE_STEPS)
    n = 0
    with eval_mode([adapter.pN]), torch.no_grad():
        for _room, obss, acts, last in walk_episodes(
            env=env, layouts=layouts, episodes_per_room=episodes_per_room,
            seed_base=seed_base, walker_seed=walker_seed,
        ):
            pred, target, h, _ = adapter.episode_prediction_rows(
                obss, np.array(acts), last, target_offset=1
            )
            hs.append(h)
            preds.append(pred)
            targets.append(target)
            n += 1
    return (
        torch.cat(hs), torch.cat(preds), torch.cat(targets),
        np.tile(shown_one, n),
    )


def fit_linear_head(
    h_train: Float[torch.Tensor, "N H"],
    px_train: Float[torch.Tensor, "N 147"],
    h_val: Float[torch.Tensor, "N2 H"],
    px_val: Float[torch.Tensor, "N2 147"],
    *, out_features: int, device: str, focal_gamma: float | None = None,
    hidden: int = 0, max_epochs: int = 8000, patience: int = 150,
) -> torch.nn.Module:
    """CE fit of Linear(H → out_features), val-early-stopped.

    `focal_gamma` applies the (1-pt)^γ weighting DURING FITTING only - the
    validation criterion stays plain CE either way, so two fits are early
    stopped against the same yardstick. Fitting plain against a
    focal-trained checkpoint reproduces the plain-CE pathology on a frozen
    h (measured: bg 0.14 / landmarks 1.16) - the objective, not the head,
    decides where a linear readout spends itself."""
    from curious_george.envs.palette import vocab_tensor

    vocab = vocab_tensor().to(device)
    C, ch = vocab.shape

    def classes_of(px):
        pix = px.reshape(px.shape[0], -1, ch).to(device)
        dist = (pix.unsqueeze(-2) - vocab).abs().sum(-1)
        mind, cls = dist.min(-1)
        assert float(mind.max()) < 1e-3
        return cls  # (N, 49)

    y_tr, y_va = classes_of(px_train), classes_of(px_val)
    x_tr, x_va = h_train.to(device), h_val.to(device)
    torch.manual_seed(0)
    head = (
        torch.nn.Linear(x_tr.shape[1], out_features)
        if not hidden else
        torch.nn.Sequential(
            torch.nn.Linear(x_tr.shape[1], hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_features),
        )
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(max_epochs):
        for i in torch.randperm(x_tr.shape[0], device=device).split(4096):
            logits = head(x_tr[i]).reshape(len(i), -1, C)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, C), y_tr[i].reshape(-1), reduction="none"
            )
            if focal_gamma is not None:
                ce = (1 - torch.exp(-ce)) ** focal_gamma * ce
            loss = ce.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            # Early stop on the FIT objective evaluated on val - stopping a
            # focal fit against plain-CE val stops it where plain CE likes,
            # which is the exact bias the focal fit exists to escape.
            vce = torch.nn.functional.cross_entropy(
                head(x_va).reshape(-1, C), y_va.reshape(-1), reduction="none"
            )
            if focal_gamma is not None:
                vce = (1 - torch.exp(-vce)) ** focal_gamma * vce
            vl = float(vce.mean())
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    tag = "focal-val" if focal_gamma is not None else "plainCE-val"
    capped = " (EPOCH CAP - not converged)" if epoch == max_epochs - 1 else ""
    print(f"fresh head: stopped epoch {epoch}, {tag} {best:.4f}{capped}")
    return head


def main() -> None:
    import argparse

    from curious_george.configs import cli
    from curious_george.envs.layouts import ROOMS_SELECTED, with_affordance
    from curious_george.models.prnn_adapter import PRNNAdapter
    from curious_george.training.setup import setup_env, setup_world_model
    from prnn.utils.checkpoints import load_pN

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--readout", choices=["LINEAR", "MLP"], default="LINEAR",
                    help="the checkpoint's readout architecture (a mismatch fails at load)")
    ap.add_argument("--positions", type=int, nargs="+", default=[0, 1, 2, 3, 5, 6, 7, 8])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gamma", type=float, default=5.0,
                    help="focal gamma for the matched-objective fresh fit")
    ap.add_argument("--train-eps", type=int, default=TRAIN_EPISODES_PER_ROOM,
                    help="training episodes per room for the fresh fits")
    ap.add_argument("--hidden", type=int, default=0,
                    help="if >0, also fit a 1-hidden-layer MLP probe this wide")
    args = ap.parse_args()

    cfg = cli(
        ["multienv-fast", "--arch-prnn.loss", "CE", "--run.no-wandb",
         "--arch-prnn.readout", args.readout,
         "env.source:selected", "--env.source.n", str(len(args.positions)),
         "--env.source.impassable", "--env.source.positions",
         *map(str, args.positions)]
    )
    layouts = [
        with_affordance((ROOMS_SELECTED[p],), impassable=True)[0]
        for p in args.positions
    ]
    env = setup_env(cfg, landmarks=list(layouts[0].landmarks))
    pN = setup_world_model(cfg, env, wandb_log=False)
    load_pN(model_ckpt_filepath=args.ckpt, device="cpu",
            pRNNtype=cfg.arch_prnn.prnn_type.value, predictive_net=pN)
    pN.pRNN.to("cpu")
    adapter = PRNNAdapter(pN, torch.device("cpu"), action_offset=0)
    torch.manual_seed(11)  # offline CLI; see surprisal_timing.measure

    splits = {
        name: collect(adapter=adapter, env=env, layouts=layouts,
                      episodes_per_room=eps, seed_base=base, walker_seed=wseed)
        for name, eps, base, wseed in [
            ("train", args.train_eps, 1000, 17),
            ("val", VAL_EPISODES_PER_ROOM, 2000, 23),
            ("test", TEST_EPISODES_PER_ROOM, 100, 11),
        ]
    }
    h_te, own_logits, px_te, shown_te = splits["test"]

    # Internal validity: the network's own readout IS Linear(h) - if this
    # fails, h is not what outlayer consumes and the comparison means nothing.
    with torch.no_grad():
        recomputed = pN.pRNN.outlayer(h_te)
    assert torch.allclose(recomputed, own_logits, atol=1e-4), (
        "outlayer(h) != the predict path's logits; h rows are not the "
        "readout's input - fix episode_prediction_rows alignment first"
    )

    print(score(own_logits, px_te, shown_te).row("own outlayer"))
    arms = [(None, 0, "fresh plainCE"), (args.gamma, 0, f"fresh focal{args.gamma:g}")]
    if args.hidden:
        arms.append((args.gamma, args.hidden, f"mlp{args.hidden} focal{args.gamma:g}"))
    for gamma, hidden, name in arms:
        head = fit_linear_head(
            splits["train"][0], splits["train"][2],
            splits["val"][0], splits["val"][2],
            out_features=own_logits.shape[1], device=args.device,
            focal_gamma=gamma, hidden=hidden,
        )
        with torch.no_grad():
            fresh_logits = head(h_te.to(args.device)).cpu()
        print(score(fresh_logits, px_te, shown_te).row(name))


if __name__ == "__main__":
    main()
