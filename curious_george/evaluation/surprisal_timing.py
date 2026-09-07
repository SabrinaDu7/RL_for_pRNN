"""Landmark surprisal vs episode time and vs steps-since-glimpse.

The question: is poor landmark prediction an INFERENCE failure (the model has
not identified which room it is in, so surprisal should fall within an episode
and collapse after a landmark glimpse) or a RECONSTRUCTION failure (surprisal
stays near chance ln C even at steps where the landmark is in the input)?

This is the measurement that fired the focal-loss trigger on 2026-08-31
(`configs.ArchPrnnCfg.focal_gamma`): landmark tiles near chance EVEN at shown
steps, background saturated far below - not inference, reconstruction. The
original one-off ran through train-mode dropout; this module fixes the
protocol to `eval_mode` and is the tracked home of the result
(docs/figures/surprisal_vs_time.png).

ANALYSIS ONLY: loads a checkpoint, walks frozen rooms with the project's
forward-biased random walker, and decomposes per-tile surprisal via
`error_decomposition.per_tile_errors`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from curious_george.configs import RAND_ACT_PROBA, Config
from curious_george.envs.layouts import LANDMARK_COLORS, Layout, Selected, resolve_layouts
from curious_george.envs.palette import TILE_CLASS_NAMES
from curious_george.evaluation.error_decomposition import per_tile_errors
from curious_george.models.device import eval_mode
from curious_george.models.prnn_adapter import PRNNAdapter

#: Class ids counted as "landmark tiles", derived from the palette - the
#: complement is background (floor, wall, agent).
LANDMARK_CLASS_IDS: frozenset[int] = frozenset(
    i for i, name in enumerate(TILE_CLASS_NAMES) if name in LANDMARK_COLORS
)

#: The probe protocol, matching the 2026-08-31 trigger measurement: 256-step
#: episodes, 2 per room, seeds 100 + 10*room_index + episode, walker rng 11.
EPISODE_STEPS = 256
EPISODES_PER_ROOM = 2
SINCE_GLIMPSE_MAX = 40
TIME_BIN = 16


@dataclass(frozen=True)
class SurprisalTiming:
    """Per-tile surprisal (nats) split landmark/background along two clocks,
    plus the blunter argmax view: recall = fraction of landmark tiles whose
    argmax class is right, miss = fraction of landmark-bearing frames whose
    prediction contains NO landmark class at all. Surprisal rewards
    calibration; recall is what a rendered figure shows."""

    landmark_by_bin: Float[np.ndarray, "T/16"]  # episode time, TIME_BIN bins
    background_by_bin: Float[np.ndarray, "T/16"]
    since_glimpse: Float[np.ndarray, "40"]  # steps since last SHOWN glimpse
    blind_step_mean: float  # landmark surprisal at obs-masked steps
    shown_step_mean: float  # ... at steps with the obs in the input
    chance: float  # ln C for the committed palette
    recall_shown: float  # landmark-tile argmax recall at shown steps
    recall_masked: float
    miss_shown: float  # frames with the landmark entirely absent
    miss_masked: float


def walk_episodes(
    *, env, layouts: list, episodes_per_room: int,
    seed_base: int = 100, walker_seed: int = 11,
):
    """The probe walk, one home: yields (room_index, obss, acts, last_obs).

    Env reset seeds are `seed_base + 10*room_index + episode`; the walker is
    the project's forward-biased categorical, one Generator consumed across
    the whole walk. The DEFAULTS are the published protocol (`measure` uses
    them; its numbers are pinned in docs/focal-loss-2026-08-31.md); other
    seeds give disjoint episodes on the same distribution - what
    `readout_probe` trains on.
    """
    T = EPISODE_STEPS
    rng = np.random.default_rng(walker_seed)
    for room_index, layout in enumerate(layouts):
        env.env.unwrapped.landmarks = list(layout.landmarks)
        for ep in range(episodes_per_room):
            out = env.reset(seed=seed_base + 10 * room_index + ep)
            obs = out[0] if isinstance(out, tuple) else out
            obss, acts = [], []
            for _ in range(T):
                a = int(rng.choice(4, p=list(RAND_ACT_PROBA)))
                obss.append(obs)
                acts.append(a)
                obs = env.step(a)[0]
            yield room_index, obss, acts, obs


def rooms_at(cfg: Config, positions: tuple[int, ...] | None) -> list[Layout]:
    """The run's rooms in training order, or the subset at these `ROOMS_SELECTED`
    positions - how a floor is measured on one room of an 8-room checkpoint.
    Positions the run never trained on are an error, not an extra room."""
    layouts = resolve_layouts(cfg)
    if positions is None:
        return layouts
    if not isinstance(cfg.env.source, Selected):
        raise ValueError(f"--positions needs a Selected room set; the run has {cfg.env.source!r}")
    at = dict(zip(cfg.env.source.positions, layouts, strict=True))
    if missing := [p for p in positions if p not in at]:
        raise ValueError(f"positions {missing} are not in the run's set {cfg.env.source.positions}")
    return [at[p] for p in positions]


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A world-model checkpoint on the CPU, wired as its run wired it."""

    cfg: Config
    layouts: list[Layout]
    env: object
    adapter: PRNNAdapter


def load_checkpoint(
    ckpt: str | Path, *, positions: tuple[int, ...] | None = None
) -> LoadedCheckpoint:
    """`ckpt` under the config its run trained under (`Config.of_run`, from the
    provenance beside the checkpoint or one level up for an archived copy):
    the same rooms through the same `resolve_layouts`, the circuit's own
    `action_offset`. Until 2026-09-06 the probe tools re-typed the config from
    flags and hard-coded `action_offset=0` (audit 2026-09-05, C11)."""
    from prnn.utils.checkpoints import load_pN

    from curious_george.training.setup import setup_env, setup_world_model
    from curious_george.utils.checkpoints import run_dir_of

    cfg = Config.of_run(run_dir_of(ckpt))
    layouts = rooms_at(cfg, positions)
    env = setup_env(cfg, landmarks=list(layouts[0].landmarks))
    pN = setup_world_model(cfg, env, wandb_log=False)
    load_pN(model_ckpt_filepath=str(ckpt), device="cpu",
            pRNNtype=cfg.arch_prnn.prnn_type.value, predictive_net=pN)
    pN.pRNN.to("cpu")
    adapter = PRNNAdapter(pN, torch.device("cpu"), action_offset=cfg.arch_prnn.action_offset)
    return LoadedCheckpoint(cfg=cfg, layouts=layouts, env=env, adapter=adapter)


def measure(*, adapter, env, layouts: list) -> SurprisalTiming:
    """Walk each layout with the forward-biased walker and decompose.

    `adapter` is a `PRNNAdapter` whose net predicts logits (CE); `env` is the
    single-room shell whose `unwrapped.landmarks` this rebinds per layout.
    Runs under `eval_mode` - dropout OFF; the pRNN's additive state noise is
    part of `predict` itself and stays, as it does for every metric.
    """
    T = EPISODE_STEPS
    shown = np.resize(np.asarray(adapter.pN.pRNN.inMask, dtype=bool), T)
    lm_sum, bg_sum = np.zeros(T), np.zeros(T)
    lm_n, bg_n = np.zeros(T), np.zeros(T)
    since_sum, since_n = np.zeros(SINCE_GLIMPSE_MAX), np.zeros(SINCE_GLIMPSE_MAX)
    blind, shown_vals = [], []
    recall: dict = {"s": [], "m": []}
    miss: dict = {"s": [], "m": []}
    # The pRNN's additive state noise draws from torch's GLOBAL stream, which
    # this offline tool may enter in any state - unseeded, repeated
    # invocations wobbled by ~0.01 nats (found 2026-08-31 when the recall
    # numbers would not reproduce exactly). An offline CLI may seed globally;
    # the training-path probes must NOT (see spatial._probe_rng).
    torch.manual_seed(11)

    with eval_mode([adapter.pN]), torch.no_grad():
        for _room, obss, acts, obs in walk_episodes(
            env=env, layouts=layouts, episodes_per_room=EPISODES_PER_ROOM
        ):
            pred, target, _, _ = adapter.episode_prediction_rows(
                obss, np.array(acts), obs, target_offset=1
            )
            per, classes = per_tile_errors(pred, target, ce=True)
            lm_ids = torch.tensor(sorted(LANDMARK_CLASS_IDS))
            is_lm = torch.isin(classes, lm_ids)
            argmax = pred.reshape(pred.shape[0], classes.shape[1], -1).argmax(-1)
            correct = argmax == classes
            since = np.inf
            for t in range(T):
                lm_mask = is_lm[t]
                if lm_mask.any():
                    lm_sum[t] += float(per[t][lm_mask].sum())
                    lm_n[t] += int(lm_mask.sum())
                    step_mean = float(per[t][lm_mask].mean())
                    (shown_vals if shown[t] else blind).append(step_mean)
                    key = "s" if shown[t] else "m"
                    recall[key].append(float(correct[t][lm_mask].float().mean()))
                    miss[key].append(
                        0.0 if bool(torch.isin(argmax[t], lm_ids).any()) else 1.0
                    )
                    if since < SINCE_GLIMPSE_MAX:
                        since_sum[int(since)] += step_mean
                        since_n[int(since)] += 1
                bg_sum[t] += float(per[t][~lm_mask].sum())
                bg_n[t] += int((~lm_mask).sum())
                since = 0 if (shown[t] and lm_mask.any()) else since + 1

    bins = np.arange(0, T, TIME_BIN)
    binned = lambda s, n: np.array(
        [s[i : i + TIME_BIN].sum() / max(n[i : i + TIME_BIN].sum(), 1) for i in bins]
    )
    return SurprisalTiming(
        landmark_by_bin=binned(lm_sum, lm_n),
        background_by_bin=binned(bg_sum, bg_n),
        since_glimpse=since_sum / np.maximum(since_n, 1),
        blind_step_mean=float(np.mean(blind)),
        shown_step_mean=float(np.mean(shown_vals)),
        chance=float(np.log(len(TILE_CLASS_NAMES))),
        recall_shown=float(np.mean(recall["s"])),
        recall_masked=float(np.mean(recall["m"])),
        miss_shown=float(np.mean(miss["s"])),
        miss_masked=float(np.mean(miss["m"])),
    )


def plot(result: SurprisalTiming, *, out_path: str, title_note: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(result.landmark_by_bin)) * TIME_BIN + TIME_BIN / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t, result.landmark_by_bin, "o-", label="landmark tiles")
    axes[0].plot(t, result.background_by_bin, "s-", label="background tiles")
    axes[0].axhline(result.chance, ls=":", c="grey", label="chance ln(7)")
    axes[0].set_xlabel("step within episode")
    axes[0].set_ylabel("surprisal (nats per tile)")
    axes[0].legend()
    axes[0].set_title(f"surprisal vs episode time\n({title_note})")
    axes[1].plot(np.arange(SINCE_GLIMPSE_MAX), result.since_glimpse, "o-")
    axes[1].axhline(result.chance, ls=":", c="grey")
    axes[1].set_xlabel("steps since last SHOWN landmark glimpse")
    axes[1].set_ylabel("landmark surprisal (nats per tile)")
    axes[1].set_title("does surprisal collapse after identification?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="predictiveNet_state.pt; the run's provenance.json supplies the config")
    ap.add_argument("--out", default="docs/figures/surprisal_vs_time.png")
    ap.add_argument("--positions", type=int, nargs="+", default=None,
                    help="walk only these ROOMS_SELECTED positions of the run's set "
                         "(default: every room the run trained on)")
    args = ap.parse_args()

    run = load_checkpoint(args.ckpt, positions=None if args.positions is None else tuple(args.positions))
    print(f"{args.ckpt}: {run.cfg.arch_prnn.loss.value} loss, {run.cfg.arch_prnn.readout.value} "
          f"readout, action_offset {run.cfg.arch_prnn.action_offset}, "
          f"{len(run.layouts)} of {len(resolve_layouts(run.cfg))} rooms")
    result = measure(adapter=run.adapter, env=run.env, layouts=run.layouts)
    print(f"landmark by {TIME_BIN}-step bin: "
          + " ".join(f"{v:.2f}" for v in result.landmark_by_bin))
    print("background:                 "
          + " ".join(f"{v:.2f}" for v in result.background_by_bin))
    print(f"blind-step landmark mean {result.blind_step_mean:.3f} | "
          f"shown-step {result.shown_step_mean:.3f} | chance {result.chance:.3f}")
    print(f"landmark argmax recall shown {result.recall_shown:.3f} / masked "
          f"{result.recall_masked:.3f} | entirely missing shown "
          f"{result.miss_shown:.3f} / masked {result.miss_masked:.3f}")
    plot(result, out_path=args.out,
         title_note=f"{args.ckpt}, eval_mode, {len(run.layouts)} rooms "
                    f"x {EPISODES_PER_ROOM} episodes, fwd-biased walker")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
