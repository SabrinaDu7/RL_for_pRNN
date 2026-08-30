"""Does the pRNN's hidden state carry the ROOM'S SHAPE? Isomap says so or it does not.

The question sRSA was supposed to answer, asked in a way that does not route
through sRSA. sRSA scores a correlation between two similarity matrices and
returns one number; this embeds the hidden state directly and lets the L-room
either appear or not. If `h` is a place code, a 2-D Isomap of theta-mean hidden
activity, coloured by the agent's TRUE position, reproduces the L.

Isomap and not PCA: a place-code manifold can be curved in the 500-D hidden
space, and PCA only finds a linear projection of it. Isomap preserves geodesic
distance along the manifold, so a curved sheet unfolds into the shape it is a
sheet OF. That is the whole reason to use it here.

THE SETTINGS COME FROM THE prnn REPO, NOT FROM THIS FILE. `metric='cosine'` is
`representationalGeometryAnalysis.defaultMetric`, and it is what
`evaluation/spatial.py` already scores hidden activity with through
`RGA.calculateSleepWakeDist(..., metric="cosine")`; euclidean distance on `h`
conflates firing-RATE magnitude with firing PATTERN. `n_components=3` is what
`prnn/utils/figures.py:451` embeds into. Fitting 2 components instead returns a
RING for both circuits - a 2-D sheet curved in 3-D cannot be flattened without
one, which is an artefact of the component count and not a fact about the map.

THE AGENT. `--agent policy` (the default) runs the run's OWN trained policy with
everything in eval: `acmodel.eval()`, `pN.eval()` - which is what disables the
0.15 input dropout - and `argmax=True`, so actions are the distribution's peak
rather than a sample. That is the policy bare, with no stochasticity of its own.
`with_CV` is untouched by any of this: it is a CONSTRUCTOR argument read from
`arch_policy.with_obs` (False in these runs), not a mode flag.

`--agent random` is kept as the CONTROL, and reading both matters. A trained
policy's coverage differs between arms - `action_offset=1` collapses transiently
- so an on-policy embedding confounds "the map changed" with "the agent stopped
visiting the room". The random walker's coverage is identical in every arm. The
script therefore reports DISTINCT CELLS VISITED for whatever agent it ran, so
the confound is a number on the page rather than an assumption.

ONE trajectory, replayed through every checkpoint. Observations depend on the
room and the actions, never on the network, so a single collection is valid for
all arms - which is what makes two checkpoints comparable rather than each
carrying its own rollout noise (`evaluation/checkpoint_series.fixed_probe`).
The actions are then re-encoded PER ARM through that arm's own `PRNNAdapter`,
because `action_offset` changes which action shares a row with `obs[t]`.

CONTROLS, both computed from the same pipeline as the measurement:
  negative  positions shuffled  -> distance correlation must fall to ~0. If it
            does not, the statistic is reading something other than space.
  positive  the ROOM'S OWN walkable cells pushed through the identical Isomap
            and scoring path -> the ceiling this estimator can reach on this
            geometry. Computed on the room rather than on the trajectory's
            positions, because a low-coverage trajectory repeats a few cells
            thousands of times and Isomap on duplicated points degenerates -
            which produced a "ceiling" BELOW the measurement it was supposed to
            bound. The measurement is read as a fraction of this, not of 1.0.

    uv run python throwaway/scripts/isomap_hidden_states.py \\
        outputs/offset0-parity_curious_26-08-29-02-10-39 \\
        outputs/offset1-parity_curious_26-08-29-02-35-59

Writes outputs/figures/isomap_hidden_states.png and .json beside it. Throwaway:
no committed result may depend on this file (see CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

#: The probe trajectory's action distribution and seed, taken from
#: `evaluation.circuit_diagnostics` so this script does not invent a third
#: sampling convention. Forward-weighted: a rollout of pure turns visits no
#: cells and would say nothing about spatial structure.
from curious_george.evaluation.circuit_diagnostics import PROBE_ACTION_P, PROBE_SEED

#: Mutable so `--probe-seed` can re-roll the TRAJECTORY. Isomap itself has no
#: seed - it is an eigendecomposition of a fixed neighbourhood graph - so the
#: only stochastic input to this whole script is which cells the walker visited.
PROBE = {"seed": PROBE_SEED}

ONSET = 20  # matches evaluation/checkpoint_series.ONSET: drop the startup transient


def _nullcontext():
    from contextlib import nullcontext

    return nullcontext()


@dataclass(frozen=True)
class Fidelity:
    """How well a 2-D embedding of `h` reproduces the room's metric structure."""

    #: Spearman(pairwise distance in the 2-D embedding, pairwise TRUE distance).
    embedding_vs_space: float
    #: The same, computed in the RAW hidden space. Says whether Isomap found
    #: structure that was already there or manufactured it.
    hidden_vs_space: float
    #: sklearn `trustworthiness` of the embedding w.r.t. the hidden space: did
    #: the projection to 2-D preserve neighbourhoods, independent of any room.
    trustworthiness: float
    #: Negative control - the same statistic with positions shuffled.
    shuffled_control: float
    #: `embedding_vs_space` recomputed WITHIN each head direction and averaged.
    #: Head direction is fed into every input row and is cyclic with four
    #: values, so it can produce a ring in the embedding and a high pooled
    #: correlation without any place code at all. Conditioning on it removes
    #: that path: what survives here is spatial structure and nothing else.
    embedding_vs_space_within_head_direction: float
    #: Head direction decoded from `h`, held-out BALANCED accuracy.
    head_direction_accuracy: float
    head_direction_chance: float
    n_points: int


def run_action_offset(run_dir: Path) -> int:
    """The circuit the run was LAUNCHED with, read from its own provenance.

    Not inferred from the directory name: `offset1-parity` is a label a human
    typed, and provenance.json records the actual argv.
    """
    argv = json.loads((run_dir / "provenance.json").read_text())["argv"]
    for flag in ("--arch-prnn.action-offset", "--arch_prnn.action_offset"):
        if flag in argv:
            return int(argv[argv.index(flag) + 1])
    return 0  # ArchPrnnCfg.action_offset's default


def latest_checkpoint(run_dir: Path, step: int | None) -> tuple[int, Path]:
    from curious_george.evaluation.checkpoint_series import archived

    points = archived(run_dir)
    if not points:
        raise SystemExit(f"no archived checkpoints under {run_dir / 'checkpoints'}")
    if step is None:
        return points[-1]
    match = [p for p in points if p[0] == step]
    if not match:
        raise SystemExit(
            f"{run_dir.name} has no checkpoint at step {step}; "
            f"available: {[s for s, _ in points]}"
        )
    return match[0]


def build_agent(*, kind: str, run_dir: Path, step: int, pN, cfg, env):
    """The run's own policy in full eval, or the seeded random control."""
    import torch as _torch

    from curious_george.evaluation.checkpoint_series import archived_policies
    from curious_george.log_and_store.storage import get_SR_acmodel, get_agent
    from curious_george.rl.collect.format import get_obss_preprocessor
    from curious_george.utils.enums import AgentType

    if kind == "random":
        return get_agent(env=env, agent_Type=AgentType.RANDOM,
                         rand_act_prob=np.array(PROBE_ACTION_P)), None
    policies = archived_policies(run_dir)
    if step not in policies:
        raise SystemExit(
            f"{run_dir.name} has no archived policy at step {step:,} - runs "
            f"finished before 2026-08-28 kept only a rolling policy.pt. "
            f"Available: {sorted(policies)}"
        )
    obs_space, _ = get_obss_preprocessor(env.observation_space)
    acmodel = get_SR_acmodel(cfg, env.action_space, obs_space,
                             _torch.device("cpu"), str(policies[step]))
    agent = get_agent(env=env, agent_Type=AgentType.AC, prnn=pN,
                      device=_torch.device("cpu"), ac_model=acmodel,
                      argmax=True, pastSR=cfg.arch_prnn.action_offset == 0)
    return agent, acmodel


def make_probe_env(hidden_size: int):
    """The default L-room, exactly as the A/B arms trained in it."""
    from prnn.utils import ActionEncodingsEnum, MinigridEnvNames

    from curious_george import AgentInputType, make_env

    return make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value,
        seed=PROBE['seed'],
    )


def collect_probe(*, env, agent, extra_eval, n_segments: int, steps: int):
    """Raw (observations, actions, final observation, positions) per segment.

    Mirrors `circuit_diagnostics.collect_segments`, which cannot be called
    directly because it does not return `agent_pos` - and position is the whole
    dependent variable here. Kept circuit-AGNOSTIC for the same reason it is
    there: the encoding is the thing that differs between arms, so it happens
    later, per arm.
    """
    from curious_george.models.device import eval_mode

    torch.manual_seed(PROBE["seed"])
    np.random.seed(PROBE["seed"])
    env.env.reset(seed=PROBE["seed"])
    segments = []
    modules = [agent.prnn] if hasattr(agent, "prnn") else []
    modules += [extra_eval] if extra_eval is not None else []
    with eval_mode(modules) if modules else _nullcontext():
        for _ in range(n_segments):
            obs, act, state, _ = agent.getObservations(env, steps)
            segments.append((
                obs[:-1], np.asarray(act).reshape(-1), obs[-1],
                np.asarray(state["agent_pos"], dtype=float),
                np.asarray(state["agent_dir"]).reshape(-1).astype(int),
            ))
    return segments


def hidden_activity(*, pN, adapter, segments):
    """Theta-mean hidden activity and the position each row was recorded AT.

    Row t of the pRNN's input carries obs[t] under BOTH circuits - only the
    action sharing that row moves - so row t pairs with position[t] either way.
    The offset changes the row COUNT (offset 1 keeps the segment's last action
    instead of discarding it), which is why the pairing truncates to the shorter
    of the two rather than assuming a length.
    """
    from curious_george.models.device import eval_mode

    h_rows, pos_rows, hd_rows = [], [], []
    with eval_mode(pN.pRNN), torch.no_grad():
        for obss, acts, last, pos, hd in segments:
            obs_f, act_f = adapter.seq2pred(
                *adapter.reward_pass_inputs(obss, acts, last, 1)
            )
            # Seeded immediately before the forward, and started from a FIXED
            # zero state, so every arm sees one realisation of the injected
            # noise rather than its own draw. The noise itself stays on: it is
            # the model's dynamics, not an artefact (evaluation/spatial.py).
            torch.manual_seed(PROBE['seed'])
            _, _, h = pN.predict(
                obs_f, act_f,
                state=torch.zeros((1, 1, pN.hidden_size), device=obs_f.device),
            )
            h_mean = torch.mean(h, dim=0)  # theta mean, as evaluation/spatial.py does
            n = min(len(h_mean), len(pos))
            h_rows.append(h_mean[ONSET:n].cpu().numpy())
            pos_rows.append(pos[ONSET:n])
            hd_rows.append(hd[ONSET:n])
    return (np.concatenate(h_rows), np.concatenate(pos_rows),
            np.concatenate(hd_rows))


def _distance_spearman(a, b, *, rng, max_points: int, metric_a: str = "euclidean") -> float:
    """Spearman between two point sets' pairwise distances, on shared indices.

    `metric_a` is cosine when `a` is raw hidden activity and euclidean when it
    is an Isomap embedding - the embedding is a euclidean coordinate space by
    construction, whatever metric was used to build the neighbourhood graph.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    idx = (rng.choice(len(a), max_points, replace=False)
           if len(a) > max_points else np.arange(len(a)))
    return float(spearmanr(pdist(a[idx], metric=metric_a), pdist(b[idx])).statistic)


def _head_direction_decodable(h, hd, *, seed: int) -> tuple[float, float]:
    """Held-out balanced accuracy of a linear head-direction readout from `h`.

    NaN when only one head direction is present - which is the case for the
    positive control, whose points are room CELLS and carry no head direction.
    A one-class fit is not a chance-level result, so it is not reported as one.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    if len(np.unique(hd)) < 2:
        return (float("nan"), float("nan"))
    X_tr, X_te, y_tr, y_te = train_test_split(
        h, hd, test_size=0.3, random_state=seed, stratify=hd)
    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    chance = balanced_accuracy_score(
        y_te, np.full_like(y_te, np.bincount(y_tr).argmax()))
    return (float(balanced_accuracy_score(y_te, clf.predict(X_te))), float(chance))


def measure(*, h, pos, hd, n_neighbors: int, n_components: int, metric: str,
            max_points: int, seed: int = 0):
    """Embed `h` and score how well that embedding reproduces the room."""
    from sklearn.manifold import Isomap, trustworthiness

    emb = Isomap(n_neighbors=n_neighbors, n_components=n_components,
                 metric=metric).fit_transform(h)
    shuffled = pos[np.random.default_rng(seed + 1).permutation(len(pos))]
    within = [
        _distance_spearman(emb[m], pos[m], rng=np.random.default_rng(seed),
                           max_points=max_points)
        for d in np.unique(hd)
        # a head direction with too few rows gives a meaningless correlation
        if (m := hd == d).sum() >= 50
    ]
    acc, chance = _head_direction_decodable(h, hd, seed=seed)
    return emb, Fidelity(
        embedding_vs_space=_distance_spearman(
            emb, pos, rng=np.random.default_rng(seed), max_points=max_points),
        hidden_vs_space=_distance_spearman(
            h, pos, rng=np.random.default_rng(seed), max_points=max_points,
            metric_a=metric),
        trustworthiness=float(
            trustworthiness(h, emb, n_neighbors=n_neighbors, metric=metric)),
        shuffled_control=_distance_spearman(
            emb, shuffled, rng=np.random.default_rng(seed), max_points=max_points),
        embedding_vs_space_within_head_direction=float(np.mean(within)),
        head_direction_accuracy=acc,
        head_direction_chance=chance,
        n_points=int(len(h)),
    )


def position_colors(pos, *, lo, hi):
    """ONE fixed position -> colour mapping, shared by every panel in the figure.

    Red tracks the x axis and blue tracks the y axis, so a colour names a cell
    and names the SAME cell in the reference panel and in every embedding. This
    is the calibration that makes the panels comparable; per-panel autoscaling
    would let two different rooms look identical.
    """
    n = (pos - lo) / np.maximum(hi - lo, 1e-9)
    return np.stack([n[:, 0], np.full(len(n), 0.45), n[:, 1]], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="training run directories")
    ap.add_argument("--step", type=int, default=None,
                    help="environment step to score; default is each run's latest")
    ap.add_argument("--n-segments", type=int, default=8,
                    help="probe trajectories (matches EvalCfg.n_trajs)")
    ap.add_argument("--steps", type=int, default=256,
                    help="steps per trajectory (matches collect.episode_steps)")
    ap.add_argument("--n-neighbors", type=int, default=50,
                    help="Isomap neighbourhood; prnn/utils/figures.py uses 50, "
                         "RGA.fitIsomap defaults to 150, BasicAnalysis.py to 15")
    ap.add_argument("--n-components", type=int, default=3,
                    help="prnn/utils/figures.py:451 embeds into 3; at 2 both "
                         "circuits return a ring")
    ap.add_argument("--metric", default="cosine",
                    help="RGA.defaultMetric, and what SWdist already scores h with")
    ap.add_argument("--max-points", type=int, default=900,
                    help="points subsampled for the pairwise-distance statistics")
    ap.add_argument("--agent", default="policy", choices=("policy", "random"),
                    help="policy = the run's own network, fully in eval "
                         "(acmodel.eval, pN.eval, argmax); random = the "
                         "coverage-matched control")
    ap.add_argument("--probe-seed", type=int, default=PROBE_SEED,
                    help="re-rolls the TRAJECTORY, the script's only stochastic "
                         "input; Isomap itself is deterministic given the points")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/figures/isomap_hidden_states.png"))
    a = ap.parse_args()

    PROBE["seed"] = a.probe_seed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from curious_george import get_pN
    from curious_george.configs import Config
    from curious_george.envs.layouts import BASE_ROOM_ID, base_walkable
    from curious_george.evaluation.checkpoint_series import checkpoint_hiddensize
    from curious_george.models.prnn_adapter import PRNNAdapter
    from dataclasses import replace

    walkable = np.array(sorted(base_walkable(BASE_ROOM_ID)), dtype=float)
    lo, hi = walkable.min(axis=0), walkable.max(axis=0)
    print(f"L-room: {len(walkable)} walkable cells, "
          f"x {lo[0]:.0f}-{hi[0]:.0f}, y {lo[1]:.0f}-{hi[1]:.0f}")

    results, embeddings = [], []
    for run_dir in a.runs:
        step, ckpt = latest_checkpoint(run_dir, a.step)
        offset = run_action_offset(run_dir)
        hidden = checkpoint_hiddensize(ckpt)
        env = make_probe_env(hidden)
        cfg = replace(Config(), arch_prnn=replace(Config().arch_prnn,
                                                  hidden_size=hidden,
                                                  action_offset=offset))
        pN = get_pN(args=cfg, env=env, device="cpu", pRNN_ckpt=str(ckpt))
        pN.wandb_log = False
        adapter = PRNNAdapter(pN, torch.device("cpu"), pastSR=offset == 0)
        assert adapter.action_offset == offset, "adapter and run disagree on the circuit"

        agent, acmodel = build_agent(kind=a.agent, run_dir=run_dir, step=step,
                                     pN=pN, cfg=cfg, env=env)
        segments = collect_probe(env=env, agent=agent, extra_eval=acmodel,
                                 n_segments=a.n_segments, steps=a.steps)
        h, pos, hd = hidden_activity(pN=pN, adapter=adapter, segments=segments)
        emb, fid = measure(h=h, pos=pos, hd=hd, n_neighbors=a.n_neighbors,
                           n_components=a.n_components, metric=a.metric,
                           max_points=a.max_points)
        cells = len({tuple(p) for p in pos})
        label = f"action_offset={offset}  {run_dir.name}"
        print(f"\n{label}  @ {step:,} env steps  ({fid.n_points} rows)")
        print(f"   distinct cells visited      {cells}/{len(walkable)}"
              f"   <- coverage; NOT matched across arms when --agent policy")
        print(f"   embedding vs space          {fid.embedding_vs_space:+.4f}")
        print(f"   ... WITHIN head direction   "
              f"{fid.embedding_vs_space_within_head_direction:+.4f}   <- the place code")
        print(f"   hidden    vs space          {fid.hidden_vs_space:+.4f}")
        print(f"   trustworthiness             {fid.trustworthiness:.4f}")
        print(f"   shuffled control            {fid.shuffled_control:+.4f}  (must be ~0)")
        print(f"   head direction from h       {fid.head_direction_accuracy:.4f} "
              f"(chance {fid.head_direction_chance:.4f})")
        results.append({"run": run_dir.name, "step": step, "action_offset": offset,
                        "agent": a.agent, "distinct_cells": cells,
                        "walkable_cells": len(walkable), **asdict(fid)})
        embeddings.append((label, emb, pos, hd))

    # POSITIVE CONTROL: the true positions through the identical pipeline. The
    # ceiling the estimator can reach on this many points - the measurements
    # above are read as a fraction of THIS, never of 1.0.
    ctrl_emb, ctrl = measure(
        h=walkable, pos=walkable, hd=np.zeros(len(walkable), dtype=int),
        n_neighbors=min(a.n_neighbors, len(walkable) // 4), n_components=2,
        metric="euclidean", max_points=a.max_points)
    print(f"\npositive control (the room's {len(walkable)} walkable cells through "
          f"the same pipeline): embedding vs space {ctrl.embedding_vs_space:+.4f}")

    # TWO rows, because the position panel alone cannot tell a place code from a
    # HEAD-DIRECTION code. HD is fed into every input row, it is cyclic with four
    # values, and a cyclic variable dominating the hidden state's variance embeds
    # as a ring. Colouring the SAME embedding by HD settles which it is.
    HD_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    HD_NAMES = ["right (+x)", "down (+y)", "left (-x)", "up (-y)"]
    ncol = len(embeddings) + 1
    fig = plt.figure(figsize=(5.4 * ncol, 10.8))
    fig.suptitle(
        "What does the pRNN's hidden state embed: the room, or the head direction?\n"
        f"Isomap n_components={a.n_components}, metric={a.metric}, "
        f"n_neighbors={a.n_neighbors} (prnn repo conventions); agent={a.agent}"
        + (" in FULL EVAL (acmodel.eval, pN.eval, argmax)"
           if a.agent == "policy" else " (coverage matched across arms)")
        + " - identical embedding in both rows, only the COLOUR VARIABLE changes",
        fontsize=11)

    ax = fig.add_subplot(2, ncol, 1)
    ax.scatter(walkable[:, 0], walkable[:, 1],
               c=position_colors(walkable, lo=lo, hi=hi), s=48, marker="s")
    ax.set_title("REFERENCE: the L-room itself\n(true coordinates)", fontsize=10)
    ax.set_xlabel("room x"); ax.set_ylabel("room y")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", color=(1.0, 0.45, 0.0), label="high x, low y"),
        Line2D([], [], marker="s", ls="", color=(0.0, 0.45, 1.0), label="low x, high y"),
        Line2D([], [], marker="s", ls="", color=(1.0, 0.45, 1.0), label="high x, high y"),
        Line2D([], [], marker="s", ls="", color=(0.0, 0.45, 0.0), label="low x, low y"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.14), frameon=False, fontsize=8,
       ncol=2, title="COLOUR = position (row 1)", title_fontsize=8)

    ax = fig.add_subplot(2, ncol, ncol + 1)
    ax.axis("off")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=n)
                       for c, n in zip(HD_COLORS, HD_NAMES)],
              loc="center", frameon=False, fontsize=11,
              title="COLOUR = head direction\n(row 2)", title_fontsize=11)

    def scatter3(idx, emb, colors, title):
        ax = fig.add_subplot(2, ncol, idx, projection="3d")
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=colors, s=5, alpha=0.7)
        ax.set_xlabel("Isomap 1", fontsize=8)
        ax.set_ylabel("Isomap 2", fontsize=8)
        ax.set_zlabel("Isomap 3", fontsize=8)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticklabels([])
        ax.set_title(title, fontsize=9)

    for i, (label, emb, pos, hd) in enumerate(embeddings):
        fid = results[i]
        scatter3(i + 2, emb, position_colors(pos, lo=lo, hi=hi),
                 f"{label}\ncoloured by POSITION\n"
                 f"{fid['distinct_cells']}/{fid['walkable_cells']} cells visited\n"
                 f"embedding vs space {fid['embedding_vs_space']:+.3f}   "
                 f"within head direction "
                 f"{fid['embedding_vs_space_within_head_direction']:+.3f}\n"
                 f"(shuffled {fid['shuffled_control']:+.3f}; ceiling "
                 f"{ctrl.embedding_vs_space:+.3f})")
        scatter3(ncol + i + 2, emb, [HD_COLORS[d] for d in hd],
                 f"{label}\ncoloured by HEAD DIRECTION - same points, same axes\n"
                 f"head direction decodable from h "
                 f"{fid['head_direction_accuracy']:.3f} "
                 f"(chance {fid['head_direction_chance']:.3f})")

    fig.tight_layout(rect=(0, 0.0, 1, 0.93))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    payload = {
        "probe": {"n_segments": a.n_segments, "steps": a.steps, "onset": ONSET,
                  "seed": a.probe_seed, "action_p": list(PROBE_ACTION_P),
                  "agent": a.agent},
        "isomap": {"n_neighbors": a.n_neighbors, "n_components": a.n_components,
                   "metric": a.metric, "max_points": a.max_points,
                   "settings_from": "prnn: RGA.defaultMetric='cosine', "
                                    "figures.py:451 n_components=3"},
        "positive_control": asdict(ctrl),
        "runs": results,
    }
    a.out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {a.out}\nwrote {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
