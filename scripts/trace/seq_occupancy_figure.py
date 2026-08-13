"""Where the agent actually goes, phase by phase, through A -> B -> C -> REMOVED.

The spatial version of the behavioural result. `seq_behavior_figure.py` reports
percentiles at three chosen cells; this shows the whole occupancy map, so the
reader sees where the agent went rather than being asked to trust a matrix.

Rolls the trained policy at each phase-end checkpoint in that phase's own
environment, bins positions into the room's 14x14 grid, and averages over seeds.

Every panel shares ONE colour scale, because the question is a comparison
between phases and per-panel normalisation would answer a different one.

    uv run python scripts/trace/seq_occupancy_figure.py [n_trajs]

Writes outputs/summary/fig_seq_occupancy.png and caches the maps to
outputs/trace/seq_occupancy.npz so the figure re-renders without the rollouts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SEQ = [(7, 11), (7, 2), (4, 7), None]
LOCS = [(7, 11), (7, 2), (4, 7)]
TITLES = ["phase 0\nobject at (7,11)", "phase 1\nobject at (7,2)",
          "phase 2\nobject at (4,7)", "phase 3\nREMOVED"]
CACHE = Path("outputs/trace/seq_occupancy.npz")
OUT = Path("outputs/summary/fig_seq_occupancy.png")
PROBE_SEED = 7


def collect(n_trajs: int) -> np.ndarray:
    """(phase, seed, 14, 14) occupancy maps, normalised per rollout set."""
    from hydra import compose, initialize_config_dir
    from prnn.utils import ActionEncodingsEnum, AgentInputType

    from curious_george import AgentType, get_agent, get_pN, make_env
    from curious_george.rl.collect.format import get_obss_preprocessor
    from curious_george.storage import get_SR_acmodel
    from scripts.trace.trace_behavior import collect_policy_rollouts

    runs = sorted(Path("outputs/seq4").glob("OTC_seq_*/SEQ4-*"))
    runs = [r for r in runs if (r / "phase3_992").is_dir()]
    print(f"{len(runs)} seed runs x 4 phases x {n_trajs} rollouts")

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        args = compose(config_name="main")

    maps = np.zeros((4, len(runs), 14, 14))
    for si, run in enumerate(runs):
        for phase, loc in enumerate(SEQ):
            ck = run / f"phase{phase}_992"
            extra = {"new_obj_pos": loc} if loc else {}
            env = make_env(env_key="MiniGrid-LRoom-v0", input_type=AgentInputType.H_PO.value,
                           act_enc=ActionEncodingsEnum.SpeedHD.value, seed=0, **extra)
            pN = get_pN(args=args, env=env, device="cpu",
                        pRNN_ckpt=str(ck / "predictiveNet_state.pt"))
            pN.wandb_log = False
            obs_space, _ = get_obss_preprocessor(env.observation_space)
            ac = get_SR_acmodel(args, env.action_space, obs_space, torch.device("cpu"),
                                acmodel_status_ckpt=str(ck / "status.pt"))
            agent = get_agent(env=env, agent_Type=AgentType.AC, prnn=pN,
                              device=torch.device("cpu"), ac_model=ac, pastSR=True)
            pos, _ = collect_policy_rollouts(pN=pN, agent=agent, env=env,
                                             n_trajs=n_trajs, n_steps=256, seed=PROBE_SEED)
            flat = pos.reshape(-1, 2)
            # world cell (x, y) -> map[y-1, x-1]; interior runs 1..14
            m = np.zeros((14, 14))
            np.add.at(m, (np.clip(flat[:, 1] - 1, 0, 13), np.clip(flat[:, 0] - 1, 0, 13)), 1)
            maps[phase, si] = m / m.sum()
        print(f"  {run.parent.name}")
    return maps


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_trajs = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    if CACHE.is_file():
        maps = np.load(CACHE)["maps"]
        print(f"using cached {CACHE}")
    else:
        maps = collect(n_trajs)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, maps=maps)
        print(f"wrote {CACHE}")

    mean = maps.mean(axis=1)                       # (4, 14, 14)
    walls = mean.sum(axis=0) == 0                  # never visited by anyone: the L block
    shown = np.where(walls[None], np.nan, mean)
    vmax = float(np.nanpercentile(shown, 99.5))    # shared scale, robust to one hot cell

    fig = plt.figure(figsize=(20.5, 4.8))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, .07, .18, 1.05], wspace=.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax = fig.add_subplot(gs[0, 4])
    axes.append(fig.add_subplot(gs[0, 6]))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.82")

    for p in range(4):
        ax = axes[p]
        im = ax.imshow(shown[p], cmap=cmap, vmin=0, vmax=vmax, origin="upper")
        for l in LOCS:
            here = SEQ[p] == l
            ax.plot(l[0] - 1, l[1] - 1, marker="o", ms=15 if here else 9,
                    mfc="none", mew=2.6 if here else 1.4,
                    mec="#39FF14" if here else "white", zorder=5)
            if here:
                ax.annotate("object", (l[0] - 1, l[1] - 1), xytext=(0, 15),
                            textcoords="offset points", ha="center", color="#39FF14",
                            fontsize=9, fontweight="bold")
        ax.set_title(TITLES[p], fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, cax=cax).set_label("occupancy fraction (shared scale)", fontsize=9)

    # the departure panel: what CHANGES when the object is taken away
    ax = axes[4]
    diff = np.where(walls, np.nan, mean[2] - mean[3])
    lim = float(np.nanmax(np.abs(diff)))
    dm = plt.get_cmap("RdBu_r").copy(); dm.set_bad("0.82")
    im2 = ax.imshow(diff, cmap=dm, vmin=-lim, vmax=lim, origin="upper")
    ax.plot(LOCS[2][0] - 1, LOCS[2][1] - 1, marker="o", ms=15, mfc="none", mew=2.6,
            mec="#39FF14", zorder=5)
    ax.annotate("(4,7)", (LOCS[2][0] - 1, LOCS[2][1] - 1), xytext=(0, 15),
                textcoords="offset points", ha="center", color="#111", fontsize=9,
                fontweight="bold")
    ax.set_title("phase 2 − phase 3\nred = lost on removal, blue = gained", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.03).set_label("Δ occupancy", fontsize=9)

    fig.suptitle("Where the agent actually goes, phase by phase — occupancy pooled over 8 seeds "
                 "(grey = the L-room's wall block, never visited)", fontsize=13)
    fig.subplots_adjust(top=0.78, bottom=0.03, left=0.015, right=0.975)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
