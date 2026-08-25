"""The four sequential-displacement environments, with what each one cost.

Figure 3.4 shows where the agent went in each phase and Figure 3.1 shows how the
maps changed; neither shows the ROOMS, or how much training each phase actually
received. Without that the §3 null cannot be read against §1's, because the two
designs give a location different amounts of exposure.

Every number here is DERIVED, never typed in:

    num_batches  from the run's own phase directories - `train` saves the final
                 checkpoint as (num_batches - 1) * trajs_per_batch, so the
                 directory label recovers the batch count
    the rest     from the composed hydra config (rl.trajs_per_batch, rl.frames,
                 rl.ppo_epochs, rl.ppo_batch_size, predNet.seqdur)
    the sequence from seq_occupancy_figure.SEQ, so this figure and Figure 3.4
                 cannot disagree about which phase had the object where

The world-model count is one gradient step PER EPISODE SEGMENT, not per batch:
these runs have `predNet.batched_wm = False`, and
`curious_george/rl/update/world_model.py` steps once per segment.

    uv run python scripts/trace/seq_phase_figure.py [run_dir]

Writes outputs/summary/fig_seq_phases.png.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.trace.seq_occupancy_figure import SEQ

DEFAULT_RUN = Path("outputs/seq4/OTC_seq_5200_10340479/"
                   "SEQ4-otc-p0.5-fixedpos-c1-0811-123058")
OUT = Path("outputs/summary/fig_seq_phases.png")
ENV_ID = "MiniGrid-LRoom-v0"
TILE = 16
AGENT_CELL = (1, 1)   # fixed in every panel, so the object is the ONLY difference


@dataclass(frozen=True)
class PhaseBudget:
    """What one displacement phase cost, derived from the run and its config."""

    batches: int
    trajs_per_batch: int
    seqdur: int
    frames: int
    ppo_epochs: int
    ppo_batch_size: int

    @property
    def trajectories(self) -> int:
        return self.batches * self.trajs_per_batch

    @property
    def env_steps(self) -> int:
        return self.trajectories * self.seqdur

    @property
    def world_model_steps(self) -> int:
        """batched_wm=False: one step per episode segment, frames/seqdur per batch."""
        return self.batches * (self.frames // self.seqdur)

    @property
    def policy_steps(self) -> int:
        return self.batches * self.ppo_epochs * (self.frames // self.ppo_batch_size)


def budget(run: Path) -> PhaseBudget:
    """Recover the per-phase budget from the run directory plus the config."""
    from hydra import compose, initialize_config_dir

    labels = {int(d.name.split("_")[1]) for d in run.iterdir()
              if d.is_dir() and d.name.startswith("phase")}
    if not labels:
        raise SystemExit(f"{run} has no phase* directories")

    with initialize_config_dir(config_dir=str(Path("Configs").resolve()), version_base=None):
        cfg = compose(config_name="main")
    tpb, seqdur = int(cfg.rl.trajs_per_batch), int(cfg.predNet.seqdur)
    if int(cfg.rl.frames) != tpb * seqdur:
        raise SystemExit(
            f"rl.frames={int(cfg.rl.frames)} is not trajs_per_batch*seqdur={tpb * seqdur}; "
            "the per-batch trajectory count this figure derives would be wrong")
    # `train` writes the last checkpoint at (num_batches - 1) * trajs_per_batch
    return PhaseBudget(batches=max(labels) // tpb + 1, trajs_per_batch=tpb, seqdur=seqdur,
                       frames=int(cfg.rl.frames), ppo_epochs=int(cfg.rl.ppo_epochs),
                       ppo_batch_size=int(cfg.rl.ppo_batch_size))


def render(loc) -> np.ndarray:
    """Top-down view of the phase's room, agent pinned so only the object moves."""
    import gymnasium as gym

    extra = {"new_obj_pos": tuple(loc)} if loc else {}
    env = gym.make(ENV_ID, agent_start_pos=AGENT_CELL, agent_start_dir=0, **extra)
    env.reset(seed=0)
    return env.unwrapped.get_frame(highlight=False, tile_size=TILE)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN
    b = budget(run)
    print(f"{run.name}: {len(SEQ)} phases x {b.batches} batches "
          f"({b.trajs_per_batch} trajectories of {b.seqdur} steps each)")

    fig, axes = plt.subplots(1, len(SEQ), figsize=(3.5 * len(SEQ), 4.9))
    names = [f"phase {i}" for i in range(len(SEQ))]
    for ax, name, loc in zip(axes, names, SEQ):
        ax.imshow(render(loc), interpolation="nearest")
        head = f"{name}\nobject at {tuple(loc)}" if loc else f"{name}\nobject REMOVED"
        ax.set_title(head, fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(
            f"{b.trajectories:,} trajectories  ({b.batches} batches x {b.trajs_per_batch})\n"
            f"{b.env_steps:,} environment steps\n"
            f"{b.world_model_steps:,} world-model gradient steps\n"
            f"{b.policy_steps:,} PPO gradient steps",
            fontsize=11, labelpad=10, linespacing=1.6)

    fig.suptitle(
        "The sequential-displacement environments, and what each phase cost.\n"
        "Every phase is identical apart from the object (the single bright-green tile); the agent "
        f"is pinned to {AGENT_CELL} in all four panels so the object is the only difference.",
        fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
