"""Visual verification of the exploration eval stack. THROWAWAY.

Renders what tests/test_exploration_evals.py, tests/test_action_graph.py and
tests/test_exploration_online.py assert, so a human can eyeball the same
claims. No result may depend on this file. PNGs land in
throwaway/figs_exploration/.

    uv run python throwaway/verify_exploration_figs.py
"""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from curious_george.configs import PRESETS, RAND_ACT_PROBA, EnvBackend, EnvCfg, EvalKind
from curious_george.envs.action_graph import (
    ActionGraph,
    categorical_walk,
    layout_tables,
    spawn_states,
    sweeper_walk,
)
from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    EnvContent,
    LandmarkKind,
    RoomSetRules,
    Selected,
    Uniform,
    Vary,
    base_walkable,
    resolve_rooms,
)
from curious_george.evaluation.exploration import (
    coverage_curves,
    distance_binned_coverage,
    visitation_by_room,
)

OUT = Path(__file__).parent / "figs_exploration"
OUT.mkdir(exist_ok=True)
T = 256
BASE = base_walkable(BASE_ROOM_ID)


def committed_rooms(*, impassable: bool):
    entry = PRESETS["multienv-fast"]
    cfg = entry[1] if isinstance(entry, tuple) else entry
    env = dataclasses.replace(cfg.env, source=Selected(n=5, impassable=impassable))
    return resolve_rooms(
        shape=env.shape, content=env.content, source=env.source,
        room_rules=env.room_rules, set_rules=env.set_rules, indices=env.indices,
    )


def curves_for(table, walkable, probs, n, rng):
    spawns = spawn_states(walkable, n=n, rng=rng)
    pos = categorical_walk(table, probs=probs, spawns=spawns, steps=T, rng=rng)
    return coverage_curves(
        torch.from_numpy(pos),
        denominators=torch.full((n,), len(walkable)),
        width=table.shape[0], height=table.shape[1],
    )


def sweeper_curves(graph, walkable, n, rng):
    spawns = spawn_states(walkable, n=n, rng=rng)
    pos = np.concatenate([
        sweeper_walk(graph, walkable=walkable, spawn=tuple(s), steps=T) for s in spawns
    ])
    return coverage_curves(
        torch.from_numpy(pos),
        denominators=torch.full((n,), len(walkable)),
        width=graph.shape[0], height=graph.shape[1],
    ), spawns, pos


# ---------------------------------------------------------------- figure 1
def fig_coverage_curves():
    """Expected: sweeper crosses 0.9 before step ~200 in both arms; the two
    random walkers separate by ~2x; nothing exceeds 1.0."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, impassable in zip(axes, (False, True)):
        rooms = committed_rooms(impassable=impassable)
        walkable = rooms[0].walkable(BASE)
        table = layout_tables(layouts=rooms[:1])[0]
        graph = ActionGraph(table)
        steps = np.arange(1, T + 1)
        for label, probs, color in (
            ("uniform random [.25 x4]", [0.25] * 4, "tab:blue"),
            ("forward-weighted random (project default)", list(RAND_ACT_PROBA), "tab:orange"),
        ):
            c = curves_for(table, walkable, probs, 400, np.random.default_rng(0)).normalized.numpy()
            ax.plot(steps, c.mean(0), color=color, label=label)
            ax.fill_between(steps, np.quantile(c, 0.25, 0), np.quantile(c, 0.75, 0),
                            color=color, alpha=0.2)
        c = sweeper_curves(graph, walkable, 12, np.random.default_rng(1))[0].normalized.numpy()
        ax.plot(steps, c.mean(0), color="tab:green", label="greedy sweeper (positive control)")
        ax.fill_between(steps, np.quantile(c, 0.25, 0), np.quantile(c, 0.75, 0),
                        color="tab:green", alpha=0.2)
        for level in (0.5, 0.9):
            ax.axhline(level, color="grey", linestyle=":", linewidth=0.8)
            ax.text(3, level + 0.01, f"T{int(level * 100)} threshold", fontsize=8, color="grey")
        arm = "impassable landmarks" if impassable else "walkable landmarks"
        ax.set_title(f"{arm} — denominator {len(walkable)} cells")
        ax.set_xlabel("environment steps into the episode")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("fraction of the room's own walkable cells visited")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Coverage curves, committed room 0 — median band = interquartile range over episodes")
    fig.tight_layout()
    fig.savefig(OUT / "1_coverage_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_distance_maps():
    """Expected: same spawn, same colour scale. Right panel shows grey holes
    (unreachable landmark cells) and larger distances around them; the cell
    BEHIND the spawn arrow reads 3, the cell ahead reads 1."""
    spawn = None
    maps, rooms_used = [], []
    for impassable in (False, True):
        rooms = committed_rooms(impassable=impassable)
        walkable = rooms[0].walkable(BASE)
        table = layout_tables(layouts=rooms[:1])[0]
        if spawn is None:  # a cell open in both arms, facing +x
            x, y = next(c for c in sorted(walkable)
                        if (c[0] + 1, c[1]) in walkable and (c[0] - 1, c[1]) in walkable)
            spawn = (x, y, 0)
        maps.append(ActionGraph(table).distance_map(spawn=spawn))
        rooms_used.append(rooms[0])
    vmax = max(m.max() for m in maps)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, m, room, impassable in zip(axes, maps, rooms_used, (False, True)):
        masked = np.ma.masked_where(m < 0, m)
        im = ax.imshow(masked.T, vmin=0, vmax=vmax, cmap="viridis")
        ax.plot(spawn[0], spawn[1], marker="*", color="red", markersize=14,
                label="spawn (facing right)")
        for cx, cy in sorted(room.cells):
            ax.plot(cx, cy, marker="x", color="white", markersize=5, linestyle="none")
        arm = "impassable" if impassable else "walkable"
        ax.set_title(f"{arm} landmarks (x = landmark cell)\n"
                     f"ahead={m[spawn[0] + 1, spawn[1]]}  behind={m[spawn[0] - 1, spawn[1]]}")
        ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.85, label="shortest path from spawn, in ACTIONS (turns cost a step)")
    fig.suptitle("BFS distance maps, committed room 0, identical spawn and colour scale\n"
                 "grey = unreachable; expected: ahead=1, behind=3 in both panels")
    fig.savefig(OUT / "2_distance_maps.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_distance_binned():
    """Expected: both agents at 1.0 for distance 0; uniform random decays with
    distance while the sweeper stays ~1.0 - what aggregate coverage hides."""
    rooms = committed_rooms(impassable=True)
    walkable = rooms[0].walkable(BASE)
    table = layout_tables(layouts=rooms[:1])[0]
    graph = ActionGraph(table)
    rng = np.random.default_rng(4)
    spawns = spawn_states(walkable, n=64, rng=rng)
    dmaps = torch.stack([torch.from_numpy(graph.distance_map(spawn=tuple(s))) for s in spawns])

    uni = torch.from_numpy(categorical_walk(table, probs=[0.25] * 4, spawns=spawns, steps=T, rng=rng))
    swp = torch.from_numpy(np.concatenate([
        sweeper_walk(graph, walkable=walkable, spawn=tuple(s), steps=T) for s in spawns[:12]
    ]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, pos, dm, color in (
        ("uniform random [.25 x4], 64 episodes", uni, dmaps, "tab:blue"),
        ("greedy sweeper, 12 episodes", swp, dmaps[:12], "tab:green"),
    ):
        frac = distance_binned_coverage(pos, distance_maps=dm).pooled_fraction().numpy()
        ax.plot(np.arange(len(frac)), frac, marker="o", markersize=3, color=color, label=label)
    ax.set_xlabel("BFS distance from the episode's spawn, in actions")
    ax.set_ylabel("fraction of cells at that distance visited (pooled over episodes)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Hardness-aware coverage, impassable room 0\n"
                 "expected: both start at 1.0; only the sweeper stays there")
    fig.tight_layout()
    fig.savefig(OUT / "3_distance_binned_coverage.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_real_rollout_room_channel():
    """The WIRING, on a real training-stack collect (device backend, 3
    impassable rooms). Expected: each room's visit histogram avoids exactly
    that room's landmark cells (x markers), and visits-on-landmarks prints 0.
    Heatmaps share one colour scale because they are compared."""
    from curious_george.training.setup import setup_training
    from tests.small_config import small_config

    cfg = small_config(
        backend=EnvBackend.DEVICE, num_envs=32, episodes_per_env=1, episode_steps=T,
        env=EnvCfg(
            content=EnvContent(kinds=tuple(LandmarkKind(s, impassable=True) for s in ("x", "plus", "block3"))),
            source=Uniform(n=3, seed=7),
            set_rules=RoomSetRules(varies=frozenset({Vary.POSITION})),
        ),
        evals=frozenset({EvalKind.SPATIAL_MULTIROOM}),
    )
    comps = setup_training(cfg)
    algo = comps.algo
    positions, layout_ids = [], []
    for _ in range(4):
        algo.collect_experiences()
        positions.append(algo.positions_episodes.cpu())
        layout_ids.append(torch.from_numpy(algo.segment_layouts.reshape(-1)))
    positions, layout_ids = torch.cat(positions), torch.cat(layout_ids)

    visitation = visitation_by_room(
        positions, layout_ids=layout_ids, supports=algo.room_supports.cpu()
    )
    counts = visitation.counts.numpy()
    on_landmarks = sum(
        counts[r][cx, cy]
        for r, room in enumerate(comps.envs.layouts)
        for cx, cy in room.cells
    )
    vmax = counts.max()
    fig, axes = plt.subplots(1, len(comps.envs.layouts), figsize=(4.2 * len(comps.envs.layouts), 4.4))
    for r, (ax, room) in enumerate(zip(axes, comps.envs.layouts)):
        im = ax.imshow(counts[r].T, vmin=0, vmax=vmax, cmap="magma")
        for cx, cy in sorted(room.cells):
            ax.plot(cx, cy, marker="x", color="cyan", markersize=6, linestyle="none")
        n_eps = int((layout_ids == r).sum())
        ax.set_title(f"room {r} — {n_eps} episodes\n"
                     f"normalized visitation entropy {float(visitation.normalized[r]):.3f}")
    fig.colorbar(im, ax=axes, shrink=0.8, label="visits (shared scale across rooms)")
    fig.suptitle(f"REAL device-backend rollouts, untrained policy, 128 episodes x {T} steps\n"
                 f"x = that room's impassable landmark cells; visits landing on them: {int(on_landmarks)} (must be 0)")
    fig.savefig(OUT / "4_real_rollout_room_channel.png", dpi=150)
    plt.close(fig)
    print(f"visits on impassable landmark cells across all rooms: {int(on_landmarks)} (must be 0)")


if __name__ == "__main__":
    fig_coverage_curves()
    fig_distance_maps()
    fig_distance_binned()
    fig_real_rollout_room_channel()
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p)
