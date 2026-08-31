"""Known baselines with known answers, before any training run is scored.

Two kinds of gate. Analytic: agents whose metrics are exactly derivable - a
turn-left-only walker visits one cell (entropy 0, coverage floor), a forward-
only walker traces a ray, enumerating a room's support hits every ceiling at
exactly 1.0. Statistical: the calibration table - uniform random, the
project's forward-weighted random, and the greedy sweeper - re-derived through
the REAL transition tables and pinned, superseding the throwaway calibration
script `exploration_baseline_calibration.py` (deleted 2026-08-30, in git
history; its independently hand-rolled dynamics produced the same numbers to
three decimals, which is what makes these pins trustworthy).
"""

import dataclasses
import math

import numpy as np
import pytest
import torch

from curious_george.configs import PRESETS
from curious_george.envs.action_graph import (
    ActionGraph,
    FORWARD,
    categorical_walk,
    layout_tables,
    spawn_states,
    sweeper_walk,
    walk,
)
from curious_george.envs.layouts import (
    BASE_ROOM_ID,
    Selected,
    base_walkable,
    pooled_walkable,
    resolve_rooms,
)
from curious_george.evaluation.exploration import (
    coverage_curves,
    distance_binned_coverage,
    visitation_by_room,
)
from curious_george.log_and_store.storage import RAND_ACT_PROBA

BASE = base_walkable(BASE_ROOM_ID)
T = 256


def _rooms(*, impassable: bool):
    entry = PRESETS["multienv-fast"]
    cfg = entry[1] if isinstance(entry, tuple) else entry
    env = dataclasses.replace(cfg.env, source=Selected(n=5, impassable=impassable))
    return resolve_rooms(
        shape=env.shape, content=env.content, source=env.source,
        room_rules=env.room_rules, set_rules=env.set_rules, indices=env.indices,
    )


@pytest.fixture(scope="module")
def arms():
    """(rooms, room0 walkable set, room0 table) per affordance arm."""
    out = {}
    for impassable in (False, True):
        rooms = _rooms(impassable=impassable)
        out[impassable] = (
            rooms,
            rooms[0].walkable(BASE),
            layout_tables(layouts=rooms[:1])[0],
        )
    return out


def _grid_shape(table) -> tuple[int, int]:
    return table.shape[0], table.shape[1]


def _curves(positions_np, *, walkable, table):
    W, H = _grid_shape(table)
    positions = torch.from_numpy(positions_np)
    return coverage_curves(
        positions,
        denominators=torch.full((len(positions),), len(walkable)),
        width=W,
        height=H,
    )


def _support(walkable, *, table) -> torch.Tensor:
    W, H = _grid_shape(table)
    support = torch.zeros((1, W, H), dtype=torch.bool)
    for x, y in walkable:
        support[0, x, y] = True
    return support


# --- structural gates: the curve is the measurement --------------------------


def test_curves_are_monotone_and_end_at_the_distinct_count(arms):
    _, walkable, table = arms[True]
    rng = np.random.default_rng(7)
    spawns = spawn_states(walkable, n=64, rng=rng)
    positions = categorical_walk(table, probs=[0.25] * 4, spawns=spawns, steps=T, rng=rng)
    curves = _curves(positions, walkable=walkable, table=table).curves
    assert (curves.diff(dim=1) >= 0).all()
    for row, pos in zip(curves, positions):
        assert row[-1].item() == len({tuple(p) for p in pos})


# --- analytic agents ---------------------------------------------------------


def test_turn_left_only_scores_the_floor_on_every_metric(arms):
    """The user-specified sanity agent: one visited cell, so location entropy
    is exactly 0, the coverage curve is flat at 1, every threshold is
    censored, and only the distance-0 bin is covered."""
    rooms, walkable, table = arms[True]
    spawn = next(iter(sorted(walkable)))
    positions = walk(
        table, spawns=np.array([(*spawn, 0)]), actions=np.zeros((1, T), dtype=np.int64)
    )
    curves = _curves(positions, walkable=walkable, table=table)
    assert (curves.curves == 1).all()
    assert curves.nauc.item() == pytest.approx(1 / len(walkable))
    assert curves.threshold(0.25).reached_fraction == 0.0

    visitation = visitation_by_room(
        torch.from_numpy(positions),
        layout_ids=torch.zeros(1, dtype=torch.long),
        supports=_support(walkable, table=table),
    )
    assert visitation.entropy_bits.item() == pytest.approx(0.0)
    assert visitation.normalized.item() == pytest.approx(0.0)

    dist = ActionGraph(table).distance_map(spawn=(*spawn, 0))
    binned = distance_binned_coverage(
        torch.from_numpy(positions), distance_maps=torch.from_numpy(dist)[None]
    )
    assert binned.fraction[0, 0].item() == pytest.approx(1.0)
    assert binned.visited[0, 1:].sum().item() == 0


def test_forward_only_traces_a_ray(arms):
    """From a spawn facing +x, curve climbs by one per step until the first
    blocked cell, then stays flat - derivable from the walkable set alone."""
    _, walkable, table = arms[True]
    x, y = next(
        c for c in sorted(walkable) if (c[0] + 1, c[1]) in walkable
    )
    ray = 1
    while (x + ray, y) in walkable:
        ray += 1
    positions = walk(
        table,
        spawns=np.array([(x, y, 0)]),
        actions=np.full((1, T), FORWARD, dtype=np.int64),
    )
    expected = np.minimum(np.arange(1, T + 1), ray)
    assert np.array_equal(_curves(positions, walkable=walkable, table=table).curves[0], expected)


def test_enumerating_the_support_hits_every_ceiling_exactly(arms):
    """Visit each of the room's cells exactly once: normalized entropy 1.0,
    final coverage 1.0, nAUC (n+1)/2n - all in closed form."""
    _, walkable, table = arms[True]
    n = len(walkable)
    positions = torch.tensor(sorted(walkable)).unsqueeze(0)  # (1, n, 2)
    W, H = _grid_shape(table)
    curves = coverage_curves(
        positions, denominators=torch.tensor([n]), width=W, height=H
    )
    assert curves.final_coverage.item() == pytest.approx(1.0)
    assert curves.nauc.item() == pytest.approx((n + 1) / (2 * n))
    visitation = visitation_by_room(
        positions,
        layout_ids=torch.zeros(1, dtype=torch.long),
        supports=_support(walkable, table=table),
    )
    assert visitation.entropy_bits.item() == pytest.approx(math.log2(n))
    assert visitation.normalized.item() == pytest.approx(1.0)


def test_a_visit_outside_the_rooms_support_raises(arms):
    """Off-support visits are a wiring error (wrong room channel), not data."""
    rooms, walkable, table = arms[True]
    landmark_cell = next(iter(sorted(rooms[0].cells)))
    positions = torch.tensor([[landmark_cell]])
    with pytest.raises(ValueError, match="outside"):
        visitation_by_room(
            positions,
            layout_ids=torch.zeros(1, dtype=torch.long),
            supports=_support(walkable, table=table),
        )


# --- the calibration table, pinned -------------------------------------------
# Reference values from `uv run python -m curious_george.envs.action_graph`
# at n=2000 (this session; matches the retired throwaway script to 3 decimals).
# n=600 here for runtime, so bands are a few SEM wide.

CALIBRATION = {
    # (impassable, probs-name): (cov@256, nAUC)
    (False, "uniform"): (0.191, 0.108),
    (False, "forward"): (0.395, 0.224),
    (True, "uniform"): (0.177, 0.102),
    (True, "forward"): (0.349, 0.200),
}


@pytest.mark.parametrize("impassable", [False, True])
@pytest.mark.parametrize("name", ["uniform", "forward"])
def test_random_baselines_match_the_calibration(arms, impassable, name):
    _, walkable, table = arms[impassable]
    probs = [0.25] * 4 if name == "uniform" else list(RAND_ACT_PROBA)
    rng = np.random.default_rng(0)
    spawns = spawn_states(walkable, n=600, rng=rng)
    positions = categorical_walk(table, probs=probs, spawns=spawns, steps=T, rng=rng)
    curves = _curves(positions, walkable=walkable, table=table)
    cov, nauc = CALIBRATION[(impassable, name)]
    assert curves.final_coverage.mean().item() == pytest.approx(cov, abs=0.03)
    assert curves.nauc.mean().item() == pytest.approx(nauc, abs=0.02)


def test_the_two_random_baselines_are_far_apart(arms):
    """Reporting one unlabeled 'random baseline' would be ambiguous by ~2x."""
    _, walkable, table = arms[False]
    rng = np.random.default_rng(0)
    spawns = spawn_states(walkable, n=600, rng=rng)
    naucs = {
        name: _curves(
            categorical_walk(table, probs=probs, spawns=spawns, steps=T, rng=rng),
            walkable=walkable, table=table,
        ).nauc.mean().item()
        for name, probs in {"uniform": [0.25] * 4, "forward": list(RAND_ACT_PROBA)}.items()
    }
    assert naucs["forward"] > 1.7 * naucs["uniform"]


def test_censoring_is_reported_not_hidden(arms):
    """Forward-weighted random reaches T25 in most episodes and T50 almost
    never - the reach fraction is the statistic, the conditional mean alone
    would describe the luckiest few percent."""
    _, walkable, table = arms[False]
    rng = np.random.default_rng(0)
    spawns = spawn_states(walkable, n=600, rng=rng)
    positions = categorical_walk(
        table, probs=list(RAND_ACT_PROBA), spawns=spawns, steps=T, rng=rng
    )
    curves = _curves(positions, walkable=walkable, table=table)
    assert curves.threshold(0.25).reached_fraction > 0.85
    assert curves.threshold(0.50).reached_fraction < 0.20
    assert math.isnan(curves.threshold(0.90).mean_step)


def test_the_sweeper_is_the_ceiling(arms):
    """nAUC ~0.61 and T90 inside the episode, in both arms - what 'a policy
    that never reaches T90 is genuinely worse than a sweeper' is read against."""
    for impassable in (False, True):
        _, walkable, table = arms[impassable]
        graph = ActionGraph(table)
        spawns = spawn_states(walkable, n=8, rng=np.random.default_rng(1))
        positions = np.concatenate([
            sweeper_walk(graph, walkable=walkable, spawn=tuple(s), steps=T)
            for s in spawns
        ])
        curves = _curves(positions, walkable=walkable, table=table)
        assert curves.nauc.mean().item() == pytest.approx(0.607, abs=0.03)
        t90 = curves.threshold(0.90)
        assert t90.reached_fraction == 1.0
        assert t90.mean_step < 230


# --- hardness-aware coverage separates what aggregates hide ------------------


def test_distance_bins_separate_sweeper_from_uniform(arms):
    """A random walker's coverage decays with spawn distance; the sweeper's
    does not. Aggregate coverage cannot see this - it is the whole point of
    binning by BFS distance."""
    _, walkable, table = arms[False]
    graph = ActionGraph(table)
    rng = np.random.default_rng(4)
    spawns = spawn_states(walkable, n=64, rng=rng)
    maps = torch.stack([
        torch.from_numpy(graph.distance_map(spawn=tuple(s))) for s in spawns
    ])

    uniform = torch.from_numpy(
        categorical_walk(table, probs=[0.25] * 4, spawns=spawns, steps=T, rng=rng)
    )
    sweeper = torch.from_numpy(np.concatenate([
        sweeper_walk(graph, walkable=walkable, spawn=tuple(s), steps=T)
        for s in spawns[:8]
    ]))

    far = distance_binned_coverage(uniform, distance_maps=maps).pooled_fraction()
    far_sweep = distance_binned_coverage(sweeper, distance_maps=maps[:8]).pooled_fraction()
    D = len(far)
    assert far[0].item() == pytest.approx(1.0)  # the spawn cell itself
    assert np.nanmean(far[2 * D // 3:].numpy()) < 0.25
    assert np.nanmean(far_sweep[2 * D // 3:].numpy()) > 0.95


# --- the denominator: why per-room, pinned as relationships ------------------


def test_pooled_ceiling_cannot_separate_the_arms(arms):
    """Verified this session and load-bearing for the whole design: the
    committed rooms' landmark placements share no always-blocked cell, so the
    POOLED walkable set - and with it `EnvCfg.loc_entropy_ceiling` - is
    identical across the affordance arms. Only the per-room denominator
    discriminates. A future room set may legitimately break the equality;
    this test is the loud notice that the metrics' meaning moved."""
    rooms_walk, _, _ = arms[False]
    rooms_imp, _, _ = arms[True]
    per_room_walk = {len(r.walkable(BASE)) for r in rooms_walk}
    per_room_imp = {len(r.walkable(BASE)) for r in rooms_imp}
    pooled_walk = pooled_walkable(BASE, rooms_walk)
    pooled_imp = pooled_walkable(BASE, rooms_imp)

    assert per_room_walk == {len(pooled_walk)}  # walkable arm: no difference
    assert all(n < len(pooled_imp) for n in per_room_imp)  # impassable: strict
    assert pooled_imp == pooled_walk  # the equality that kills ceiling-normalizing
