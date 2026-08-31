"""The transition table as a graph: action distances and reference walkers.

The device backend steps an exhaustive ``(W, H, 4, A) -> (x, y, dir)`` table
(`obs_bank.build_transition_tables`, stacked per layout at
`vector.DeviceTableShellPool.next_state`). This module consumes the SAME table
two more ways:

- as a graph over ``(x, y, dir)`` states, to measure shortest-path distance
  **in actions** - a turn costs a step, so the cell directly behind the agent
  is at distance 3, not 1;
- as the dynamics under reference walkers - `walk` replays any action
  sequence, `categorical_walk` samples state-independent actions, and
  `sweeper_walk` greedily heads for the nearest unvisited cell. Together they
  bound what any trained policy could score on the exploration metrics in
  `evaluation.exploration` without training anything.

Env geometry, not analysis (the promotion rule in `access.py`): everything
here is a pure function of a layout's table, and the table comes from the one
builder the live environments step.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Int

from curious_george.envs.layouts import BASE_ROOM_ID, MULTI_ROOM_ID, Layout
from curious_george.envs.obs_bank import NUM_TABLE_ACTIONS, build_transition_tables

#: `env.actions` order pinned by the table builder: left, right, forward, pickup.
LEFT, RIGHT, FORWARD, PICKUP = range(NUM_TABLE_ACTIONS)


def layout_tables(
    *, room: str = BASE_ROOM_ID, layouts: "list[Layout] | tuple[Layout, ...]"
) -> Int[np.ndarray, "R W H 4 A 3"]:
    """One transition table per layout, in layout order.

    Builds the room's `-Multi-v0` env once and revisits it per layout, exactly
    as `DeviceTableShellPool._collect_layout_banks` does - but without
    rendering the observation bank, which the graph never needs.
    """
    import gymnasium as gym

    env = gym.make(
        MULTI_ROOM_ID[room],
        landmarks=list(layouts[0].landmarks),
        agent_start_pos=None,
        agent_start_dir=None,
    )
    try:
        tables = []
        for layout in layouts:
            env.unwrapped.landmarks = list(layout.landmarks)
            env.reset(seed=0)
            tables.append(build_transition_tables(env.unwrapped)[0])
    finally:
        env.close()
    return np.stack(tables)


@dataclass
class ActionGraph:
    """One layout's transition table as a graph over ``(x, y, dir)`` states.

    Distances are cached per spawn state: an eval scores hundreds of episodes
    against tens of distinct spawns, and a map is ~W*H ints.
    """

    table: Int[np.ndarray, "W H 4 A 3"]
    _maps: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.table.shape[0], self.table.shape[1]

    def distance_map(
        self, *, spawn: tuple[int, int, int]
    ) -> Int[np.ndarray, "W H"]:
        """Min actions from `spawn` ``(x, y, dir)`` to occupy each cell.

        A cell's distance is the min over the four facings of the state
        distance. Unreachable cells (walls, impassable landmarks, and anything
        walled off) are -1. The spawn cell itself is 0.
        """
        key = tuple(int(v) for v in spawn)
        if (cached := self._maps.get(key)) is None:
            cached = self._maps[key] = self._bfs(key)
        return cached

    def _bfs(self, spawn: tuple[int, int, int]) -> np.ndarray:
        W, H = self.shape
        # State id = (x * H + y) * 4 + d; successors via the table's own rows.
        successors = (
            (self.table[..., 0].astype(np.int64) * H + self.table[..., 1]) * 4
            + self.table[..., 2]
        ).reshape(W * H * 4, NUM_TABLE_ACTIONS)
        dist = np.full(W * H * 4, -1, dtype=np.int32)
        x, y, d = spawn
        start = (x * H + y) * 4 + d
        dist[start] = 0
        queue = deque([start])
        while queue:
            s = queue.popleft()
            for n in successors[s]:
                if dist[n] < 0:
                    dist[n] = dist[s] + 1
                    queue.append(n)
        cell_dist = dist.reshape(W * H, 4)
        reached = (cell_dist >= 0).any(axis=1)
        out = np.where(
            reached, np.where(cell_dist < 0, np.iinfo(np.int32).max, cell_dist).min(axis=1), -1
        )
        return out.reshape(W, H).astype(np.int32)


def spawn_states(
    walkable: frozenset[tuple[int, int]], *, n: int, rng: np.random.Generator
) -> Int[np.ndarray, "n 3"]:
    """`n` i.i.d. spawn states ``(x, y, dir)``: uniform walkable cell x uniform
    direction - the support MiniGrid's `place_agent` samples in this room."""
    cells = np.array(sorted(walkable), dtype=np.int64)
    picks = cells[rng.integers(0, len(cells), size=n)]
    dirs = rng.integers(0, 4, size=n)[:, None]
    return np.concatenate([picks, dirs], axis=1)


def walk(
    table: Int[np.ndarray, "W H 4 A 3"],
    *,
    spawns: Int[np.ndarray, "E 3"],
    actions: Int[np.ndarray, "E T"],
) -> Int[np.ndarray, "E T 2"]:
    """Replay action sequences from spawn states; returns PRE-action positions.

    The convention matches the collector: ``positions[e, t]`` is where the
    agent stands when action ``actions[e, t]`` is chosen, so ``positions[:, 0]``
    is the spawn cell. The only stepping loop in this module - every walker is
    a policy over the same machine.
    """
    E, T = actions.shape
    xs, ys, ds = (spawns[:, i].astype(np.int64).copy() for i in range(3))
    positions = np.empty((E, T, 2), dtype=np.int64)
    for t in range(T):
        positions[:, t, 0] = xs
        positions[:, t, 1] = ys
        nxt = table[xs, ys, ds, actions[:, t]]
        xs, ys, ds = (nxt[:, i].astype(np.int64) for i in range(3))
    return positions


def categorical_walk(
    table: Int[np.ndarray, "W H 4 A 3"],
    *,
    probs: "list[float] | tuple[float, ...] | np.ndarray",
    spawns: Int[np.ndarray, "E 3"],
    steps: int,
    rng: np.random.Generator,
) -> Int[np.ndarray, "E T 2"]:
    """A state-independent categorical policy: sample all actions, then walk."""
    p = np.asarray(probs, dtype=np.float64)
    if p.shape != (NUM_TABLE_ACTIONS,) or not np.isclose(p.sum(), 1.0):
        raise ValueError(f"probs must be {NUM_TABLE_ACTIONS} values summing to 1, got {probs}")
    actions = rng.choice(NUM_TABLE_ACTIONS, size=(len(spawns), steps), p=p)
    return walk(table, spawns=spawns, actions=actions)


def sweeper_walk(
    graph: ActionGraph,
    *,
    walkable: frozenset[tuple[int, int]],
    spawn: tuple[int, int, int],
    steps: int,
) -> Int[np.ndarray, "1 T 2"]:
    """The POSITIVE control: repeatedly walk a shortest path to the nearest
    unvisited walkable cell. Not optimal, but achievable under the same
    dynamics (a turn costs a step), so it bounds what any policy could reach.
    Deterministic given the spawn; once everything is visited it stays put.
    """
    W, H = graph.shape
    successors = graph.table[..., :3]
    unvisited = set(walkable)
    x, y, d = (int(v) for v in spawn)
    positions = np.empty((1, steps, 2), dtype=np.int64)
    plan: deque[int] = deque()
    for t in range(steps):
        positions[0, t] = (x, y)
        unvisited.discard((x, y))
        if not plan and unvisited:
            plan = _plan_to_nearest(successors, W=W, H=H, start=(x, y, d), targets=unvisited)
        action = plan.popleft() if plan else PICKUP
        x, y, d = (int(v) for v in successors[x, y, d, action])
    return positions


def _plan_to_nearest(
    successors: np.ndarray,
    *,
    W: int,
    H: int,
    start: tuple[int, int, int],
    targets: set[tuple[int, int]],
) -> deque[int]:
    """BFS over states; actions of a shortest path to the first target cell hit."""
    seen = {start: (None, -1)}  # state -> (parent state, action taken)
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state[:2] in targets and state != start:
            actions: deque[int] = deque()
            while (edge := seen[state])[0] is not None:
                actions.appendleft(edge[1])
                state = edge[0]
            return actions
        x, y, d = state
        for action in range(NUM_TABLE_ACTIONS - 1):  # pickup never moves
            nxt = tuple(int(v) for v in successors[x, y, d, action])
            if nxt not in seen:
                seen[nxt] = (state, action)
                queue.append(nxt)
    return deque()


def main() -> None:
    """Re-derive the exploration-baseline calibration table from the library.

    What any policy's coverage numbers are read against: two random walkers
    (the two baselines a run can train with) and the greedy sweeper positive
    control, in both affordance arms, one 256-step episode, coverage of the
    ROOM'S OWN walkable set. The bands these print are pinned in
    tests/test_exploration_evals.py; the throwaway script that first sized
    them is superseded by this entry point.
    """
    import dataclasses

    from curious_george.configs import PRESETS
    from curious_george.envs.layouts import Selected, base_walkable, resolve_rooms
    from curious_george.log_and_store.storage import RAND_ACT_PROBA

    steps, n_random, n_sweeps = 256, 2000, 40
    entry = PRESETS["multienv-fast"]
    cfg = entry[1] if isinstance(entry, tuple) else entry
    base = base_walkable(BASE_ROOM_ID)
    levels = (0.25, 0.50, 0.75, 0.90)

    for impassable in (False, True):
        env = dataclasses.replace(cfg.env, source=Selected(n=5, impassable=impassable))
        room = resolve_rooms(
            shape=env.shape, content=env.content, source=env.source,
            room_rules=env.room_rules, set_rules=env.set_rules, indices=env.indices,
        )[0]
        walkable = room.walkable(base)
        table = layout_tables(layouts=[room])[0]
        graph = ActionGraph(table)
        arm = "impassable" if impassable else "walkable"
        print(f"\n=== {arm}, denominator {len(walkable)} ===")
        print(f"{'agent':<24}{'cov@256':>9}{'nAUC':>7}   " + "".join(f"{f'T{int(100 * v)}':>13}" for v in levels))

        arms = {
            f"uniform {[0.25] * NUM_TABLE_ACTIONS}": [0.25] * NUM_TABLE_ACTIONS,
            f"forward-weighted {list(RAND_ACT_PROBA)}": list(RAND_ACT_PROBA),
        }
        for label, probs in arms.items():
            rng = np.random.default_rng(0)
            spawns = spawn_states(walkable, n=n_random, rng=rng)
            positions = categorical_walk(table, probs=probs, spawns=spawns, steps=steps, rng=rng)
            _print_row(label, positions, walkable, levels)

        rng = np.random.default_rng(0)
        spawns = spawn_states(walkable, n=n_sweeps, rng=rng)
        positions = np.concatenate([
            sweeper_walk(graph, walkable=walkable, spawn=tuple(s), steps=steps)
            for s in spawns
        ])
        _print_row("greedy sweeper", positions, walkable, levels)


def _print_row(label, positions, walkable, levels) -> None:
    import torch

    from curious_george.evaluation.exploration import coverage_curves

    pos = torch.from_numpy(positions)
    W, H = int(pos[..., 0].max()) + 2, int(pos[..., 1].max()) + 2
    curves = coverage_curves(
        pos, denominators=torch.full((len(pos),), len(walkable)), width=W, height=H
    )
    cells = f"{curves.final_coverage.mean():9.3f}{curves.nauc.mean():7.3f}   "
    cells += "".join(f"{str(curves.threshold(v)):>13}" for v in levels)
    print(f"{label[:24]:<24}{cells}")


if __name__ == "__main__":
    main()
