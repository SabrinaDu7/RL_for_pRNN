2026-08-13

# Multi-room training: making the landmark necessary

## Question

Does training in several rooms at once force the pRNN to bind what it sees to
where it is — and does that produce object or object-vector coding that single-room
training never did?

## Why this design

Every object-coding result in this project has been null, and by 2026-08-12 the
reason was no longer a mystery. `docs/results/result-summary-2026-08-12.md`
established **two demonstrated routes** by which the network locates itself
without ever using an object, and every design so far closed at most one:

| room | how it localises without the object | evidence |
|---|---|---|
| L-room | **geometry** — an L-shaped wall plus three fixed landmarks | the whole §1–§4 null series |
| square room | **trajectory history** — dead reckoning from a fixed start | quadrant decode ~84% with the object *absent*, delta ~0.000 |

Multi-room training attacks the second route directly, and it is the manipulation
the summary already recommended first: **the same integrated trajectory lands at a
different absolute position depending on which room the stream is in**, so dead
reckoning cannot resolve position and the network must identify which room it is
in from what it can see. That is the pressure to bind a visual feature to a place.

Two runs, differing only in how many rooms:

- **Run 1 — three rooms.** `exp.layouts=rooms`, the frozen `layouts.ROOMS_RUN1`.
  Few enough to measure individually.
- **Run 2 — a pool.** `exp.layouts=pool`, a seeded uniform sample of the
  admissible set. Tests whether the effect needs *specific* rooms or just
  *varying* ones.

Same code path; set size is the only difference.

## What was built, and where

| piece | file |
|---|---|
| landmark vocabulary, shapes, `MiniGrid-LRoom-Multi-v0` | `../minigrid` `minigrid/envs/Lroom.py` |
| layout generation, constraints, the frozen three rooms | `curious_george/envs/layouts.py` |
| per-stream layout axis, per-episode resampling | `curious_george/envs/vector.py::DeviceTableShellPool` |
| per-room + pooled spatial metrics | `curious_george/evaluation/spatial.py::evaluate_multi_room_representation` |
| step-tagged checkpoint series | `curious_george/training/loop.py::save_checkpoint` |
| run shape | `Configs/run/multienv.yaml`, `Configs/env/lroom_multi.yaml` |
| design figures and the layout gallery | `scripts/layout_figures.py`, `scripts/layout_artifact.py` |

Constraint values, the colour palette and the reasons for each live in the
`layouts.py` module docstring — one home, not restated here.

## Decisions that go against a previous finding, and why

**Pooled world-model training (`batched_wm: True`).** The 2026-08-11 sweep
(`docs/sab_context/open_choices.md`) measured pooling as a loss on loss-per-
gradient-step, and for a single room that verdict stands. With several rooms per
rollout a pooled step averages the gradient **across rooms** rather than visiting
them one at a time, which is the point of the design. The cost is explicit: one
update is exactly one gradient step, so the update count *is* the gradient budget.

**`entropy_coef: 0.01`.** `Configs/algo/ppo.yaml` ships 0.0, and nothing resisted
policy collapse in the symmetric room — entropy fell 1.49 → 0.19 bits and made
sessions 4–6 uninformative rather than negative. Gate on per-room exposure.

## Metrics, with what each value would mean

The remapping index is the one this experiment turns on.

| metric | if the network still uses ONE position map | if it has bound rooms to places |
|---|---|---|
| `mean(per-room sRSA) − pooled sRSA` | ≈ 0 | > 0: per-room stays high, pooled collapses |
| cross-room rate-map correlation per unit | high (≈ the r ≈ 0.98 stability baseline) | low |
| room-identity decode from `h` | chance | above chance |
| cross-room position transfer | high | low |

Two readings share "index ≈ 0" and must be told apart, which is why per-room sRSA
is always reported beside it: **index ≈ 0 with high per-room sRSA** is the null
this experiment is trying to break; **index ≈ 0 with low per-room sRSA** means the
map itself degraded and the run is uninformative, not negative.

Per-room **episode counts** are reported with every per-room number. A room with
few episodes is UNTESTED, not negative — the failure that made square-room
sessions 4–6 worthless.

Only the sRSA family runs inside training (it is a `calculateSpatialMetrics` call
already on the CPU eval path). Cross-room map correlation, the room decode and the
position-transfer test read the archived checkpoint series offline, which is what
the archive exists for.

## Known limitations, to state with any result

1. **Cross-room landmark distance maxes at 3 cells**, measured over the whole
   admissible set. The room is 172 walkable cells; rooms cannot be made more
   different than this without changing the room.
2. **The pool sees each layout rarely.** With 8 streams and 500 rooms, a given
   room gets ~1/500 of the episodes. Run 2 tests whether *variability* matters,
   not whether any particular room is learned.
3. **A pooled gradient step is not guaranteed to contain every room.** With 8
   streams over 3 rooms a given room is absent from ~3.9% of steps.
4. **The landmarks are still floor tiles the agent walks over**, so the
   object-vector disanalogies in `instructions-OVC.md` still apply — smaller and
   off the walls now, but not obstacles.
