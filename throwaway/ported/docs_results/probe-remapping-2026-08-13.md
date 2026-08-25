2026-08-13

# Multi-room training builds a better SHARED map, not room-specific ones

**Status: INFORMATIVE NULL.** The remapping index sits at ~0 across the whole
archived series while per-room sRSA climbs to 0.79 — the map is excellent and
simply does not encode which room the agent is in. This is the null the
multi-room design exists to break, and it is the informative version of it: the
alternative reading, "the map degraded so the run says nothing", is excluded by
the per-room number being high.

Figure: `outputs/summary/fig_remapping_index.png`
(`scripts/multienv/remapping_figure.py`).

## 1. The question, and why this measures it

`docs/results/result-summary-2026-08-12.md` ("The shape of the problem")
established two routes by which the network localises without ever using an
object, and that every design had closed at most one:

| room | route left open | evidence |
|---|---|---|
| L-room | geometry — an L-shaped wall plus three fixed landmarks | the §1–§4 null series |
| square room | dead reckoning from a fixed start | §5 Fig 5.4: quadrant decode ~84% with the object *absent* |

Multi-room training attacks dead reckoning: if the room changes between
episodes, the same integrated trajectory lands at a different absolute position
depending on which room a stream is in, so path integration cannot resolve
position and the network must identify *which room it is in* from what it sees.

**The remapping index** = `mean(per-room sRSA) − pooled sRSA`
(`curious_george/evaluation/spatial.py::evaluate_multi_room_representation`).
sRSA asks whether representational similarity tracks spatial proximity. Computed
*within* one room it says the map is good there; computed over rows *pooled
across* rooms it additionally requires that the same (x, y) in different rooms be
represented alike. So:

- ONE shared position map → pooled ≈ per-room → index ≈ 0
- ROOM-SPECIFIC maps → per-room stays high, pooled collapses → index > 0

Two readings share "index ≈ 0" and must be told apart, which is why per-room
sRSA is always reported beside it: **index ≈ 0 with HIGH per-room sRSA** is the
informative null; **index ≈ 0 with LOW per-room sRSA** means the map degraded and
the run is uninformative.

## 2. The scale the measurement is read against — derived, not quoted

Synthetic activity satisfying each hypothesis exactly
(`scripts/multienv/remapping.py::synthetic`, at `predNet.hiddensize = 500`), put
through the **same** `calculateSpatialMetrics` the real curve uses **[live]**,
cached at `outputs/summary/remapping_reference.json`:

```
H_position (one shared map)     index  -0.0002
H_room     (room-specific maps) index  +0.3904
```

The measurement therefore separates the hypotheses by ~0.39, and anything near
zero is squarely `H_position`.

(`compaction-2026-08-13-multienv.md` §7 records +0.3812 for the same quantity;
that is the same measurement at a different synthetic seed.)

## 3. Result

Both runs are L-room, curiosity-driven PPO **from scratch** — no random-action
pre-training, no policy warmup, `logging.load_worldmodel: False` (verified from
source; `exp.random_action_agent` is a single static flag with no switching path).
Probe: 8 trajectories × 256 steps per room, fixed seed, replayed against every
archived checkpoint. Cached in each run's `checkpoint_curve.json` **[live]**.

**3 rooms** (`ROOMS_RUN1`, keys `0107cd84 2ffe8e32 b6b47c8d`):

```
        step   mean sRSA   pooled     remap   SWdist       loss
  10,485,760      0.4562   0.4459   +0.0103   0.0181   0.007237
  31,457,280      0.6690   0.6599   +0.0091   0.0242   0.005226
  52,428,800      0.7382   0.7285   +0.0096   0.0306   0.004862
  73,400,320      0.7870   0.7677   +0.0192   0.0334   0.004380
  94,371,840      0.7905   0.7639   +0.0266   0.0379   0.003857
```

**500-room pool** (4 of 500 scored, a fixed prefix):

```
        step   mean sRSA   pooled     remap   SWdist       loss
  10,485,760      0.4898   0.4891   +0.0007   0.0131   0.007117
  31,457,280      0.6809   0.6783   +0.0026   0.0199   0.005729
  52,428,800      0.7862   0.7845   +0.0017   0.0295   0.004593
```

**Per-room sRSA reaches 0.79 — higher than the single-room reference has ever
reached — while the index stays between +0.0007 and +0.0266, i.e. 0.2–7% of the
`H_room` scale.** Prediction loss falls monotonically in both. The manipulation
trains well and has not produced room-specific maps.

### 3.1 More room variability gives LESS remapping, not more

```
3 rooms       +0.0091 .. +0.0266
500-room pool +0.0007 .. +0.0026      ~10x flatter
```

Opposite to the dose-response the design predicted. The mechanistic reading is
straightforward: with 500 rooms each seen ~1/500 of the time, no individual room
*can* be memorised, so a single shared map is the only thing learnable; with 3
rooms there is some room for individuation and a little appears.

## 4. ⚠️ The L-room arm is a weaker manipulation than intended

`ROOMS_RUN1`'s three rooms are **exact translations of one configuration** —
room 0 → 1 by (0, 3), room 0 → 2 by (3, 0), all three at separation signature
(6, 6, 6), i.e. congruent right triangles **[live]**. Shape and colour
assignments are permuted across the shift, so identity varies; geometry does not.

**A network can therefore cover all three rooms with one map plus a translation,
and remapping is never *required* in this arm.** An index of ~0 here is
consistent both with "bound nothing" and with "manipulation too weak to force
binding". The selection's configuration-distance floor separated the rooms by
position, not by internal geometry — the same failure the square-room selection
was later fixed for with `distinct_signatures`. `ROOMS_RUN1` predates that fix.

`ROOMS_SQUARE` does not have this problem: signatures (6,6,6) / (6,6,9) /
(6,6,8), and no pair is a translation. **The square runs are the arm that
matters for this question.** They were at 31.5M and 21.0M steps when this was
written — too young to score here.

Verify with:

```bash
uv run python -c "from curious_george.envs.layouts import ROOMS_RUN1 as R; \
  print({(dx,dy) for dx in range(-14,15) for dy in range(-14,15) \
  if {(x+dx,y+dy) for x,y in R[0].anchors} == set(R[1].anchors)})"
```

## 5. Other bounds on this result

- **n = 1 seed per arm.** No replication.
- **The runs are ~19% and ~11% of their 491.5M-step budget.** The index does
  drift upward in the 3-room arm (+0.010 → +0.027), which is ~7% of the `H_room`
  scale and not something to call a trend at n=1.
- **The pooled estimate saturates.** prnn caps the sRSA pairwise sample at
  `maxNtimesteps = 4000` rows while each room contributes ~1,888, so pooling more
  than 2–3 rooms adds nothing; `eval_rooms_max` scores a fixed prefix of 4.
- **`SWdist` rises with training** (0.018 → 0.038 for 3 rooms). Not investigated
  here; it should be low, and it is still low in absolute terms, but the
  direction is worth watching.

## 6. Reproducing

```bash
uv run python scripts/multienv/checkpoint_curve.py --run <run_dir> \
    --env lroom_multi|squareroom_multi [--layouts one|rooms|pool] --spatial
uv run python scripts/multienv/remapping_figure.py <run_dir> [<run_dir> ...]
```

`checkpoint_curve.py` reads the room, the base geometry, the layout set, the pool
size/seed and the D4 dedup from the run's own config through the same
`resolve_layouts` the training loop uses — re-specifying any of them is how a
square run gets silently scored in an L-room. Its json carries a `meta` block so
the series can say which run it is; `remapping_figure.py` refuses a
metadata-less json rather than guessing.
