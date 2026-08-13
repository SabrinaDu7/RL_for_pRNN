2026-08-13 (second compaction of the day — the first,
`compaction-2026-08-13.md`, closed the OVC/trace era; this one opens the
multi-room era)

# Multi-room training: five runs launched, and why

## 1. The one idea behind all five runs

`docs/results/result-summary-2026-08-12.md` ("The shape of the problem") established that
every design in this project has tried to make the object *necessary*, and every one closed
**at most one** of the routes by which the network localises without it:

| room | the route left open | evidence |
|---|---|---|
| L-room | **geometry** — an L-shaped wall plus three fixed landmarks | the entire §1–§4 null series |
| square room | **dead reckoning** — path integration from a fixed start | §5 Figure 5.4: quadrant decode ~84% with the object *absent*, delta ~0.000 |

**Multi-room training attacks dead reckoning directly.** If the room changes between
episodes, the same integrated trajectory lands at a different absolute position depending
on which room a stream is in, so path integration cannot resolve position and the network
must identify *which room it is in* from what it can see. That is the pressure to bind a
visual feature to a place, and it is the manipulation the summary already ranked first.

**The square room removes geometry.** Four-fold symmetric walls carry no position
information, so the landmarks are the only cue.

**Together they are the first design that closes BOTH.** That is the whole thesis of the
five runs.

## 2. What is running

Launched 2026-08-13, all on Mila `long` (7-day limit), GPU, `slurm/multienv.sh`.
State at 12:10 cluster time:

| job | room | layouts | elapsed | progress | rate |
|---|---|---|---|---|---|
| `10362462` | L-room | 3 (frozen `ROOMS_RUN1`) | 8:55 | 71.8M / 491.5M | 2363 it/s |
| `10362463` | L-room | 500 pool — **has the eval bug** | 8:55 | far behind, see §4 | — |
| `10363139` | L-room | 500 pool — eval fixed | 3:01 | 23.2M / 491.5M | 2303 it/s |
| `10364585` | **square** | 3 (frozen `ROOMS_SQUARE`) | 0:54 | 6.7M / 491.5M | 2053 it/s |
| `10364586` | **square** | 500 pool | 0:54 | 4.0M / 491.5M | 1510 it/s |

Every run: 240,000 **pooled** world-model gradient steps, `entropy_coef 0.01`,
step-tagged checkpoints every 10.5M environment steps rsynced to
`$SCRATCH/pRNN/<job>/checkpoints/` every 10 minutes. The runs are EXPECTED to hit the wall
clock — the archived series is the deliverable, not a finished run.

That gives the factorial **{L-room, square} × {3 rooms, 500 pool}** with the geometry route
as the manipulated variable, and the two L-room runs (already 9 h in) as the
geometry-present arm.

### Why 3 rooms AND a 500-layout pool

Different questions. Three rooms asks whether *specific alternative rooms* force binding and
is few enough to measure individually. The pool asks whether *variability itself* is what
matters — each layout is seen ~1/500 of the time, so no individual room can be memorised.

### Decisions that go against a previous finding, both deliberate

- **Pooled world-model training (`batched_wm: True`).** The 2026-08-11 sweep measured
  pooling as a loss on loss-per-gradient-step, and for ONE room that verdict stands. With
  several rooms per rollout a pooled step averages the gradient *across rooms* rather than
  visiting them one at a time, which is the point. Cost accepted explicitly: one update is
  exactly one gradient step, so the update count IS the budget.
- **GPU on `long`, not CPU/sapphire.** That verdict was measured for SERIAL training where
  the bill is op count; pooling cuts trainStep calls 8× per update, the exact quantity it
  rested on. And every sapphire node was reserved/draining/allocated, so a CPU job would
  have queued indefinitely. Measured outcome: ~2300 it/s, against the ~1270 the CPU note
  predicted. The GPU choice was right, for a reason that only became true with pooling.
- **`entropy_coef: 0.01`** against `ppo.yaml`'s 0.0. The square room collapsed at 0.0
  (1.49 → 0.19 bits) and took three sessions with it. Now holding at **1.96 bits of a 2.0
  maximum** with `loc_entropy` 6.49–6.69 — that failure mode is retired, with evidence.

## 3. What the L-room runs already show

From wandb, 22.5k gradient steps (`multienv-rooms`):

```
mean per-room sRSA 0.745     pooled 0.728     SWdist 0.015-0.023
pRNN loss 0.005907           reference: 0.005604 at 80,000 steps
policy entropy 1.96 bits     loc entropy 6.49     episodes/room 57237 / 57558
```

**sRSA is higher than the single-room reference has ever reached, and loss is at reference
level after 28% of the gradient steps.** Training on three rooms at once produces a *better*
spatial representation, not a degraded one. The pool run (500 rooms, each seen ~85–148
times) reaches sRSA 0.643 — it generalises across rooms it has barely seen individually.

**The remapping index is ~0** (+0.013 to +0.021, occasionally negative). Per the prediction
table that is `H_position`: one shared position map, room identity not bound in. With
per-room sRSA *high*, this is the informative null, not the "map degraded, run
uninformative" one. So at 22.5k of 240k steps the manipulation trains well and has not yet
produced room-specific maps.

**Object/object-vector probe on the 42M-step checkpoint: negative**, on the strongest
control design so far — three landmark configurations inside ONE network:

```
map from   scored at        frac > own p95        OWN mean 0.162
room 0     room 0 anchors            0.238  OWN   OTHER mean 0.162
room 2     room 0 anchors            0.238
room 2     room 2 anchors            0.130  OWN
```

Own and other identical to three decimals. The elevation above chance is real but sits at
the same room positions regardless of where the landmarks are — geometry, not object-vector
coding. Re-run on later archives before treating as settled.

## 4. Corrections made this session

Four, all mine, all caught by a control rather than by inspection.

1. **The OVC lead was retracted.** A conjunction statistic put 14.4% of units above their
   own 95th percentile at the real landmarks (chance 5%), and it passed a location control
   built from 40 random separation-matched triples. The **cross-over** killed it: the
   small-object net scores 0.203 at the L-ROOM's landmark positions against 0.164 at its
   own, having never had a landmark there. Own 0.153 vs other 0.138.
   **Transferable rule: a location control drawn from RANDOM positions is weaker than one
   drawn from geometrically comparable positions. Score the criterion under a second network
   whose landmarks are elsewhere.** Third time a within-unit null has been fooled by
   structure shared across units at particular locations, after (14,7) and the occlusion
   gradient. Write-up: `../results/probe-ovc-conjunction-2026-08-13.md`.
2. **The Stage 0 diagnosis in `compaction-2026-08-13.md` §4 was wrong.** That note said a
   unit's own null sits high; measured, the real-anchor score is median 0.023 against a
   per-unit null p95 of 0.567. The null is **wide**, not shifted — a correlation over ~47
   spatially autocorrelated offsets has few effective degrees of freedom. **Detrending, the
   step that note proposed next, does not address this and should not be attempted.**
3. **The pool run's spatial eval scored EVERY layout.** The eval costs one CPU rollout set
   per room, so the 500-room run spent more wall-clock measuring than training: 7,167
   gradient steps against the 3-room run's 22,542 in the same 5h40m. Fixed with
   `exp.eval_rooms_max` (a fixed prefix, so the series stays comparable). The utility
   argument is stronger than the cap: prnn caps the sRSA pairwise sample at
   `maxNtimesteps = 4000` rows and each room contributes 1888, so the **pooled estimate
   saturates at 2–3 rooms** — scoring 500 collected 944,000 rows and used 4,000. Extra rooms
   buy only the per-room spread. Confirmed in production: the fixed run runs at 2303 it/s
   against the 3-room run's 2363. `10362463` is left running as the unfixed comparison.
4. **The square-room artifact rendered L-rooms.** `build()` read `ENV_ID` from its own
   module namespace while `layout_artifact.py` assigned `globals()["ENV_ID"]` in *its*
   namespace, which `build()` never reads. Every room in the artifact was drawn as an L-room
   carrying square-room coordinates. The PNG figures were correct throughout, which is why
   looking at the figure did not catch it. Env id is now an argument.

## 5. Design work that changed the experiment

- **D4 deduplication is mandatory in the square room.** All three landmark shapes are
  D4-invariant, so a rotated layout is a *valid layout with identical shapes and colours*.
  Measured: 32,280 admissible assignments collapse to 4,035 orbits, and **100% of layouts
  share an orbit with another**. A rotated room puts landmarks at different absolute
  coordinates, so a network using ONE map, rotated, reads as remapping — a manufactured
  positive. Worse, the configuration-distance floor added to stop *translated* rooms rates
  one such pair at 12, comfortably "different rooms". Distinct colours and shapes do NOT
  prevent this; they only break the *within-room* symmetry.
- **Rooms must differ in internal geometry, not just position.** Selecting on distance alone
  returned three square rooms all at separation signature (6,6,6) — congruent triangles,
  7.1% of the admissible set, out of 13 signatures. Three rooms with identical internal
  geometry test one configuration moved around. Selection now requires distinct signatures:
  `(6,6,6) / (6,6,9) / (6,6,8)`.
- **Colours are chosen on RENDERED separation.** `Floor` paints `76 + 0.35 × colour`, so
  empty floor is (76,76,76) and every nominal separation reaches the network at ~a third of
  face value. `grey` renders 61 from empty floor and was visibly invisible; `purple` is 46
  from `blue`. Palette is blue/green/red/yellow, closest pair 89.
- **Shapes are `x` (5 cells), `plus` (5), `block3` (3×3 solid)**, all centre-anchored so one
  reference point serves every landmark. A 4-cell diamond was tried and rejected: it is
  `plus` minus its centre pixel, one pixel apart in the 7×7 view the network receives.

## 6. New results in `result-summary-2026-08-12.md` §3

Behaviour through the sequential displacement, which §3 never had — Figures 3.3 and 3.4.
**Behaviour tracks the present object and abandons the departed one**: (4,7) raw occupancy
0.213 → 0.043 on removal, 5.0×, p=0.0022, 8/8 seeds. The location control leaves (7,2)
+60.1 (p<1e-4) and (4,7) +29.2 (p=0.0010); (7,11) +6.2 (p=0.47) does not survive — it is
high-traffic regardless.

Three measurements now share one signature: readout and behaviour both follow the object and
collapse when it leaves; the hidden state never engages. Mechanistically forced — curiosity
reward is prediction MSE, a function of the readout, while the policy's input is `h`.

Also fixed: `collect_policy_rollouts` seeded only its start-direction rng. **Three**
generators needed seeding, not one — `torch.manual_seed`, `np.random.seed`, and
`env.env.reset(seed=)` for the gymnasium generator that owns `place_agent`. Removing that
noise *strengthened* the result: (4,7) +17.5 (p=0.23) → +29.2 (p=0.0010), departure 7/8
(p=0.082) → 8/8 (p=0.0012). ⚠️ Full bitwise determinism across calls REUSING one agent
object is still not established.

## 7. CERTAIN vs UNCERTAIN

**Certain (measured this session)**

- Multi-room training in the L-room reaches sRSA 0.745 / loss 0.0059 at 22.5k pooled steps,
  above the single-room reference. No policy collapse (1.96 of 2.0 bits).
- The remapping index detects remapping when it exists: synthetic `H_position` gives
  +0.0001, `H_room` gives +0.3812, through the real `calculateSpatialMetrics`.
- sRSA itself is correct: perfect place code +0.9985, shuffled positions −0.0035, noise
  +0.0003.
- The OVC signal on existing checkpoints is geometry, by cross-over, twice (two nets; and
  three rooms in one net).
- The square room admits 4,035 D4-distinct anchor assignments; 100% of raw layouts have a
  D4 twin.
- Behaviour follows the present object and abandons it on removal (n=8, seeded).

**Uncertain**

- **Whether multi-room training produces room-specific maps.** Index ~0 at 22.5k of 240k
  steps. The runs exist to answer this.
- **Whether the square room changes that.** Zero data — the square runs are <1 h old.
- Whether the square pool's 1510 it/s is a startup artifact or a real per-step cost of a
  500-layout observation bank resident on device (~100 MB). ETA 90 h against a 108 h wall,
  so it fits, but with less margin than the others.
- Whether object-vector coding emerges later in training. The probe was run at 42M of 491M
  steps.
- The `[2,0,8]` run's training object location (see §8 Q2) — not locally recoverable, needs
  the wandb run config.

---

## 8. Open questions — THE CONTINUATION POINT

Sabrina's questions on `../results/result-summary-2026-08-12.md`, verbatim, each followed by
what I confirmed from source so the next session does not start cold.

### Figure 0.2 — `outputs/trace/fig_otc_maps_diff.png`

> (1) why did we choose learning rates [2, 0, 8] for [phase A, B and C]? Also (2) can we
> have these rate maps for a [7, 2] phase instead of a [7, 11] phase? This is all part of
> the same run I'm pretty sure. (3) You also write that the cells that are presented in this
> plot are ones weighted heavily by decoder (ie W_out). What does that mean? (4) Is there a
> way to show that the activations in W_out are high at the target object locations? I guess
> it doesn't apply that much here, because these activations are just high for green at the
> object location, which is expected...

**Confirmed from source, and two of these expose documentation defects:**

- **(1) `[2,0,8]` is NOT per phase.** `lr_trials` is a per-optimizer-group learning-rate
  multiplier over **`[W, W_out, W_in]`** — the recurrent matrix, the readout, and the input
  weights. So `[2,0,8]` = recurrent at 2×, **readout FROZEN**, input at 8× (i.e. 4× the
  recurrent). Baseline is `[2,2,2]`. §2 states this correctly; Figure 0.2's caption gives
  the bare triple with no expansion, which is what invites the phase reading.
  **Action: expand the triple wherever it appears in a caption.**
- **(3) "the object decoder" is NOT `W_out`.** `scripts/otc_figures.py:300-301` fits an
  `sklearn` `LogisticRegression(max_iter=2000, C=0.05)` to classify object-present vs
  object-absent from `h` at ph0, restricted to timesteps where the object is in the 7×7
  view, then takes `np.argsort(-np.abs(coef))[:8]`. It is the §2 analysis probe, a separate
  linear readout fitted by us — not the pRNN's own `W_out` (`Linear(500→147)`).
  **The caption's "the object decoder weights most heavily" reads as `W_out` and should not.
  Action: rename to "the ph0 object-presence probe" in caption and docstring.**
- **(2) A (7,2) version is feasible but is a DIFFERENT question, and which one depends on an
  unverified fact.** `outputs/trace/probe_lroom_obj7_2` exists, so the probe side is ready.
  But `LOC = (7,11)` is a module constant (`scripts/otc_figures.py:24`) used for the probe,
  the view-coords mask and the `+` marker. The open fact: **the `[2,0,8]` checkpoint
  (`outputs/otcFrzIn8-cur-dot-0801-120212`) has no config saved locally**, so which object
  location it was TRAINED with is not locally recoverable — check the wandb run config.
  - If it trained at (7,11): scoring at (7,2) asks "which units encode an object at a
    location this net never saw one at" — a control, and arguably the more interesting one.
  - Sabrina's "this is all part of the same run" needs checking: §1's three locations are
    **separate runs** (3 locations × 3 seeds); `[2,0,8]` is one condition. So "the (7,2)
    phase" may not exist for this net.
- **(4) Sabrina's own caveat is right, and the non-trivial version already exists.**
  "Green is high at the object location" is expected and says nothing. The informative
  measurement is Figure 3.2 / §0: the object-centred readout map, where **~89% of the signal
  at a location is already there BEFORE the object arrives** ((4,7) +0.0395 before vs
  +0.0445 during, t=9.57). That is the claim worth showing, and it is the opposite of
  trivial.

### Figure 1.2 — `outputs/trace/fig_trace_3loc.png`

> this figure makes no sense because we're looking at the same cells across four different
> conditions: pre-exposure, exposure (7,11), a separate exposure (14,7) and another exposure
> (7,2). But why are we only showing the units most modulated at (7,11) and only plotting a
> + at (7,11)? We should be plotting the 3 most modulated cells for (7,11) and have four
> columns, then the three most modulated cells at (14, 7) as rows (probably different cells
> in this case) and another four columns and the three most modulated cells at (7,2) as
> another three rows and four columns.

**The critique is correct.** `scripts/trace/trace_cell_figures.py:88-105`: `L = (7,11)`
hard-coded; rows are the top 8 units by Δ object-modulation **at (7,11) only**; the `+`
marks (7,11) in **every** column. Columns 3 and 4 therefore show units selected for (7,11),
marked at (7,11), in runs whose object was somewhere else — they carry no information about
their own object.

**Action: rebuild as three blocks of 3 rows × 4 columns, each block selecting its units for
its own location and marking its own location.** That makes it a proper location-control
matrix in spatial-tuning form, which is exactly what §4 says every successor must do.

### Figure 3.1 — `outputs/trace/fig_seq_fields_3058.png`

> this show cells that have the most changed receptive fields. What does "most changed"
> mean? What is the metric used here.

**Confirmed** (`scripts/seq_figures.py:89-95`): for radius-2 discs at the three object
locations, `g0 = field_gain(baseline)`, and

```
dg[unit] = max over (phases, object locations) of | field_gain(phase) − g0 |
units    = argsort(-dg)[:8]
```

So "most changed" = **largest absolute change in field gain at ANY of the three object
locations, relative to pre-exposure, maximised over phases**. It is NOT a global map-change
measure.

⚠️ **This creates a tension worth resolving.** Units are selected *because* their gain
changed at the object locations, and the caption then says the reorganisation is "not at the
object". Either the selected changes are decreases, or the visible reorganisation elsewhere
dominates the eye. **Action: either report the sign and magnitude of the selecting `dg` on
the figure, or select units on a global map-change measure and let the object locations be
tested rather than assumed.**

---

## 9. Next steps

1. **Babysit the five runs.** `uv run python scripts/multienv/checkpoint_curve.py --run <dir>
   --spatial` reads the archives without waiting for a job to finish; the newest row prints
   `(latest)` so a running job's checkpoint is never read as final. Watch sRSA high, SWdist
   low, prediction loss falling, per-room episode counts balanced.
2. **Re-run the object/OVC cross-over on later archives.** The 42M-step probe was negative;
   the remapping index was ~0 at the same point. Both should be re-asked at 150M+.
3. **Answer §8.** All three are figure/documentation defects with concrete fixes, and two of
   them (the `W_out` mislabel, the single-location row selection) could mislead a reader who
   was not in the room when the figures were made.
4. Verify the `[2,0,8]` run's object location from its wandb config, which unblocks Q2.
5. `git push origin sdu/multi-env` — the branch has never been pushed. minigrid IS pushed
   (`ce375b0`) and pinned in `uv.lock`.
