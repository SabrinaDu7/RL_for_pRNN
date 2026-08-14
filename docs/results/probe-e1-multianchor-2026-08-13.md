2026-08-13

# E1 — multi-anchor consistency across five checkpoints: null everywhere

**Status: NULL, and a weak bound.** No checkpoint shows above-chance
object-vector coding. The bound is weak because the detector's own Stage 0 gate
does not pass (§4), so this rules out large effects only. Nothing here should be
read as "there is no object-vector code"; it says "there is none at the scale we
can currently see, and the location control is clean."

## 1. What was measured

E1 is the multi-anchor criterion (Høydal Fig. 2a,b), ported in
`../exp_instructions/instructions-objectAndOVC.md` §4: a unit that fires at a
fixed allocentric offset from *every* landmark is coding a metric relation, while
a place cell fires at one room location and so appears around only one landmark.

Pipeline, per checkpoint — `scripts/trace/e1_cell_figure.py`:

1. Probe: one trajectory from every walkable cell × 1 head direction (172
   trajectories × 256 steps), actions from a `RandomActionAgent`, independent of
   the network. Replay the checkpoint, drop a 20-step onset transient.
2. Rate maps: mean `h` per 14×14 bin, bins under 20 samples masked.
3. Anchors: the landmarks' centroids, read from the env's `Landmark` objects in
   `SHAPES` order — NOT recovered by colour, which reorders between rooms.
4. Offsets: every (dx, dy) with |dx|,|dy| ≤ 4 walkable for **all three** anchors.
5. Offset maps `V^a(Δ) = M[a + Δ]` — three crops of one rate map.
6. `vector_score` = mean pairwise correlation of the crops, over offsets at
   Chebyshev 2–4 (the landmark itself excluded); `vector_percentile` = where that
   sits among 100 random walkable triples drawn **for that same unit**;
   `peak_ratio ≥ 1.5` as the rate screen.
7. Rank by percentile, tie-break by `vector_score`, take the top 3.

Chance is 5% by construction. Full derivation in `scripts/trace/ovc_metric.py`.

## 2. Result

Fraction of screened units above their own 95th percentile **[live]**, cached in
`outputs/summary/fig_e1_cells_*.json`:

| checkpoint | env steps | fraction | vs 5% chance |
|---|---|---|---|
| 1-room L-room, 6×6 landmarks | 80k grad steps | 5/493 = **0.010** | below |
| 3-room L-room | 94.4M | 23/492 = **0.047** | at |
| 500-room L-room pool | 52.4M | 1/496 = **0.002** | below |
| 3-room square | 31.5M | 13/493 = **0.026** | below |
| 500-room square pool | 21.0M | 0/494 = **0.000** | below |

Three of five sit *below* chance, so the within-unit null is if anything
conservative: real landmark triples score no better, and often worse, than random
triples drawn for the same unit.

### 2.1 Step-matched, because the arms are not equally trained

The square archives are roughly a third of the L-room ones, so a raw cross-arm
comparison confounds geometry with training length. Both 3-room runs have a
checkpoint at exactly 31,457,280 steps and both pools at exactly 20,971,520, so
the comparison can be made exactly **[live]**:

```
                        unmatched              step-matched
3 rooms    L-room  0.047 (94.4M)          0.006 (31.5M)     square 0.026 (31.5M)
500 pool   L-room  0.002 (52.4M)          0.004 (21.0M)     square 0.000 (21.0M)
```

**The comparison inverts.** Unmatched, the L-room 3-room arm looks highest of
all five; step-matched it is *four times lower* than the square arm at the same
step count. Any dose-response claim across these arms must be step-matched. All
values remain at or below chance, so the null is unaffected — only the ordering
between arms was an artifact.

## 3. The location control, and the displacement test

Two controls ran with the result, both free on multi-room checkpoints.

**Location control** (`scripts/trace/e1_multi_anchor.py`, 3-room L-room at
94.4M): scoring the identical criterion at the *other* rooms' anchor positions —
landmark-free in these rollouts, and geometrically comparable by construction
rather than random.

```
OWN landmarks    0.047
OTHER positions  0.046
chance           0.050
```

Flat. So there is no geometry artifact either — which is what killed every
previous positive in this project (`probe-ovc-conjunction-2026-08-13.md` §6).

**Displacement** (`scripts/trace/e1_across_rooms.py`): rank units in one room,
then display those same units with the triad moved. `vector_score` per room:

```
                       room0    room1    room2
3-room SQUARE   #421  +0.792   +0.364   -0.081
                #128  +0.654   +0.252   -0.142
                #109  +0.722   -0.039   -0.168
500-pool SQUARE #162  +0.391   +0.001   +0.221
                #283  +0.498   +0.362   -0.056
                #291  +0.667   +0.195   -0.006
3-room L-room   #469  +0.857   +0.772   +0.807
                #177  +0.605   +0.091   +0.264
                #222  +0.774   +0.463   +0.434
```

In the square rooms the score collapses to zero or negative for all six units
when the triad moves: these are place cells whose within-room agreement was
coincidence. That is the point of ranking in one room and displaying the others —
picking the best unit per room would make every column look good by construction.

⚠️ The square arm is the informative one here. `ROOMS_RUN1`'s three L-rooms are
**exact translations** of one configuration (room 0→1 by (0,3), 0→2 by (3,0), all
at separation signature (6,6,6)), so a network can cover all three with one map
plus a translation. `ROOMS_SQUARE` has signatures (6,6,6)/(6,6,9)/(6,6,8) and is
not a translation set. See `curious_george/envs/layouts.py`.

⚠️ 3-room L-room #469 holds up across rooms (+0.86/+0.77/+0.81) but its field
tracks the top wall band, so this is consistent with boundary-vector tuning. The
border/BVC exclusion (E5) has not been run.

## 4. Why this is a bound and not a verdict

**The detector fails its own Stage 0 gate.** Committed characterisation
(`outputs/ovc/stage0_lroom.json`, plotted in
`outputs/summary/fig_ovc_power.png`):

```
negative control (odd/even)      0.014      want <= 0.05   PASS
specificity, injected place      0.012/0.006               PASS
positive, injected vector field  amp 0.5 -> 0.028
                                 amp 1.0 -> 0.062
                                 amp 2.0 -> 0.210
                                 amp 4.0 -> 0.575          want >= 0.90  FAIL
```

Specific but not sensitive: 42% of injected vector fields at 4× the unit's own
mean rate are missed. So the null bounds large effects only.

⚠️ That characterisation is on the **80k L-room net**, not on the checkpoints
scored here. The sensitivity floor is inherited, not measured. Re-running Stage 0
per checkpoint is the first thing any successor should do.

**And OVC-4 is a vacuous screen.** The spatial-information screen keeps
**500/500** units on the 3-room L-room net, changing no denominator **[live]** —
a trajectory-shuffle null destroys all spatial structure, so every spatially
tuned unit clears it trivially. Recorded against OVC-4 in the instructions.

## 5. Positive control: the place code is healthy

Median spatial information per checkpoint, top-16 units shown in
`outputs/summary/fig_e1_gallery_*.png` **[live]**:

```
3-room L-room 0.769   1-room L-room 0.748   3-room square 0.693
500-pool L-room 0.679   500-pool square 0.645        (top unit 2.2-6.8)
```

Clean, localised place fields in every net. That is what makes the object-coding
null a real bound rather than a failed measurement.

## 6. Reproducing

```bash
uv run python scripts/trace/e1_cell_figure.py  --ckpt <f.pt> [--env-config lroom_multi|squareroom_multi] \
    [--layouts rooms|pool] [--si-screen] --label "<what it is>" --tag <tag>
uv run python scripts/trace/e1_across_rooms.py --ckpt <f.pt> --env-config <cfg> --layouts <mode> \
    --label "<what it is>" --tag <tag>
uv run python scripts/trace/e1_multi_anchor.py --ckpt <f.pt> --env lroom_multi --room 0
uv run python scripts/trace/ovc_power_figure.py
```

Every figure writes a sibling `.json` carrying the checkpoint path, env, room
description, landmark list, chosen units and the population fraction.

⚠️ These figures use a **172-trajectory** probe (`--n-dirs 1`) while
`ovc_eval.py` defaults to 344 (`n_dirs=2`). The probe size is printed on each
figure; it is not matched between the two paths.
