2026-08-18

# E1 at the FINAL checkpoint — the fraction rises in every arm, and the extended location control shows why that is not object coding

**Status: NULL, and still a weak bound — but the reason is now different and
stronger.** At 482,344,960 environment steps the population fraction rises in
every arm, and in two arms it sits at or above the 5% chance line. The
`--room 0` location control appears to *support* a landmark-driven signal in the
square arm (OWN 0.053 / 0.145 vs OTHER 0.009 / 0.010). Extending that control
over every (rollout room × anchor triad) pair kills it: **the score is a
property of the anchor triad, essentially independent of whether that room's
landmarks are physically present** (§3). That is the same failure mode that
retracted the previous OVC lead (`probe-ovc-conjunction-2026-08-13.md` §6),
caught here before it was reported as a positive.

Base commit `84f7275f406c1f65cf6d24ae9cd8e297cf09a35d`. Every number below names
the JSON it was read from; the commands that wrote those JSONs are in §8.

## 0. What is new since [`probe-e1-multianchor-2026-08-13.md`](probe-e1-multianchor-2026-08-13.md)

- All four multi-room runs finished their budget. **Every one is now at exactly
  482,344,960 environment steps**, so the training-length confound that forced
  the step-matched re-analysis in that document's §2.1 is gone: the arms differ
  in room geometry and layout-set size and in nothing else about the schedule.
- The location control is extended from one (rollout room, anchor triad) cell to
  all nine per checkpoint. This is what changes the reading.
- The positive control no longer shows what it showed on 08-13 (§5).

Unchanged and carried forward, because both still bound everything here:

⚠️ **The detector fails its own Stage 0 gate**, so every null is a bound on
large effects only — see §6.

⚠️ **`ROOMS_RUN1`'s L-rooms are exact translations of one configuration**, so
that arm is a weak manipulation; `ROOMS_SQUARE` is not — see §7.

## 1. What was measured

The pipeline is unchanged from
[`probe-e1-multianchor-2026-08-13.md`](probe-e1-multianchor-2026-08-13.md) §1
and is not restated here; `scripts/trace/ovc_metric.py` owns the maths. The two
facts that carry the whole reading:

- **Chance is 5% by construction.** Each unit's real-anchor `vector_score` is
  ranked against a null of random walkable anchor triples drawn *for that same
  unit*, and the unit counts only above its own 95th percentile
  (`ovc_metric.vector_percentile`). No population threshold is involved.
- **The offset window is a property of the room and the triad, not a knob.**
  `layouts.common_offsets` keeps only offsets that land on a walkable cell for
  *every* anchor. This is the quantity §3 turns on.

The checkpoints, rsynced from `$SCRATCH/pRNN/<job>/<run>/checkpoints/` on Mila
and verified by `md5sum` against the remote (all four distinct, all
`hidden_size` 500):

| local path | job | trains on |
|---|---|---|
| `outputs/final-lroom3/predictiveNet_state_step0482344960.pt` | `multienv_rooms_10362462` | `ROOMS_RUN1`, L-room |
| `outputs/final-lroom500/predictiveNet_state_step0482344960.pt` | `multienv_pool_10363139` | 500-layout pool, L-room |
| `outputs/final-square3/predictiveNet_state_step0482344960.pt` | `multienv_squareroom_rooms_10364585` | `ROOMS_SQUARE`, square |
| `outputs/final-square500/predictiveNet_state_step0482344960.pt` | `multienv_squareroom_pool_10364586` | 500-layout pool, square |

⚠️ **The probe is not the same size in the two room geometries**, and the
08-13 document states it as if it were (it says 172 trajectories under a heading
that covers every checkpoint). The probe is one trajectory per walkable cell per
head direction at `--n-dirs 1`; the L-room has 172 walkable cells and the square
room has 196, so the square arm's probe is and always was 196 trajectories. This
is a hard property of the rooms, confirmed by
`layouts.walkable_cells` and printed on every figure. Corrected in that document
as part of this work; recorded here because it is a difference between the arms
that step-matching does *not* remove.

## 2. Population fraction, EARLY vs FINAL

Fraction of rate-screened units (`peak_ratio >= 1.5`) over their own 95th
percentile. EARLY values from `outputs/summary/fig_e1_cells_<tag>.json`; FINAL
from `outputs/summary/fig_e1_cells_final_<tag>.json`.

| run | EARLY step | EARLY fraction | FINAL step | FINAL fraction | vs 5% chance |
|---|---|---|---|---|---|
| 3-room L-room | 94,371,840 | 23/492 = **0.047** | 482,344,960 | 33/491 = **0.067** | above |
| 500-pool L-room | 52,428,800 | 1/496 = **0.002** | 482,344,960 | 11/496 = **0.022** | below |
| 3-room square | 31,457,280 | 13/493 = **0.026** | 482,344,960 | 26/494 = **0.053** | at |
| 500-pool square | 20,971,520 | 0/494 = **0.000** | 482,344,960 | 5/492 = **0.010** | below |

Every arm rises. Read naively that is a dose-response in training time, and the
two 3-room arms have reached or passed chance. §3 is why that reading does not
survive.

The ordering across arms is now interpretable, which it was not on 08-13: at a
common step count the 3-room arms exceed their 500-pool counterparts in both
geometries (0.067 > 0.022 and 0.053 > 0.010). The earlier document had to
step-match to say anything, and found the comparison inverted when it did.

**Figures.** `outputs/summary/fig_e1_cells_final_lroom3.png`,
`fig_e1_cells_final_lroom500.png`, `fig_e1_cells_final_square3.png`,
`fig_e1_cells_final_square500.png` — each with a sibling `.json` carrying the
checkpoint path, room, landmark list, chosen units and the fraction.

Inspected directly (`fig_e1_cells_final_lroom3.png`): the top-ranked unit #222
does show a bump at a similar displacement in all three offset maps, which is
the object-vector signature the figure is built to display. #423 does too. #314
shows a horizontal band across its offset maps rather than a localised bump.
So the figure is not producing garbage — the candidates look like candidates.
That is exactly why the control in §3 had to be run before anything was claimed.

## 3. The location control, extended — and this is the result

`scripts/trace/e1_multi_anchor.py` scores the identical criterion at the *other*
rooms' anchor positions, which are landmark-free in the rollouts being scored.
Run at its default `--room 0`, the square arm looks like a hit:

```
                        OWN landmarks   OTHER positions   chance
3-room L-room               0.067            0.068         0.050
500-pool L-room             0.145            0.144         0.050
3-room SQUARE               0.053            0.009         0.050
500-pool SQUARE             0.145            0.010         0.050
```

The L-room arm is flat, as it was on 08-13. The square arm is not: OWN is 6x and
14x OTHER. On its face that is landmark-driven coding — the thing this project
has been looking for since July.

**It is not.** The control's own premise is that the other rooms' anchors are
"geometrically comparable by construction". That premise holds in the L-room and
fails in the square room, and the failure is measurable without touching a
network — from `layouts.common_offsets` alone:

```
ROOMS_RUN1    room 0 / 1 / 2   separation signature (6,6,6) (6,6,6) (6,6,6)   testable offsets 49 / 49 / 49
ROOMS_SQUARE  room 0 / 1 / 2   separation signature (6,6,6) (6,6,9) (6,6,8)   testable offsets 49 / 40 / 42
```

So in the square arm, OWN is scored over 49 offsets and OTHER over 40 and 42.
`ROOMS_SQUARE` was deliberately chosen to have *distinct* separation signatures
(`curious_george/envs/layouts.py`), which is right for the remapping question
and is precisely what breaks the comparability this control assumes.

The direct test needs no new code — `e1_multi_anchor.py` already takes `--room`,
so the same criterion can be scored for every (rollout room, anchor triad) pair.
If the signal is landmark-driven, the **diagonal** (landmarks physically
present) must stand out. If it is triad geometry, the **columns** must stand out.
Fraction over each unit's own 95th percentile, `*` marks the diagonal, read from
`outputs/ovc/e1_multi_anchor_final_*.json`:

```
                    anchors of      anchors of      anchors of
                    room 0          room 1          room 2
3-room L-room       (n_off 49)      (n_off 49)      (n_off 49)
  rollouts room 0     0.067*          0.069           0.067
  rollouts room 1     0.043           0.047*          0.059
  rollouts room 2     0.073           0.073           0.053*

500-pool L-room     (n_off 49)      (n_off 49)      (n_off 49)
  rollouts room 0     0.145*          0.142           0.145
  rollouts room 1     0.116           0.111*          0.118
  rollouts room 2     0.138           0.136           0.111*

3-room SQUARE       (n_off 49)      (n_off 40)      (n_off 42)
  rollouts room 0     0.053*          0.000           0.018
  rollouts room 1     0.026           0.004*          0.018
  rollouts room 2     0.039           0.002           0.022*

500-pool SQUARE     (n_off 49)      (n_off 40)      (n_off 42)
  rollouts room 0     0.145*          0.000           0.020
  rollouts room 1     0.075           0.000*          0.022
  rollouts room 2     0.121           0.002           0.018*
```

Column means (per anchor triad) against row means (per rollout room):

```
                  column means (triad)        row means (rollout room)   diagonal   off-diagonal
3-room L-room     0.061  0.063  0.060         0.068  0.050  0.066         0.056       0.064
500-pool L-room   0.133  0.130  0.125         0.144  0.115  0.128         0.122       0.133
3-room SQUARE     0.039  0.002  0.020         0.024  0.016  0.021         0.026       0.017
500-pool SQUARE   0.114  0.001  0.020         0.055  0.033  0.047         0.054       0.040
```

**The square arm's variance lives entirely in the columns.** The room-0 triad
scores 0.053 / 0.026 / 0.039 in the 3-room square net depending on which room
the rollouts came from — high even when its landmarks are *absent*. The room-1
triad scores 0.000 / 0.004 / 0.002 — near zero even when its landmarks are
*present*. In the 500-pool square net the ordering by triad is 0.114 / 0.001 /
0.020 while the ordering by rollout room spans only 0.033 to 0.055.

Within each square checkpoint the column means order exactly as the offset
counts do (49 > 42 > 40). That is confirmed in these four checkpoints; whether
the driver is the window size itself or the triad's shape is not separated here,
and both are geometry rather than landmarks.

**The diagonal does not stand out.** In both L-room checkpoints the diagonal
mean is *below* the off-diagonal mean (0.056 vs 0.064; 0.122 vs 0.133) — having
the landmarks actually present lowers the score slightly. In the square
checkpoints the diagonal is nominally higher (0.026 vs 0.017; 0.054 vs 0.040),
but that gap is produced by the single room-0 cell, which is the cell whose
triad happens to carry the largest offset window. Remove nothing and read across
the diagonal instead: 0.053 / 0.004 / 0.022 for the 3-room square net. Landmark
presence does not predict the score.

Two further observations that point the same way, both confirmed:

- `e1_multi_anchor.py` hardcodes `layouts="rooms"` (it exposes `--env` but not
  `--layouts`), so on a pool-trained checkpoint it probes a room from the frozen
  set. Neither `ROOMS_RUN1[0]` nor `ROOMS_SQUARE[0]` (both layout key
  `0107cd84`) is in its 500-layout pool. **The two pool nets therefore score
  0.145 on a layout they never trained on** — higher than either 3-room net
  scores on layouts it trained on for the entire run.
- Consequently the OWN column here is **not** the same measurement as the
  fraction in §2 for the pool runs: §2 probes pool room 0
  (`21fbce58` / `73bcc621`), §3 probes `0107cd84`. 0.145 vs 0.022 and 0.145 vs
  0.010 are two different rooms, not a contradiction.

🔴 **`e1_multi_anchor.py` prints a verdict rule it does not check.** Its output
reads "the signal is landmark-driven only if OWN clearly exceeds OTHER", and on
`ROOMS_SQUARE` at `--room 0` it prints exactly that pattern from an offset-set
mismatch it also prints but never compares. It is not wrong about any number —
`n_offsets` is in the output and in the JSON — but a reader following the
printed rule reaches a false positive. **Not fixed here**: whether OWN and OTHER
should be scored over the intersection of their offset sets, or the control
restricted to triads with matched signatures, is a methodological choice and
belongs with the user, not in a results run. Nothing else in this document
depends on which way it is resolved — the `--room` sweep answers the question
without it.

## 4. Displacement across rooms, EARLY vs FINAL

Rank units in the home room, then display those same units with the triad moved
(`scripts/trace/e1_across_rooms.py`). `vector_score` per room, EARLY from
`outputs/summary/fig_e1_rooms_<tag>.json`, FINAL from
`fig_e1_rooms_final_<tag>.json`. Units are re-ranked at each checkpoint, so unit
numbers are not generally comparable between the two blocks — #222 in the 3-room
L-room arm happens to appear in both, at a different rank. The comparison is of
the *pattern*, not of identities.

```
                        EARLY                                 FINAL
                   room0   room1   room2                 room0   room1   room2
3-room SQUARE  #421 +0.792  +0.364  -0.081          #75  +0.797  -0.004  +0.000
               #128 +0.654  +0.252  -0.142          #140 +0.783  +0.209  -0.063
               #109 +0.722  -0.039  -0.168          #61  +0.747  -0.027  +0.684
500-pool SQ    #162 +0.391  +0.001  +0.221          #107 +0.668  +0.286  +0.629
               #283 +0.498  +0.362  -0.056          #398 +0.663  +0.685  +0.336
               #291 +0.667  +0.195  -0.006          #224 +0.741  +0.863  +0.931
3-room L-room  #469 +0.857  +0.772  +0.807          #222 +0.801  +0.178  +0.062
               #177 +0.605  +0.091  +0.264          #423 +0.758  +0.034  +0.119
               #222 +0.774  +0.463  +0.434          #314 +0.710  +0.570  +0.749
500-pool L     #189 +0.590  +0.414  +0.350          #405 +0.721  +0.308  +0.394
               #437 +0.597  +0.492  +0.569          #105 +0.623  +0.463  +0.595
               #372 +0.622  +0.404  +0.543          #275 +0.725  +0.343  +0.130
```

The 3-room square arm is the informative one (§7), and its verdict is unchanged
and if anything sharper: two of its three top units collapse to zero or negative
when the triad moves (#75 to -0.004/+0.000, #140 to +0.209/-0.063). These are
place-like or band-like units whose within-room agreement was coincidence. #61
holds in room 2 (+0.684) but not room 1 (-0.027), which no vector code
explains.

Inspected directly (`fig_e1_rooms_final_square3.png`): #75's room-0 map has no
visible localised field at all, while its room-1 map has a bright blob sitting
on the displaced yellow `block3` landmark; #140 is dominated by a bright
left-edge column; #61 is a horizontal band that moves between rows across
rooms. **The top-ranked candidates in this arm are wall-aligned bands and edge
columns, not landmark-anchored fields.** That is consistent with §3 and with §5,
and it makes the untouched border/BVC exclusion (E5) the binding gap (§7).

The 500-pool square arm now has a unit (#224) that holds across all three rooms
(+0.741/+0.863/+0.931), where on 08-13 no square unit did. Read together with
§3 — that same net scores 0.145 at a triad whose landmarks are absent — the
parsimonious reading is room-wide banded structure that correlates at any triad,
not a vector code. **Inferred, not confirmed.** What would confirm it: the E5
border/boundary-vector exclusion on #224, and a repeat of the criterion with the
landmarks removed from the rollout entirely.

**Figures.** `outputs/summary/fig_e1_rooms_final_lroom3.png`,
`fig_e1_rooms_final_lroom500.png`, `fig_e1_rooms_final_square3.png`,
`fig_e1_rooms_final_square500.png`.

## 5. Positive control: it no longer shows what it showed on 08-13

Median spatial information per checkpoint, and the top unit, printed by
`e1_cell_figure.py` and plotted in `outputs/summary/fig_e1_gallery_final_*.png`:

```
                    EARLY median    FINAL median   FINAL top
3-room L-room          0.769            0.918         3.499
500-pool L-room        0.679            0.667         3.551
3-room SQUARE          0.693            0.933         2.835
500-pool SQUARE        0.645            0.635         3.094
```

The 3-room arms gained spatial information; the pool arms did not move.

⚠️ **But the number is no longer measuring what the 08-13 document invoked it
for.** That document's §5 reads "clean, localised place fields in every net",
and that is what makes a null a bound rather than a broken measurement. At
482M this is no longer what the panel shows. Inspected directly
(`fig_e1_gallery_final_lroom3.png` and `fig_e1_gallery_final_square3.png`): the
most spatially informative units are dominated by **wall-parallel bands, edge
columns and boundary strips** — a top row, a left column, a bottom band, a
corner arc — with only a small minority reading as compact interior fields. The
square net's panel is almost entirely stripes.

This is a confirmed observation from the two figures named, on the top-16-by-SI
units only; it is **not** a measured claim about the whole population, which
would need the E5 border/BVC criterion actually run. Stating it plainly because
it cuts two ways: a wall-parallel band is genuinely spatially informative, so
the SI median rising is real — and a wall-parallel band is exactly the structure
that correlates across arbitrary anchor crops, which is the mechanism §3
measures.

## 6. Why this is still a bound and not a verdict

**The detector fails its own Stage 0 gate.** Committed characterisation, read
from `outputs/ovc/stage0_lroom.json` and plotted in
`outputs/summary/fig_ovc_power.png`:

```
negative control (odd/even)      0.014      want <= 0.05   PASS
specificity, injected place      0.012/0.006               PASS
positive, injected vector field  amp 0.5 -> 0.028
                                 amp 1.0 -> 0.062
                                 amp 2.0 -> 0.210
                                 amp 4.0 -> 0.575          want >= 0.90  FAIL
```

Specific but not sensitive: 42.5% of injected vector fields at 4x the unit's own
mean rate are missed. **Every null in this document is therefore a bound on
large effects only.**

⚠️ **That characterisation is on the 80k-step single-room L-room net, not on any
checkpoint scored here.** The sensitivity floor is inherited, not measured. The
file is dated before all four of these runs finished, and §5 shows the
population these nets now carry is not the population Stage 0 was characterised
on. Re-running Stage 0 per checkpoint remains the first thing any successor
should do, and it matters more now than it did on 08-13.

**And OVC-4 is a vacuous screen** — carried forward unchanged from 08-13 §4: the
spatial-information screen kept 500/500 units on the 3-room L-room net, changing
no denominator, because a trajectory-shuffle null destroys all spatial structure
and every spatially tuned unit clears it trivially. Not re-measured at 482M.
Recorded against OVC-4 in the instructions.

## 7. Standing caveats, and what could not be verified

⚠️ **`ROOMS_RUN1`'s L-rooms are exact translations of one configuration**, so
the L-room 3-room arm is a weak manipulation. Re-verified against the source
this session: room 0 → room 1 is a shift by (0,3), room 0 → room 2 by (3,0), all
three at separation signature (6,6,6). A network can cover all of them with one
map plus a translation, so remapping is never required there.
`ROOMS_SQUARE` has signatures (6,6,6)/(6,6,9)/(6,6,8) and admits no translation
between any pair — it is the arm that carries interpretive weight in §4.
This is the same property that produces the offset-count mismatch in §3: the
L-room's congruence is what makes its location control comparable, and the
square room's non-congruence is what makes its control confounded. One design
choice, opposite consequences for the two tests.

**All four checkpoints are at exactly 482,344,960 environment steps.** The
training-length confound that forced 08-13 §2.1's step-matched re-analysis is
gone. What is *not* matched between the arms is the probe size — 172
trajectories in the L-room against 196 in the square room (§1) — a consequence
of room geometry that no step-matching removes.

**Not verified.** The E5 border / boundary-vector exclusion has still not been
run, and §4 and §5 both now turn on it: the top candidates and the top-SI units
in both 3-room arms read as wall-aligned structure, and nothing here separates a
boundary-vector cell from an object-vector cell. Stage 0 has not been re-run on
these checkpoints. And whether the §3 column effect is driven by the offset
window's size or by the triad's shape is not separated by these data.

## 8. Reproducing

Checkpoints, `<job>` and `<run>` as tabulated in §1:

```bash
rsync -avh mila:'$SCRATCH'/pRNN/<job>/<run>/checkpoints/predictiveNet_state_step0482344960.pt \
    outputs/final-<lroom3|lroom500|square3|square500>/
```

Analyses, with `<env-config>` `lroom_multi` for the L-room arms and
`squareroom_multi` for the square arms, `<layouts>` `rooms` for the 3-room arms
and `pool` for the pool arms, and `<tag>` one of
`final_lroom3 final_lroom500 final_square3 final_square500`:

```bash
CKPT=outputs/final-<name>/predictiveNet_state_step0482344960.pt
uv run python scripts/trace/e1_cell_figure.py  --ckpt $CKPT --env-config <env-config> \
    --layouts <layouts> --label "<what it is>" --tag <tag>
uv run python scripts/trace/e1_across_rooms.py --ckpt $CKPT --env-config <env-config> \
    --layouts <layouts> --label "<what it is>" --tag <tag>
uv run python scripts/trace/e1_multi_anchor.py --ckpt $CKPT --env <env-config> \
    --room 0 --tag <tag>
for room in 1 2; do
  uv run python scripts/trace/e1_multi_anchor.py --ckpt $CKPT --env <env-config> \
      --room $room --tag <tag>_room$room
done
```

The `--room` loop is what §3 turns on; `--tag` must be passed because
`e1_multi_anchor.py` defaults it to `<env>_room<room>`, which collides between
the 3-room and pool checkpoints of the same geometry.

The geometric facts in §3, which need no checkpoint:

```bash
uv run python -c "
from curious_george.envs.layouts import ROOMS_RUN1, ROOMS_SQUARE, base_walkable, common_offsets, \
    BASE_ROOM_ID, SQUARE_ROOM_ID
for name, rooms, rid in (('ROOMS_RUN1', ROOMS_RUN1, BASE_ROOM_ID),
                         ('ROOMS_SQUARE', ROOMS_SQUARE, SQUARE_ROOM_ID)):
    w = base_walkable(rid)
    print(name, [len(common_offsets(walkable=w, anchors=l.anchors)) for l in rooms])
"
```

Every figure writes a sibling `.json` carrying the checkpoint path, env, room
description, landmark list, chosen units and the population fraction.
