2026-09-10 · branch `sdu/mixed-count-mse` @ `24a309a` · minigrid `c5dd5f2` · prnn `852cc7d2`

# Circuit x loss x object count on the 8-room impassable arm

## Question

Three things are varied here, and only the first is new to this repository as a
production axis:

1. **The circuit** (`arch_prnn.action_offset`). Offset 0 pairs `obs[t]` with
   `a[t]`, the action chosen *after* seeing it, and hands the policy `h[t-1]`.
   Offset 1 pairs `obs[t]` with `a[t-1]`, the action that *produced* it, and
   hands the policy `h[t]`. `docs/figures/circuit-offset0.png` and
   `circuit-offset1.png` draw both from one checkpoint on one trajectory.
   Every run in this repository to date is offset 0, which is the dataclass
   default; the audit raised that as C1 and the cleanup left it as an open
   methods question.
2. **The prediction loss** — MSE on pixels, or focal-5 CE with the MLP readout.
3. **The object count per room** — every room at three landmarks, or the
   mixed-count design, `keep_landmarks = ("012","012","012","12","01","0","2","-")`
   over positions `(0,1,2,3,5,6,7,8)`: counts 3,3,3,2,2,1,1,0, colour-balanced
   at five each of blue, green and red.

The question the circuit axis answers: **does pairing an observation with the
action that produced it, and letting the policy act on the state that already
represents the current position, change what the world model learns on the
multi-room arm?** The single-room A/B (`docs/action-offset-ab-2026-08-29.md`)
could not answer it — see Controls.

The question the count axis answers: does a room set whose landmark count
varies, including a room with none, produce a different spatial
representation than one where every room has three?

## Method

Twelve runs, one seed (2), all in wandb project `curious-george-multienv`,
all at the `multienv-fast` budget — 43,936 world-model and 175,744 policy
gradient steps, 89,980,928 environment steps — on Mila L40S nodes. Launched
through `slurm/multienv.sh`, whose eleventh argument (`keep`) is new in this
branch: `keep_landmarks` is a SOURCE-level flag and must follow
`env.source:selected`, so it could not ride the existing `extra` passthrough.

```
sbatch slurm/multienv.sh true 8 2 sdu/mixed-count-mse '' '' '' '' 0,1,2,3,5,6,7,8 \
    <label> <keep> <loss flags> --arch-prnn.action-offset <0|1> \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT \
    --eval.plot-every-steps 3333328
```

`keep` is `''` for the fixed-count arms and `'012 012 012 12 01 0 2 -'` for the
mixed-count ones; the run name gains `-mixed` from the launcher when it is set,
so an arm and its twin cannot collide. The offset is passed EXPLICITLY on both
sides, including where it equals the default, so the changed variable is visible
in every `provenance.json` argv.

| cell | width | jobs (offset 0, offset 1) |
|---|---|---|
| fixed count, MSE | 500 | 10755348, 10755352 |
| fixed count, MSE | 1024 | 10755349, 10755353 |
| fixed count, CE focal-5 MLP | 500 | 10755350, 10755354 |
| fixed count, CE focal-5 MLP | 1024 | 10755351, 10755355 |
| mixed count, MSE | 500 | 10755356, 10755358 |
| mixed count, MSE | 1024 | 10755357, 10755359 |

All twelve command lines were parsed through `curious_george.configs.cli` and
asserted against their own run names before submission — offset, loss, width,
resolved landmark counts, room count, budget and reward normalisation.

**What is held fixed**: seed, budget, room set (positions `0,1,2,3,5,6,7,8`,
impassable), reward normalisation on, whitened advantages, `entropy_coef` at
the preset's 0.035, `probe_seed` 10007, and 26 "Observation Sequence"
prediction images per run.

**What the online eval sees**: `multienv-fast` sets `rooms_max=5` and
`training/loop.py` scores a strict PREFIX, so `multiroom/mean_room_sRSA` covers
rooms 0-4 — counts 3,3,3,2,2 in the mixed arms. The count-1 and count-0 rooms
are trained on and NOT scored online. That is deliberate: it keeps the headline
metric comparable to every earlier run on this arm. The count gradient is read
offline instead, from the ten archived checkpoints:

```
uv run python -m curious_george.evaluation.checkpoint_series \
    --run <run> --rooms-scored 8 --spatial
```

(the flag is `--rooms-scored`; `--spatial` is what turns on sRSA/SWdist, which
this scorer leaves off by default because it is the slow part)

## Controls, and what each one is for

- **The circuit's own control is the other offset**, same seed, same code, same
  day, same GPU class — which is what makes this a 2x2x(2+1) design rather than
  a comparison against history.
- **The fixed-count arms are the mixed-count arms' twins**: identical but for
  `keep_landmarks`.
- **The 2026-09-09 series** (`mse-clean`, `mse-h1024`, `focal5mlp-clean`,
  `focal5mlp-h1024`, code `04cdc42`) are the offset-0 fixed-count cells run a
  day earlier under the same protocol. They are a REPLICATION check on the four
  offset-0 fixed-count cells here, not a substitute for them — the minigrid pin
  moved between the two (fingerprint-identical on every gated path, but a
  different commit).
- **Seed spread**: the only measured seed-to-seed control on this arm is the CE
  pair `mx-impassable-n8-s2/s3-focal5mlp`, which sits outside the within-run
  band at 23 of 48 matched points and differs by 0.05 on final mean room sRSA.
  ⚠️ There is NO MSE seed control anywhere, and none of tonight's cells has a
  second seed. Any effect smaller than ~0.05 on mean room sRSA is not separable
  from seed noise by this design. This is the design's main weakness and the
  cheapest thing to fix next.
- **Negative control, already on the record**: a random forward-weighted walker
  scores `exploration/coverage` ~0.36 and nAUC 0.206 on this room set
  (`uv run python -m curious_george.envs.action_graph`). A learned policy below
  that is not exploring.

## Predictions

Stated before the runs finish, so they can be wrong.

- **Circuit.** No effect on `multiroom/mean_room_sRSA` larger than the 0.05 seed
  spread. The single-room A/B found no representation metric that separated the
  two circuits at n=2, and its one reproducible offset-1 signature — a transient
  exploration collapse — was at `entropy_coef=0.001` in the RAW-advantage era.
  These runs whiten and use 0.035, so the mechanism that produced the collapse
  should be absent. **The measurement that would falsify this**: `policy_entropy`
  MIN and `loc_entropy` MIN, which every run logs. Offset-1 runs spending any
  meaningful fraction of updates below 1.0 bits would mean the collapse survives
  into the whitened era, which is unmeasured today.
- **Prediction loss under offset 1.** Slightly HIGHER than offset 0 at equal
  entropy is possible but should be small: offset 1's segment has one more row
  to predict and its first row has no preceding action to condition on. The A/B
  measured ~20% at entropy 0.001 and a TIE (0.00429 vs 0.00431) at 0.01, so at
  0.035 the expectation is a tie.
- **Loss.** CE focal-5 MLP above MSE on mean room sRSA, reproducing
  `docs/ce-multienv-impassable-2026-08-31.md` and the 2026-09-09 replication
  (0.784 vs 0.660), with MI_policy moving the other way.
- **Width.** Prediction improves at 1024 under both losses; mean room sRSA does
  not (CE ended 0.046 below its twin on 2026-09-09, MSE inside band).
- **Count.** No prediction — this is the first honest measurement of it. The
  2026-09-01 attempt does not count; its landmark-free room was the default
  6x6 walkable room (`docs/invalid-runs.md`, 2026-09-10). Two mechanisms pull
  opposite ways: fewer landmarks in some rooms means less to anchor on, which
  should lower sRSA; but a room set whose count VARIES cannot be solved by
  memorising a fixed three-landmark template, which should raise it.

## Code changed for these runs

Three defects, each named where it lives. All are in the branch, gated at
651 passed / 1 deselected against a 647-passed baseline, with the training-path
fingerprint IDENTICAL to `throwaway/2026-09-06/fp/base1` on prod, walkable,
parity and reference.

1. **minigrid `c5dd5f2`** — `landmarks or defaults` made an empty landmark list
   mean "the historical three". See `docs/invalid-runs.md`, 2026-09-10.
2. **`envs/vector.py::_cache_layout_grids`** — one landmark-free room disabled
   the pool's cached-grid reset for every room. The predicate is per layout now,
   and the cached-vs-full equivalence is asserted, not argued.
3. **`evaluation/prediction_figures.py`** — the circuit figure had no committed
   driver, raised on any CE checkpoint, called its error `mse` under a loss where
   it is surprisal, and was not on seeded-probe protocol v2 (two traces of one
   trajectory started in different places and diverged after three steps with a
   bit-identical torch stream). `python -m curious_george.evaluation.prediction_figures
   --run <dir> --offsets 0 1` regenerates both figures.

Also found and NOT fixed, because it is outside this task and the rule is to ask
first: in the minigrid fork, the obstacle suite's
`test_render_cache_does_not_alias_the_two` fails at `22ef960` and still fails — its second assertion asks a red `Wall` to
render differently from a red `Obstacle`, but at `tile_size=1` every tile is a
solid fill of its own colour, so that probe cannot discriminate. The baseline is
63 passed / 1 failed before this branch and 65 passed / 1 failed after.

## Results

All twelve COMPLETED (SLURM 0:0), 22-27 min each on L40S. Run names are
`mx-impassable-n8-s2-<label>_curious_<timestamp>` in `curious-george-multienv`,
with `<label>` as in the job table above.

🔴 Every sRSA number below is a TAIL MEAN over the last three analysis points
with the run's own adjacent-sample band beside it, never a final logged point -
the rule `docs/action-offset-ab-2026-08-29.md` sets in red, after endpoint
reading produced three wrong conclusions there.
Recomputed by `curious_george.check.run_tails`, which reports each tail
beside its band and refuses a metric it cannot read densely rather than
returning a whole-run mean dressed as a tail:

```
uv run python -m curious_george.check.run_tails \
    --project curious-george-multienv --prefix mx-impassable-n8-s2-
```

### The circuit does not collapse exploration at whitened entropy 0.035

The prediction that could have been falsified, and the reason the circuit axis
was worth a night:

| | `policy_entropy` MIN | % updates < 1.0 bits | `loc_entropy` MIN |
|---|---|---|---|
| tonight, offset 0 (6 runs) | 1.43 - 1.65 | **0.0%** | 6.57 - 6.98 |
| tonight, offset 1 (6 runs) | 0.99 - 1.63 | **0.0%**, one cell 0.1% | 6.20 - 6.76 |
| A/B 2026-08-29, offset 1 @ entropy 0.001 | 0.39 - 0.74 | 1.9 - 21.1% | 2.78 - 2.84 |

CONFIRMED. The transient collapse that was offset 1's only reproducible
signature in the raw-advantage era does not occur here.

The mechanism is still present and subcritical, which is the more interesting
half: offset 1 has a lower `policy_entropy` MIN than its offset-0 twin in
**6 of 6** cells and a lower `loc_entropy` MIN in **6 of 6**. Twelve of twelve
in the direction the A/B's mechanism predicts is not noise. The circuit still
sharpens the policy on its state; at this entropy it never parks.

### CE over MSE, replicated 4 of 4

sRSA tail-3, bands ~0.05:

| width, circuit | CE focal-5 MLP | MSE | gap |
|---|---|---|---|
| 500, offset 0 | 0.786 | 0.632 | +0.154 |
| 500, offset 1 | 0.792 | 0.568 | +0.224 |
| 1024, offset 0 | 0.777 | 0.527 | +0.250 |
| 1024, offset 1 | 0.828 | 0.663 | +0.165 |

The only axis tonight that separates cleanly, and it reproduces
`docs/ce-multienv-impassable-2026-08-31.md` under a third protocol.

### Circuit, count and width do not separate on the headline

Circuit (offset 1 - offset 0) on sRSA tail-3: **+0.006, +0.051, -0.064,
+0.136, +0.070, -0.036** - four up, two down.
Count (mixed - fixed), MSE cells: **-0.039, +0.095, +0.118, -0.054** - two up,
two down.

Mixed signs at magnitudes comparable to the ~0.05 seed spread. Read as "no
effect demonstrated", NOT "no effect": with one seed per cell this design
cannot tell them apart, which the Controls section said in advance.

### Offset 1 costs prediction under CE, and not under MSE

The focal-weighted `pRNN loss` tail reads 0.0227 vs 0.0128 for the CE pair -
a 77% penalty - and that number should not be quoted. Audit C12: on focal arms that
metric IS the focal-weighted loss, reweighting by `(1-pt)^5`, and offset 1
contributes one maximally-hard row per segment. Against plain surprisal
(`cur_reward_mean`, which `PRNNAdapter._prediction_errors` computes unweighted
under either loss), by `check.wandb_compare` on matched env steps:

| pair | metric | offset 0 -> offset 1 (end) | band | outside |
|---|---|---|---|---|
| CE 500 | `cur_reward_mean` | 18.34 -> 21.62 nats (+18%) | 2.36 | last two points |
| CE 500 | `pRNN loss` (focal) | 0.0101 -> 0.0199 | 0.0202 | 0 of 6 |
| MSE 500 | `cur_reward_mean` | 0.0116 -> 0.0120 | 0.0009 | 0 of 6 |
| MSE 500 | `pRNN loss` | 0.0120 -> 0.0118 | 0.0009 | 2 of 6, mid-run only |

So the cost is real under CE (+18%, not +74%) and absent under MSE. INFERRED,
not confirmed, and the usual structural story does NOT cover it: one
chance-level row (49 tiles x ln 7 = 95 nats) among 256 rows at ~18 nats moves
the mean by ~1.7%, not 18%. One seed per cell; the remaining gap is unexplained.
⚠️ The `pRNN loss` band of 0.0202 is inflated by the early decay from 0.078,
so "inside band" is weak evidence on that row - which is itself a reason to
read the plain measure.

### The count gradient, which is what the mixed-count design is for

Per-room values at the final checkpoint, from
`checkpoint_series --rooms-scored 8 --spatial`. `room_sRSA` had to be added:
the scorer computed the per-room list, averaged it, and discarded it.

Prediction loss falls monotonically with landmark count in every run (the
500-wide offset-0 run shown; the other three have the same shape):

| room | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| landmarks | 3 | 3 | 3 | 2 | 2 | 1 | 1 | 0 |
| loss | .0087 | .0098 | .0092 | .0067 | .0065 | .0056 | .0054 | **.0039** |

Expected - an emptier room holds less to predict - and it confirms the design
renders what it claims. Independent check: room 7's layout fingerprint is
`da39a3ee`, the first eight hex of `sha1("")`.

sRSA by landmark count, averaged over the rooms at each count:

| run | 3 lm | 2 lm | 1 lm | 0 lm | 2-3 | 1-3 | 0-3 |
|---|---|---|---|---|---|---|---|
| mixed-mse-off0 | 0.540 | 0.568 | 0.560 | 0.535 | +0.028 | +0.020 | -0.005 |
| mixed-mse-off1 | 0.612 | 0.665 | 0.659 | 0.642 | +0.054 | +0.048 | +0.030 |
| mixed-mse-h1024-off0 | 0.621 | 0.629 | 0.631 | 0.583 | +0.008 | +0.010 | -0.038 |
| mixed-mse-h1024-off1 | 0.449 | 0.480 | 0.488 | 0.450 | +0.032 | +0.039 | +0.001 |
| | | | | **sign** | **4/4** | **4/4** | 2/4 |
| | | | | **mean** | **+0.030** | **+0.029** | -0.003 |

An INVERTED U: rooms holding one or two landmarks carry more spatial structure
than rooms holding three, in 4 of 4 runs, across two widths and both circuits;
the landmark-free room falls back to the three-landmark level. These are
WITHIN-run, within-checkpoint contrasts - one network, one probe, different
rooms - so they do not pay the between-run seed penalty that flattens every
other axis tonight.

🔴 **Count is confounded with room identity here, and the design cannot separate
them.** Each count is realised by one to three SPECIFIC rooms, so "two
landmarks" and "rooms 3 and 4" name the same thing. The zero-landmark cell
carries a second confound: with nothing impassable it has 172 reachable cells
against 152, so its sRSA is taken over a different support. The reproducibility
across four runs says the ordering is a property of these rooms, not of the
seed; it does not say the property is the COUNT.

The design that separates them is available and cheap, because `keep_landmarks`
acts on a fixed set of anchors: hold ONE room and vary its own count
(`"012"`, `"12"`, `"0"`, `"-"` on the same anchors), so count moves and
geometry does not. That is the next run worth its allocation, ahead of a second
seed on anything here.

### Prediction scorecard

| prediction | outcome |
|---|---|
| no offset-1 exploration collapse at 0.035 | **CONFIRMED** (0.0% duty cycle, 11 of 12 runs) |
| circuit moves sRSA by less than the 0.05 seed spread | **FALSIFIED AS STATED** - 4 of 6 cells exceed 0.05 - but signs are mixed, so no directional effect |
| offset-1 prediction loss ties offset 0 at this entropy | **WRONG under CE** (+18% plain surprisal); right under MSE |
| CE above MSE on mean room sRSA | **CONFIRMED**, 4 of 4 |
| prediction improves at 1024 under both losses | partly - 4 of 6 cells |
| count: no prediction stated | the inverted U above, confounded as noted |

