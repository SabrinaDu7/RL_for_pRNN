2026-08-31 evening · branch `sdu/optim-pred` · runs at repo `3cb9196`+ / fork `600c7e09`+ · wandb project `curious-george-multienv`

# Focal CE: does reweighting the gradient fix landmark reconstruction?

## The question, and why it gates everything else

The pRNN's multienv predictions are not good enough to trust hidden-state
analysis: the model does not reconstruct the landmarks of rooms it has seen
millions of steps of (the user's framing: it "can't even memorize the
environments"). The audit-controlled measurement
(`evaluation/surprisal_timing.py`, eval_mode, plain-surprisal protocol) put
plain-CE landmark tiles at 1.49 nats/tile at SHOWN steps on the full-budget
checkpoint - near chance ln 7 = 1.95 - while background saturates ~0.16.
That fired the plan's focal trigger: a rare-class reconstruction failure,
focal's textbook case. `focal_gamma` reweights only the training gradient
((1-pt)^γ · CE); the curiosity reward stays plain surprisal.

## Protocol

Every number below is `uv run python -m curious_george.evaluation.surprisal_timing
--ckpt <run>/predictiveNet_state.pt` (eval_mode, fixed seeds, fwd-biased
walker, 8 rooms x 2 episodes; `--positions 0` for the single-room floor).
Landmark surprisal is nats/tile at the palette's four landmark classes;
"shown" = steps where the obs is in the pRNN's input (1-of-6 masking),
"blind" = masked steps. Chance ln 7 = 1.946. Checkpoints under `outputs/`
(local) and `outputs/fetched/` (pulled from Mila scratch); figures in
`docs/figures/surprisal_*.png`.

## Results - half budget (wm 21,968), matched protocol

| arm (checkpoint) | landmark shown | landmark blind | background | final mean room sRSA |
|---|---|---|---|---|
| **floor**: 1 room, plain CE, s2 (`floorP0`, job 10607209) | **0.610** | 0.739 | ~0.11 | n/a (single room) |
| 8 rooms, plain CE, s2 (`ce8-fastreset-check`, local) | 1.337 | 1.461 | ~0.14 | (STATE_SHA check run) |
| 8 rooms, focal γ=2, s2 (job 10607255) | **1.197** | 1.282 | ~0.27 | 0.666 |
| 8 rooms, focal γ=2, s3 (job 10607256) | **0.927** | 1.043 | ~0.27 | **0.711** |

Reference points from the full-budget era (same protocol, different budget):
plain CE full s2 (`ce8full-local`) = 1.490 shown / 0.16 background;
plain-CE full-budget sRSA board: 0.668 (s2), double-budget peak 0.704.

## What the numbers say (confirmed, from the table above)

1. **The failure decomposes into two stacked parts.** A single-room model
   still sits at 0.61 nats shown-step - the objective/capacity floor is NOT
   ~0 - and the 8-room plain-CE model adds ~0.73 nats of multienv
   interference on top (1.34 vs 0.61). "Can't memorize the rooms" is mostly
   the interference term, but not only.
2. **Focal attacks the interference term.** At matched (budget, seed):
   1.337 → 1.197 (s2). The s3 arm reached 0.927 - 60% of the way from
   plain-CE (1.34) down to the single-room floor (0.61).
3. **The expected trade is visible and mild**: background rose 0.14 → 0.27
   nats under focal - the gradient moved from easy tiles to rare ones, as
   designed. Background remains far from harming anything (chance 1.95).
4. **More budget does NOT fix this under plain CE - it makes it worse.**
   Full-budget plain CE (1.49) is worse at landmark reconstruction than
   half-budget plain CE (1.34): late in training, plain CE's gradient is
   increasingly spent on the already-saturated background. This is the
   mechanism the focal trigger predicted, and it explains why the sRSA
   budget curve peaked (~0.704 @ 90-105M) then oscillated.
5. **sRSA agrees**: focal-half s3 (0.711) already exceeds the plain-CE
   full-budget board (0.668) and touches the double-budget peak (0.704) at a
   QUARTER of that budget. n=2 seeds, so a band not a point - the full-budget
   2x2 below is the confirmation.

Caveat (inferred, one seed pair): the focal s2/s3 spread (1.20 vs 0.93) is
wide; per the noise-floor doc, treat single-seed deltas smaller than ~0.1
nats as unresolved. The s2-matched comparison (1.34 → 1.20) is the
conservative read; the 2x2 is the test.

## In flight (launched ~21:50, tree `e56d1bf` - audit protocol v2 era)

All in wandb project `curious-george-multienv`, labels as named:

| job | arm | label |
|---|---|---|
| 10611931/32 | focal γ=2, FULL budget, s2/s3 | `focal2full` |
| 10611933/34 | plain CE, FULL budget, s2/s3 (same-tree comparators) | `ce8full-v2` |
| 10611935 | focal γ=5, half budget, s2 (γ curvature probe) | `focal5` |
| 10611936 | focal γ=2 on the SINGLE room, s2 (does focal lower the floor itself?) | `floorP0-focal2` |

⚠️ The `-v2` relaunch of plain CE exists because the audit's probe protocol
v2 (docs/invalid-runs.md) moves seeded sRSA values: full-budget focal must
be compared against a same-tree plain-CE arm, not against the pre-audit
`ce8full` numbers. Surprisal numbers (this doc's protocol) are unaffected -
`surprisal_timing` carries its own protocol.

## Results - full budget 2x2 + probes (landed ~22:20, all COMPLETED 0:0)

Same protocol as above; checkpoints under `outputs/fetched/`, figures
`docs/figures/surprisal_*.png`. Background is the per-bin mean's typical
value.

| arm | landmark shown | landmark blind | background | mean room sRSA |
|---|---|---|---|---|
| plain CE full s2 (`ce8full-v2`, 10611933) | 1.346 | 1.416 | ~0.14 | 0.481 |
| plain CE full s3 (10611934) | 1.100 | 1.164 | ~0.12 | 0.571 |
| focal γ=2 full s2 (`focal2full`, 10611931) | **0.837** | 0.872 | ~0.19 | **0.670** |
| focal γ=2 full s3 (10611932) | **0.763** | 0.829 | ~0.19 | **0.664** |
| focal γ=5 HALF s2 (`focal5`, 10611935) | 0.928 | 0.978 | ~0.43 | **0.743** |
| floor + focal γ=2, 1 room s2 (10611936) | **0.464** | 0.484 | ~0.13 | 0.773 (own room) |

(⚠️ `ce8full-v2` sRSA is lower than the pre-audit `ce8full` 0.668 because
probe protocol v2 moved seeded sRSA values - exactly why the same-tree
comparators were run. Surprisal numbers are protocol-stable throughout.)

## Verdict (n=2 seeds on the 2x2; every cell recomputable by one command)

1. **Focal wins the full-budget 2x2 on both metrics, both seeds.** Landmark
   shown-step surprisal: 1.35→0.84 (s2), 1.10→0.76 (s3). Mean room sRSA:
   0.48→0.67, 0.57→0.66. No cell disagrees.
2. **Focal repairs the budget lever itself.** Under plain CE, going
   half→full budget made landmarks WORSE (1.34→1.35 s2; the pre-audit
   full-budget run measured 1.49). Under focal γ=2, half→full budget helps:
   1.20→0.84 (s2), 0.93→0.76 (s3). The gradient keeps buying landmark
   reconstruction instead of background polish.
3. **The "memorization floor" was mostly gradient allocation, not
   capacity**: focal lowers the single-room floor 0.610→0.464. A capacity
   residual (~0.46 nats) remains - the readout/hidden-size thread's target.
4. **γ curvature is real and not exhausted at 2**: γ=5 at HALF budget beats
   γ=2 half on both axes (0.93 vs 1.20 shown; sRSA 0.743 - the best sRSA
   measured in this project to date, at a quarter of the double-budget
   peak's spend). Its cost is background 0.27→0.43 nats - no longer free,
   still far under chance.

## In flight, round 2 (launched ~22:30, tree `b220fbd`)

| job | arm | label |
|---|---|---|
| 10612308/09 | focal γ=5, FULL budget, s2/s3 - the new best-candidate config | `focal5full` |
| 10612310 | focal γ=10, half budget, s2 - where does the γ curve turn? | `focal10` |

## The honest view: argmax recall (added after the gallery discussion)

The landmark gallery (`docs/figures/focal2full_s3_landmark_gallery.png`) is
the BEST case twice over: shown steps only, frames chosen for landmark
pixels. The user's read - "the pRNN doesn't learn predictions that well for
most of the predictions" - is correct, and `surprisal_timing` now reports
the blunt version alongside surprisal: recall = fraction of landmark tiles
whose argmax class is right; miss = fraction of landmark-bearing frames
whose prediction contains no landmark class at all. (torch stream now
seeded in `measure` - unseeded, repeat invocations wobbled ~0.01 nats;
values elsewhere in this doc are single draws inside that wobble.)

| arm | recall shown | recall masked | miss shown | miss masked |
|---|---|---|---|---|
| focal γ=2 HALF s2 | 0.394 | 0.352 | 0.310 | 0.396 |
| focal γ=5 HALF s2 | 0.513 | 0.434 | 0.216 | 0.277 |
| plain CE FULL s3 | 0.494 | 0.448 | 0.234 | 0.267 |
| focal γ=2 FULL s3 | **0.649** | **0.578** | **0.127** | **0.171** |

Readings (matched cells only): γ curvature holds on recall too (γ=5 > γ=2
at half/s2, +12pts); focal+budget compound to the best cell; but even the
best arm gets ~35% of landmark tiles wrong and misses the landmark
entirely in ~1 of 7 landmark-bearing frames. The shown-vs-masked gap is
consistently small (~6pts): the bottleneck is reconstruction, not memory
across masked steps. Hidden-state analysis on landmark binding should wait
for (or condition on) better recall; spatial/sRSA analysis is already
well-supported.

*(Bookkeeping, ~23:30: rounds 1-2 and the mlp pair were launched without
the `--run.wandb-project` flag the earlier waves carried, so those eleven
runs logged to the default `curious-george` project - run names
`mx-impassable-*` dated 26-08-31 21:42 onward. An API move was attempted
and silently no-ops (`run.update()` ignores project changes - verified:
the runs stayed put), so moving them needs the UI: select them in the
curious-george runs table -> Move -> curious-george-multienv. The launcher
now bakes the project in so this cannot recur.)*

## Results - round 2 (landed ~22:45, all COMPLETED 0:0)

| arm | landmark shown | recall shown/masked | miss shown/masked | background | mean room sRSA |
|---|---|---|---|---|---|
| focal γ=5 FULL s2 (10612308) | 0.725 | 0.664 / 0.609 | 0.107 / 0.139 | ~0.36 | 0.731 |
| focal γ=5 FULL s3 (10612309) | **0.712** | **0.677 / 0.593** | 0.117 / 0.154 | ~0.35 | **0.762** |
| focal γ=10 HALF s2 (10612310) | 0.968 | 0.516 / 0.422 | 0.170 / 0.281 | ~0.63 | 0.716 |

## Verdict, final

- **The γ knee is at ≈5.** γ=10 at half budget is flat-to-worse than γ=5 on
  every axis (sRSA 0.716 vs 0.743, shown 0.97 vs 0.93, recall equal) at
  ~1.5x the background cost (0.63 vs 0.43). Sweep closed.
- **Operating point: γ=5, full budget.** Best sRSA measured in the project
  (0.762 / 0.731, n=2 seeds, protocol v2), best landmark surprisal (0.71),
  best miss rate (~11%). Background ~0.35 nats - a real cost, well under
  chance, and SWdist/pooled sRSA show no damage from it.
- **Recall is plateauing under γ at ~65-68% shown.** γ2full → γ5full bought
  +0.06-0.10 sRSA but only ~+2pts recall. The remaining reconstruction
  deficit (a third of landmark tiles wrong; ~1 in 8 frames missing the
  landmark) is not a gradient-allocation problem any more - the readout /
  capacity thread is the ranked next lever, with budget-under-focal (still
  paying: γ5 half→full moved recall 51→66) as the cheap alternative.

## Next (queued, not launched)

- Readout thread: the linear 343-wide head off 500 hidden units is now the
  suspect for the ~0.46-nat single-room focal floor and the recall plateau.
- Double-budget γ=5 (the budget lever is repaired under focal; untried
  past full).
- Hidden-state analysis can start on `focal5full` s3 for SPATIAL questions
  (sRSA 0.762 is the best representation yet); landmark-BINDING analysis
  should still wait on recall.
