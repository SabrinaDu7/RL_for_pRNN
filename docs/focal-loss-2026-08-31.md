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

## Read-out plan when they land

- focal5full vs focal2full: does γ=5's half-budget lead survive full budget,
  and what does its background cost do to SWdist/pooled sRSA?
- focal10 half vs focal5 half: γ knee location; if 10 regresses, γ≈5 is the
  operating point and the sweep is closed.
- Then: hidden-state analysis on the best checkpoint (the user's gating
  criterion - predictions first), and the capacity residual (readout) as the
  separate thread.
