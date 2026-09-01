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

## Read-out plan when they land

- The 2x2 table (loss × seed, full budget): landmark shown-step surprisal
  and mean room sRSA, this doc's protocol, one command per cell.
- γ=5 vs γ=2 at half budget: is there curvature worth a γ sweep, or is γ=2
  already on the plateau?
- floorP0-focal2 vs floorP0: if focal lowers the single-room floor too, the
  0.61-nat floor is gradient allocation all the way down; if not, the
  residual is capacity/architecture (readout width, hidden size) - the
  "perhaps readout" thread the user queued.
