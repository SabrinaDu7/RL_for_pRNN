2026-08-31 · branch `sdu/optim-pred` · prnn pin `sdu/ce-loss` @ `600c7e09` · overnight, autonomous

# Phase 2: multienv training on 8 impassable-object rooms, MSE vs CE

Executing `docs/claude_logs/plan-ce-multienv-overnight-2026-08-31.md` Phase 2.
Project: **`curious-george-multienv`** (new; every run in it is bright-era,
whitened, seeded-probe). STATUS: IN PROGRESS.

## The room set

`ROOMS_SELECTED` positions (0, 1, 2, 3, 5, 6, 7, 8) = source-pool indices
0, 14, 31, 35, 126, 144, 169, 191 — source index 83 dropped (one cell from
index 0, the recorded near-duplicate; user call 2026-08-31), 191 promoted
from the spares. Impassable affordance; anchors identical to the walkable
twin by construction (`Selected`). Standable cells per room: 152.

## Exploration calibration at the launch tree (regenerated, per the handoff rule)

`uv run python -m curious_george.envs.action_graph` at `5cf2e65`:

```
=== impassable, denominator 152 ===
agent                     cov@256   nAUC     T25          T50          T90
uniform                     0.182  0.104   228 ( 11%)   censored     censored
forward-weighted            0.360  0.206   152 ( 90%)   234 (  4%)   censored
greedy sweeper              1.000  0.611    42 (100%)    92 (100%)   198 (100%)
```

Sanity rule: a forward-weighted RANDOM arm's online `exploration/coverage`
must sit near 0.36 from its first log points, else the wiring is wrong.

## Arms

Launcher: `slurm/multienv.sh true 8 <seed> sdu/optim-pred 21968 [agent] '' ''
0,1,2,3,5,6,7,8 <label> <extra...>` — half budget (~30 min L40S), whitened
preset defaults, `probe_seed` on.

| wave | arm | label / run stem | slurm job | extra flags |
|---|---|---|---|---|
| 2a | MSE curious s2 | mx-impassable-n8-s2-wm21968-mse8 | 10576566 | `--run.wandb-project curious-george-multienv` |
| 2a | MSE curious s3 | mx-impassable-n8-s3-wm21968-mse8 | 10576567 | same |
| 2b | CE curious | *(pending Phase-1 gate + entropy pick)* | | `+ --arch-prnn.loss CE --train-policy.normalize-reward` |
| 2b | CE random fwd | | | 2b's flags, agent arg `random` |
| 2b | CE random uniform | | | + `--arch-policy.random-action-probs 0.25 0.25 0.25 0.25` |
| 2b | CE count-based | | | `--train-policy.no-curious --train-policy.k-count 0.1` + CE flags |

🔴 Ordering honoured: wave 2a (bright-MSE) submitted 2026-08-31 ~02:00 (first wave ~01:50 mislaunched and cancelled - see the Phase-1 doc's deviations),
BEFORE any CE run exists in the project. Wave 2b launches after the Phase-1
gate (`docs/ce-single-room-2026-08-31.md`) reads out.

## Results

**Wave 2a (bright-MSE baseline, COMPLETED ~02:20, 32:22-32:42 elapsed):**

| run | wm loss | room sRSA | pooled sRSA | remap | SWdist | SI | MI | locH | cov |
|---|---|---|---|---|---|---|---|---|---|
| mx8-mse s2 | 0.0143 | 0.517 | 0.327 | +0.191 | 0.059 | 0.518 | 0.054 | 6.99 | 0.225 |
| mx8-mse s3 | 0.0129 | 0.502 | 0.312 | +0.190 | 0.050 | 0.539 | 0.058 | 7.16 | 0.234 |

The MSE-multienv pattern repeats Phase 1's: barely-committed policy
(MI ~0.056), moderate per-room sRSA. Notable: remapping index +0.19 - far
above the pale-era 5-room values (~0.02-0.06) - room-specific coding is
forming even under MSE in the 8-room bright set.

**Wave 2b (CE + baselines): jobs 10577305 (ce8 s2), 10577306 (ce8 s3),
10577307 (ce8-rndfwd), 10577308 (ce8-rnduni), 10577309 (ce8-count),
submitted ~02:50 at 6593723, entropy 0.035 per the Phase-1 scan.**
*(results pending)*

## Operational notes

- ~03:10: the Mila SSH control master died and reconnection now prompts for
  an email OTP (first occurrence on this machine; the runbook forbids
  automating or retrying credentials). Runs are unaffected - they are
  self-contained and rsync to $SCRATCH on exit; wandb carries all metrics
  and figures. BLOCKED until the user supplies the OTP: log tails, sacct,
  $SCRATCH checkpoint access (and with it the offline bump-contrast and
  per-tile decomposition), and any further launches. The user was notified.
- Mid-run sanity (~03:05, via wandb): ce8-rndfwd coverage 0.368 vs
  calibration 0.360; ce8-rnduni 0.180 vs 0.182 - both behavioral baselines
  sit exactly on the analytic references.

## Reading list when runs land

- Per-room sRSA (seeded probe), pooled sRSA, remapping index, SWdist, SI —
  CE vs MSE within-project.
- `exploration/*` vs the calibration rows above and vs the three behavioral
  baselines.
- Prediction figures by eye (argmax renders under CE).
- Bump events: prediction error at table-refused forward poses vs free moves
  (offline; the impassable-specific affordance-learning probe).
- Trajectory stats/plots per arm: wall-hugging, spinning, object-edge
  shuffling are the named pathologies to look for.
