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

**Wave 2b (all COMPLETED ~03:25; jobs 10577305-09, submitted ~02:50 at
6593723, entropy 0.035 per the Phase-1 scan). Final summaries, seeded
probes (`scratchpad/phase2_readout.py`):**

| arm | wm loss (nats/tile) | room sRSA | pooled | remap | SWdist | SI | MI | locH | cov |
|---|---|---|---|---|---|---|---|---|---|
| ce8 curious s2 | 0.299 | 0.625 | 0.443 | +0.183 | 0.112 | 0.574 | 0.067 | 7.23 | 0.287 |
| ce8 curious s3 | 0.357 | 0.543 | 0.392 | +0.151 | 0.080 | 0.588 | 0.059 | 7.26 | 0.284 |
| ce8-count | 0.248 | **0.659** | 0.502 | +0.157 | 0.070 | 0.666 | 0.014 | 7.36 | 0.163 |
| ce8-rndfwd | 0.194 | 0.607 | **0.510** | +0.097 | 0.095 | 0.688 | — | 7.13 | 0.361 |
| ce8-rnduni | 0.244 | 0.573 | 0.454 | +0.119 | 0.085 | 0.709 | — | 7.31 | 0.179 |
| *(2a)* mse8 s2 | *0.0143 (MSE)* | 0.517 | 0.327 | +0.191 | 0.059 | 0.518 | 0.054 | 6.99 | 0.225 |
| *(2a)* mse8 s3 | *0.0129 (MSE)* | 0.502 | 0.312 | +0.190 | 0.050 | 0.539 | 0.058 | 7.16 | 0.234 |

**What holds (confirmed, n=2 CE-curious vs n=2 MSE-curious, seeded probes):**

- **CE lifts the multienv representation across the board**: every CE arm's
  room sRSA (0.543-0.659) is at or above both MSE baselines (0.502-0.517);
  CE-curious mean 0.584 vs MSE-curious 0.510; SI likewise (0.57-0.71 vs
  0.52-0.54). The morning deliverable - CE multienv training with
  above-baseline sRSA - stands.
- Both behavioral baselines matched the analytic calibration mid-run
  (rndfwd 0.368 vs 0.360, rnduni 0.180 vs 0.182): the exploration wiring
  is sound and the baseline numbers mean what they claim.
- Prediction figures by eye (`throwaway/figs_ce/phase2_ce8_s2_...png` +
  per-run wandb media): floor/wall/agent crisp everywhere; landmark binding
  is INTERMITTENT - step 96 reproduces the room's blue-left/red-right
  structure, neighbouring blind steps predict background where landmarks
  are. The model UNDERCALLS rare classes under room uncertainty.

**What does NOT hold yet (honest negatives):**

- Within CE, the CURIOUS policy has not separated from its baselines:
  count-based beats curious s2 (0.659 vs 0.625) and curious s3 (0.543)
  falls below rndfwd (0.607). MI stayed low (0.06-0.07 vs Phase-1's 0.36):
  at half budget on 8 rooms the policy never strongly commits.
- SWdist favours the MSE arms (0.05-0.06 vs 0.07-0.11) at this n; read
  against its known high estimator noise, but do not claim CE wins it.

**Next steps, ranked (all data-grounded):**
1. FULL-budget ce8 (43,936 wm steps, ~60 min): the intermittent landmark
   binding + wm loss 0.30-0.36 (vs single-room 0.16) reads as underfitting
   first; budget before machinery. ⛔ BLOCKED on the Mila OTP.
2. If full budget saturates with landmark recall still stalled while
   background saturates - that is the named focal-loss trigger
   (`sdu/focal-loss`, timeboxed).
3. The curious-vs-count gap is the interesting science: count explores less
   (cov 0.163) yet represents better - suggests visitation-targeted data
   beats error-targeted data for the pRNN here. A learning-progress reward
   is the principled contender; hold for the user.

## Wave 3 — full budget (launched morning 2026-08-31)

The ranked next step from the wave-2b reading: full budget (the preset's own
43,936 wm / 175,744 policy gradient steps, 89,980,928 env steps) to separate
underfitting from the class-imbalance story before any new machinery.

| arm | where | id / run stem | flags beyond wave 2b |
|---|---|---|---|
| ce8full s2 | local 4060 | mx-impassable-n8-s2-ce8full-local | *(budget only)* |
| mse8full s2 | local 4060 (queued after) | mx-impassable-n8-s2-mse8full-local | |
| ce8full s3 | Mila 10581798 | mx-impassable-n8-s3-ce8full | |
| mse8full s3 | Mila 10581799 | mx-impassable-n8-s3-mse8full | |
| ce8-countfull s2 | Mila 10581800 | mx-impassable-n8-s2-ce8-countfull | count vs curious at full budget |

**ce8full s2 (local) FINISHED, exit 0** — the new board leader:
room sRSA **0.668** (curve still rising at the end: 0.651 → 0.668 over the
last two events), pooled 0.472, remap +0.196, SWdist 0.088, SI 0.675,
MI **0.157** (2.3x the half-budget arm - the policy commits given budget),
wm loss 0.281 nats/tile, coverage 0.177. Above every half-budget arm
including count (0.659). Figure
(`throwaway/figs_ce/phase2_ce8full_s2_...png`): wall geometry crisp,
landmark binding improved but still partial (right class and place, wrong
extent) - and since sRSA had NOT plateaued, the next lever remains budget,
not focal.

**ce8-countfull s2 FINISHED (54:36)**: room sRSA 0.637, MI 0.014, SI 0.690.
🔴 **The curious-vs-count ordering FLIPS at full budget**: half-budget count
beat curious (0.659 vs 0.625); full-budget curious beats count (0.668 vs
0.637). The curiosity signal needs budget to pay off - the wave-2b "honest
negative" resolves in curiosity's favour, pending the s3 replication.

**Cluster s3 pair FINISHED (58:08 / 57:05)**: ce8full s3 room sRSA 0.628,
MI 0.282 (entropy 1.06 - the most committed multienv policy of the series),
pooled 0.491; mse8full s3 0.589, MI 0.065, pooled 0.347, remap +0.242. The
s3-vs-s3 room-sRSA gap narrows to +0.04 (a strong MSE seed), but the
structural contrast holds: CE pooled sRSA 0.44-0.49 vs MSE 0.33-0.35, and
CE gains representation WHILE committing 3-4x harder on MI.

Local runs carry `-local` in the name; comparisons stay on the gradient-step
axis (GPU type moves wall-clock, not curves - the dev box reproduces cluster
curves). Cluster tree reset to `7db61dd` BEFORE sbatch (the wave-1 lesson).
Mila auth recovered key-only (~morning); the OTP path was never needed again.

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
