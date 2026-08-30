2026-08-29 · branch `sdu/predict-next-obs` · commits `d2538a5`, `5d1a419`, `bb1e816`

# Does pairing obs[t] with the action that produced it improve the place code?

**Question.** The pRNN's input row `t` carries `a[t]`, the action chosen *after* seeing
`obs[t]`. Row `t+1` then needs `fwd(a[t])` to path-integrate but only receives `HD[t+1]`,
so `h[t]` is forced to carry a **pending-action bit** alongside "where I am" — a nuisance
variable in exactly the statistic sRSA measures. Setting `arch_prnn.action_offset = 1`
makes row `t` carry `a[t-1]` instead, and hands the policy `h[t]` rather than `h[t-1]`.

**Hypothesis.** Removing the pending bit from `h[t]` improves the spatial representation
(sRSA, SI, SWdist) at unchanged prediction loss.

## Method

Both arms ran the `mila-parity` configuration verbatim on one local RTX 4060, differing in
one integer. Standard single L-room (`EnvDefault` landmarks); the 256 environments are 256
copies of the same room, so batching is throughput, not diversity — there are no per-room
metrics and `configs.py:726` makes them unrepresentable here.

```
89,980,928 env steps | 43,936 pRNN and 175,744 policy gradient steps (identical in both)
device table + pooled world model + compile=LAYER + rollout/prnn/policy CUDA graphs
runtime 25.1 min per arm (the L40S cluster reference: 30.1 min)
```

n = 1 per arm. Runs: `offset0-parity_curious_26-08-29-02-10-39`,
`offset1-parity_curious_26-08-29-02-35-59`.

**The change is one integer, verified.** Fingerprinting every tensor that reaches
`pN.predict` under both settings: 17 of 18 identical (architecture, encoding, `predOffset`,
`actOffset`, `inMask`, weights, trajectory, observation input, target); only the action
input differs. `tests/golden` is bitwise green at offset 0, and the local loop measures
34.15 vs 33.94 wm_grad_steps/s before and after, so offset 0 is unchanged.

## Result: the mechanism worked, the outcome did not follow

**The mechanism is confirmed, decisively.** On each arm's own final checkpoint, over 12
segments x 256 steps:

| | offset 0 | offset 1 |
|---|---|---|
| `fwd(a[t])` decoded from `h[t]` (held-out balanced accuracy, chance 0.500) | **1.000** | **0.496** |
| end-of-segment MSE, relative to the segment median | **3.29x** | **1.14x** |
| MSE where `inMask` SHOWS the observation, relative to masked | 0.702 | 0.667 |

The pending-action bit is perfectly decodable from `h[t]` at offset 0 and *at chance* at
offset 1 — it is gone. The end-of-segment spike goes with it, as predicted: offset 1's tail
row carries a real action and a real head direction where offset 0's carries neither.

**The representation did not improve.** On matched environment steps
(`check/wandb_compare`, band = the reference's own adjacent-sample spread):

| metric | verdict | detail |
|---|---|---|
| `SWdist_onPolicy` | **improved** | 3 of 6 points outside band, **all** negative (lower is better): −0.072, −0.043, −0.050 over the second half. Final 0.0475 vs 0.0972. |
| `sRSA_onPolicy` | **no difference** | 2 of 6 outside band with **opposite signs** (+0.118, +0.115 early; −0.087 late). The curves cross. Final 0.664 vs 0.750 is within the oscillation the curve itself shows. |
| `mean SI_onPolicy` | **no difference** | 1 of 6 outside band, mixed sign. Active-only is 0.998 vs 0.996 — the same. |
| `pRNN loss` | **slightly WORSE, consistently** | 4 of 6 outside band, all positive. Final 0.0058 vs 0.0044. |

Endpoint summary against the L40S cluster reference (`mila-parity-e0.001`, a different
machine and commit — the local offset-0 arm is the control that matters):

```
metric                          reference    offset 0    offset 1
pRNN loss              control    0.00593     0.00442     0.00576
sRSA_onPolicy                     0.72787     0.74958     0.66390
mean SI_onPolicy                  1.03584     0.98633     0.96966
SI_mean_active_only               1.05914     0.99629     0.99759
SI_units_zeroed                        11           5          14
SWdist_onPolicy   (lower better)  0.04860     0.09720     0.04752
```
`mean SI` cannot be read alone: it averages a structural zero into every unit that fired in
fewer than `active_time_threshold` samples (`training/logging.py:115-119`). Offset 1 zeroed
14 units to offset 0's 5, which is why its mean is lower while its active-only mean is not.

## What was predicted wrongly

**The "control" moved, and the prediction that it would not was too strong.** Prediction
loss is consistently ~0.001 higher at offset 1. The two circuits do *not* pose the same
problem: offset 1's segment has one MORE row to predict (the tail row it no longer discards)
and its row 0 has no preceding action to condition on. "Same information either way" was
wrong; "one extra prediction and a slightly harder first row" is right.

## Two bugs found and fixed on the way

Both invisible at offset 0, because `init_sr` returns zeros there and never reads the
observation. Both would have silently corrupted the new arm.

- **`collect_rollout` reset the tracker before the environment**, so `h[0]` was built from
  the FINISHED episode's last view, at a position the agent had already left. Measured:
  `SR[L] == bootstrap(previous episode's last obs)`.
- **`init_sr` zeroed the whole action vector**, dropping `HD[0]` with it, so the rollout
  disagreed with the training pass at row 0.

`tests/test_action_offset.py` pins both, plus the shift itself and B=2-batched-equals-two-serial.

## Reading, and what would settle it

At n = 1 this is **not a win**: the mechanism did exactly what it was designed to do, and
sRSA did not follow. SWdist improved consistently and in one direction, which is the only
outcome signal here that survives its own noise band.

The honest next step is n = 3 per arm (~2.5 h total at 25 min a run) on sRSA and SWdist. If
SWdist holds and sRSA stays flat, the reading is that the pending-action bit was real but
was not what limited the place code — and the cleanup phase (`Circuit` enum, deleting
`pastSR`) should not be spent on it. `docs/prnn-io-alignment.md` remains superseded and
still needs rewriting either way.

---

# n = 2 (seed 3 added), and the cause

## The result at n = 2

| metric | off0 s2 | off0 s3 | off1 s2 | off1 s3 | reading |
|---|---|---|---|---|---|
| `pRNN loss` | 0.00442 | 0.00403 | 0.00576 | 0.00563 | **consistently ~35% higher, 2/2** |
| `sRSA_onPolicy` | 0.74958 | 0.75084 | 0.66390 | 0.84640 | **means equal (0.750 vs 0.755); offset 1's SPREAD is 0.18 against offset 0's 0.001** |
| `mean SI_onPolicy` | 0.98633 | 1.26356 | 0.96966 | 0.88127 | lower 2/2, but confounded by coverage |
| `SI_units_zeroed` | 5 | 6 | 14 | 5 | |
| `SWdist_onPolicy` | 0.09720 | 0.10939 | 0.04752 | 0.08152 | final points; on last-10 means (0.1033, 0.1175, 0.0843, 0.1724) it is 1/2, not 2/2 |
| `policy_entropy` MIN | 1.4490 | 1.5135 | **0.4910** | **0.3912** | **collapse reproduces 2/2** |
| `loc_entropy` MIN | 4.9425 | 4.8581 | **2.8448** | **2.7754** | **collapse reproduces 2/2** |
| % updates `policy_entropy` < 1.0 | 0.0% | 0.0% | **19.2%** | **21.1%** | the collapse is a TRANSIENT with a duty cycle, not an end state |
| MEDIAN `policy_entropy` | 1.746 | 1.752 | 1.657 | 1.583 | medians barely separate - which is why end-of-run metrics look normal |

All four runs: 25.0-25.1 min, 43,936 pRNN and 175,744 policy gradient steps.

**sRSA is not worse - it is UNSTABLE.** Offset 0 lands at 0.7496 and 0.7508, a spread of
0.001. Offset 1 lands at 0.6639 and 0.8464, a spread of 0.18. The seed-2 conclusion
("offset 1 is worse on sRSA") was an artifact of n=1; the real difference is variance, and
the variance has a cause.

## The cause: the policy collapses, and the circuit change is why

**The collapse is the most reproducible thing here** - `policy_entropy` falls to 0.49/0.39
bits (uniform over four actions is 2.000) and spatial coverage to 2.84/2.78 bits (~7 cells)
in BOTH offset-1 seeds, and in NEITHER offset-0 seed.

It is a TRANSIENT, not an end state, and that distinction is load-bearing: offset 1 spends
19.2% / 21.1% of logged updates below 1.0 bits locally and 1.9% on the cluster, while every
offset-0 run spends 0.0%. The agent dips and recovers. Median `policy_entropy` is 1.58-1.72
against offset 0's 1.75, and median `loc_entropy` 6.63-6.78 against 6.99-7.06 - barely
separated. Reading the MIN as the state of the run overstates it.

Ruled out as causes, by measurement:

- **Not a backend bug.** The device path and the CPU-table path produce identical rollouts
  at offset 1. That combination was untested - `test_device_collector.py` parametrizes over
  `reward_alignment`, not `action_offset` - and is now covered.
- **Not a biased reward.** Under a UNIFORM policy the per-action curiosity reward is flat
  in both circuits: stay/forward is 1.04x at offset 0 and 1.02x at offset 1. The 2.2x
  stay-put ratio seen during training is a selection effect OF the collapse, not its cause.

What the data does say: the correlation between `policy_entropy` and `loc_entropy` is
**+0.94 at lag 0** under offset 1 and **~0.00 at every lag** under offset 0. The circuit
change created that coupling.

**The mechanism is the change working, not failing.** `h[t]` is a strictly better basis for
choosing an action than `h[t-1]`: it is trained to be linearly sufficient for `obs[t]` and
it encodes the CURRENT position. Hand the policy that, with `entropy_coef=0.001` and
nothing else restraining it, and it can condition sharply on where it is - so it does, and
parks. Offset 0's staler state was acting as an accidental exploration regulariser.

Then the trap closes: the spatial metrics are `spatial_onpolicy`. A policy that stops
covering the room degrades both the MEASUREMENT and the world model's TRAINING
distribution. sRSA's instability is a behavioural collapse being sampled at whatever phase
the run happened to end in.

## What to do next

**Offset 1 with a higher `entropy_coef`.** The reference used 0.001; the `ultra` preset uses
0.01. `[[local-4060-reproduces-cluster]]` already records that 0.0005 vs 0.001 moved
`MI_policy` with the pRNN metrics unchanged, so this is the known knob for exactly this
axis. If coverage holds and sRSA stabilises at or above offset 0, the circuit is good and
only needed its exploration re-tuned - and the prediction-loss gap (which is structural:
offset 1 predicts one more row per segment, from a first row with no preceding action)
becomes the only remaining cost.

Do NOT start the Phase 4 cleanup on this evidence. The circuit is not yet shown to be
better; it is shown to be more informative to the policy, which is a different claim.

---

# The entropy x offset 2x2 (Mila, commit `a9c665a`, seed 2, L40S)

Four jobs, one commit, one GPU type, run in parallel; 27.3-27.9 min each, all at
43,936 pRNN and 175,744 policy gradient steps.

🔴 **Read the TAIL MEAN, never the final point.** `sRSA_onPolicy` is logged 26 times per
run and its own adjacent-sample spread is 0.068-0.114; `pRNN loss` is logged 10,000 times.
A single endpoint is inside the noise of both, and reading endpoints produced three wrong
conclusions in the first version of this document (recorded under "Corrections" below).
`check/wandb_compare` already bands its comparisons for this reason. Both tables follow.

FINAL LOGGED POINT (what wandb's summary shows - kept because it is what the run object
reports, NOT because it is the estimator to use):

| metric | off0 e0.001 | off0 e0.01 | off1 e0.001 | off1 e0.01 |
|---|---|---|---|---|
| `pRNN loss` | 0.00425 | 0.00468 | 0.00479 | 0.00387 |
| `sRSA_onPolicy` | 0.67249 | 0.54028 | 0.67051 | 0.60696 |
| `mean SI_onPolicy` | 0.97888 | 1.02642 | 0.83239 | 0.96492 |
| `SI_mean_active_only` | 0.99077 | 1.05382 | 0.85637 | 0.97664 |
| `SI_units_zeroed` | 6 | 13 | 14 | 6 |
| `SWdist_onPolicy` | 0.09164 | 0.07019 | 0.03957 | 0.05442 |
| `policy_entropy` MIN | 1.5305 | 1.8187 | 0.7386 | 1.8270 |
| `loc_entropy` MIN | 4.7155 | 6.1124 | 3.2205 | 5.9425 |
| `MI_policy` MAX | 0.2111 | 0.0406 | 0.5306 | 0.0520 |

TAIL MEAN - **this is the table to read.** sRSA/SI/SWdist over the last 10 analysis points,
`pRNN loss` over the last 500 gradient steps, `MI_policy` over the last 200 updates:

| metric | off0 e0.001 | off0 e0.01 | off1 e0.001 | off1 e0.01 |
|---|---|---|---|---|
| `sRSA_onPolicy` | 0.7201 | 0.6974 | 0.7216 | 0.6780 |
| &nbsp;&nbsp;its own adjacent-sample band | 0.080 | 0.068 | 0.114 | 0.080 |
| `pRNN loss` | 0.00462 | 0.00431 | 0.00553 | 0.00429 |
| `mean SI_onPolicy` | 0.9899 | 1.0697 | 0.8717 | 0.9672 |
| `SI_mean_active_only` | 1.0043 | 1.0881 | 0.9065 | 0.9848 |
| `SI_units_zeroed` | 7.3 | 8.4 | 20.7 | 9.0 |
| `SWdist_onPolicy` | 0.0897 | 0.0674 | 0.0620 | 0.0708 |
| `MI_policy` | 0.0866 | 0.0121 | **0.2075** | 0.0144 |
| **% updates with `policy_entropy` < 1.0** | 0.0% | 0.0% | **1.9%** | 0.0% |

## Confirmed: the loss cost was exploration, not the circuit

On the last 500 gradient steps, **offset 1 at entropy 0.01 TIES offset 0 at the
same entropy** - 0.00429 against 0.00431. At 0.001 offset 1 is genuinely worse:
0.00553 against 0.00462, a real ~20% gap that holds over any tail window from
500 to 2000 steps. Nothing about the circuit changed between those two cells;
only the entropy bonus did, and it closed the gap completely.

The collapse goes with it: `policy_entropy` MIN rises 0.7386 -> 1.8270 and
`loc_entropy` MIN 3.2205 -> 5.9425, both from worst-of-four to best-of-four.
The prediction made before the runs - "offset 1 at 0.01 holds coverage and closes
most of the loss gap" - is confirmed, and the original intuition that pairing
obs[t] with the action that produced it should HELP prediction is vindicated
once the policy is not starving the world model of data.

`SWdist_onPolicy` does NOT hold up as a consistent win. On last-10 means it
favours offset 1 in 2 of 4 comparisons - cluster e0.001 (0.0620 vs 0.0897) and
local seed 2 (0.0843 vs 0.1033) - and offset 0 in the other two: local seed 3
(0.1724 vs 0.1175) and cluster e0.01 (0.0708 vs 0.0674). The earlier "4/4" was
counted on final points.

## But 0.01 over-corrects

`MI_policy` MAX falls to 0.0406 and 0.0520, against 0.2111 and 0.5306 at 0.001.
The policy is barely state-dependent at all - `policy_entropy` MIN of 1.82/1.83
against a 2.000 ceiling is very nearly a uniform random walker. That matches
what `slurm/train_fast.sh` already records: at 0.01 the entropy bonus is 0.0198
against an advantage scale of 0.0369, over half the learning signal.

sRSA, however, does NOT fall when entropy is raised - that reading was an
endpoint artifact. On last-10 means the four cells are 0.7201, 0.6974, 0.7216
and 0.6780, spanning 0.044 against per-run bands of 0.068-0.114. **No cell is
distinguishable from any other**, and the apparent off0 collapse to 0.540 was
one noisy final sample from a run whose last-10 mean is 0.6974. sRSA has not
separated these conditions at any point in this experiment.

⚠️ The cluster sRSA values (0.54-0.67) sit below the local ones (0.66-0.85) at
the same configuration. Treat sRSA comparisons ACROSS machines as meaningless
here; only within-batch comparisons count.

## Corrections to the first version of this document

All three came from reading a single final logged point as if it were the run's value. The
underlying wandb data never changed; only the estimator did. Recomputed 2026-08-29 from
`wandb.Api()` history, 1,373 update points and 26 analysis points per run.

| claim as first written | what the tail mean says |
|---|---|
| "offset 1 at 0.01 has the LOWEST loss of all four arms (0.00387)" | **Tie.** 0.00429 vs off0's 0.00431 over the last 500 gradient steps. The direction survives - at 0.001 offset 1 really is ~20% worse and entropy closes it - but not the ranking. |
| "sRSA falls for BOTH arms when entropy is raised, off0 0.672 -> 0.540" | **No.** Last-10 means are 0.7201 / 0.6974 / 0.7216 / 0.6780, all inside bands of 0.068-0.114. The 0.540 was one noisy sample. |
| "`SWdist` favours offset 1 in 4/4 comparisons" | **2/4** on last-10 means. |

The reproducible offset-1 signature is therefore NOT any representation metric. It is the
transient exploration collapse at `entropy_coef=0.001` (1.9-21% duty cycle, versus 0.0% for
every offset-0 run) and the higher `MI_policy` that comes with it (0.2075 vs 0.0866 tail
mean, 0.53 vs 0.21 max).

## 🔴 Why sRSA has no resolving power here: the in-training probe is UNSEEDED

`evaluate_spatial_representation` takes a `probe_seed` that seeds torch, numpy AND
`env.env.reset(seed=...)`, and its own comment says why: without it "the eval [is]
unreproducible: identical action sequences, different trajectories ... Seeding the env too
makes this a FIXED probe, i.e. checkpoints become comparable to each other rather than each
carrying its own rollout noise."

**Nothing in `curious_george/` passes it.** `grep -rn probe_seed --include=*.py
curious_george/` matches only the definition and its own body; `training/loop.py`'s call
site passes `sleepstd`, `wandb_nameext`, `n_trajs`, `traj_timesteps`, `trainDecoder` and
`legacy_timesteps`, and `EvalCfg` has no field for it. So every point on every
`sRSA_onPolicy` curve in this document is a fresh unseeded rollout: new start position, new
sampled actions, and new injected pRNN noise (`trainNoiseMeanStd` is deliberately KEPT
through `eval_mode`, per the same function's comment).

That is the mechanism behind the 0.068-0.114 adjacent-sample band, and it is a defect, not
a property of the representation. Until it is fixed, no sRSA difference in this project
smaller than ~0.11 means anything.

## Next

Three things, in this order:

1. **Fix the probe.** Give `EvalCfg` a `probe_seed` and pass it, so the curve measures the
   representation instead of the rollout. Everything below is under-powered until this lands.
2. **Lower the entropy without losing the metrics.** The axis with real separation is
   `MI_policy` (0.0121 to 0.2075, a 17x range) and the collapse duty cycle (0.0% to 21%) -
   NOT sRSA. Sweep `entropy_coef` between 0.001 and 0.01 at offset 1 and find the largest
   value that keeps the duty cycle at 0% while holding MI near its 0.001 level.
3. **Look at the hidden states directly.** If the place code is what changed, a 2-D Isomap
   of the theta-mean hidden activity should recover the L-room's shape; that is a claim
   about the representation that does not route through sRSA at all.

`train_policy.entropy_coef_final` ramps the coefficient LINEARLY in environment steps and
`train_fast.sh` argues for it ("Collapse is a LATE phenomenon ... so a RISING coefficient
puts the resistance where the drift is"). It remains a candidate for step 2, but it is now
a WEAKER one than it looked: a flat 0.01 already drives the duty cycle to 0%, so a ramp is
only worth it if an intermediate flat value cannot hold both MI and coverage.
