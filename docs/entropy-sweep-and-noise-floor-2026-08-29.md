2026-08-29 · branch `sdu/predict-next-obs` · commit `7d335a1` · Mila L40S

# The entropy knee at action_offset=1, and what every metric's noise floor is

Two results. The second is the more useful one: five seeds at ONE configuration
give the seed-to-seed spread of every metric this project logs, which is the
band any future eval - multi-room included - has to beat to say anything.

## 1. The knee is `entropy_coef = 0.003`

`action_offset=1`, seed 2, `main_train.py parity` shape (89,980,928 env steps;
43,936 world-model and 175,744 policy gradient steps). Tail means: sRSA over the
last 10 analysis points, loss over the last 500 gradient steps, `MI_policy` over
the last 200 updates.

| `entropy_coef` | pRNN loss | sRSA | `MI_policy` | **% updates < 1.0 bits** | median `loc_entropy` |
|---|---|---|---|---|---|
| 0.001 | 0.00553 | 0.7216 | **0.2075** | 1.9% | 6.780 |
| 0.002 | 0.00507 | 0.7369 | 0.1239 | 3.7% | 6.814 |
| **0.003** | **0.00458** | 0.7178 | 0.0530 | **0.0%** | 6.989 |
| 0.005 | 0.00439 | 0.7326 | 0.0290 | 0.0% | 7.110 |
| 0.01 | 0.00429 | 0.6780 | 0.0144 | 0.0% | 7.217 |
| 0.001 -> 0.01 ramp | 0.00590 | 0.7457 | 0.3325 | **13.3%** | 6.753 |
| *offset 0 @ 0.001* | *0.00462* | *0.7201* | *0.0866* | *0.0%* | *6.989* |

0.003 is the lowest flat coefficient with a 0.0% collapse duty cycle, at a loss
matching the offset-0 baseline and 3.7x the mutual information of a flat 0.01.

🔴 **The ramp is the worst arm ever run here** - 13.3% duty cycle against flat
0.001's 1.9%, worst loss of all, and it starts collapsing at 35.2% of training
rather than 80.2%. Collapse IS late under a flat coefficient, but it is not
driven by the CONTEMPORANEOUS coefficient: at 35% the ramp already sits near
0.0042, above the flat 0.003 that never collapses. What matters is whether the
policy was allowed to sharpen EARLY, which a ramp specifically permits.
`training/schedule.py::EntropySchedule` carries this correction.

## 2. The noise floor: five seeds, one configuration

`mila-off1-e0.003-s{2,3,4,5,6}`. Everything identical but `run.seed`.

| metric | mean | sd | range | **coefficient of variation** |
|---|---|---|---|---|
| `pRNN loss` | 0.00457 | 0.00008 | 0.00019 | **1.7%** |
| `loc_entropy` (median) | 6.99387 | 0.01677 | 0.04410 | **0.24%** |
| `mean SI_onPolicy` | 0.95581 | 0.03235 | 0.07904 | 3.4% |
| `sRSA_onPolicy` | 0.72834 | 0.03283 | 0.08270 | 4.5% |
| `MI_policy` | 0.04997 | 0.00282 | 0.00701 | 5.6% |
| `SWdist_onPolicy` | 0.07871 | 0.02227 | 0.05472 | **28.3%** |
| % updates < 1.0 bits | 0.0 | 0.0 | 0.0 | - |

**How many seeds a difference needs.** For a two-arm comparison the difference's
sd is `sd * sqrt(2)`, so detecting an effect `d` at n per arm needs roughly
`d > 2.8 * sd * sqrt(2/n)`:

| metric | n=1 resolves | n=3 | n=5 | n=8 |
|---|---|---|---|---|
| `pRNN loss` | 0.0003 | 0.0002 | 0.0001 | 0.0001 |
| `sRSA_onPolicy` | 0.130 | 0.075 | 0.058 | 0.046 |
| `SWdist_onPolicy` | 0.088 | 0.051 | 0.039 | 0.031 |

Which is why every sRSA conclusion in `docs/action-offset-ab-2026-08-29.md` at
n=1 was wrong: nothing below 0.13 was ever resolvable there. Prediction loss, by
contrast, resolves a 0.0003 difference at n=1 - every loss gap discussed in this
project is 3-10 sd and real.

⚠️ **The collapse result IS solid**: 0.0% duty cycle in 5 of 5 seeds, against
1.9%/3.7% at lower coefficients and 13.3% under the ramp.

## 3. What this means for the move to multi-room

`evaluation/spatial.py:341` defines the multi-room headline as

    remapping_index = mean(per-room sRSA) - pooled sRSA

a DIFFERENCE OF TWO sRSAs. It therefore inherits sRSA's 0.033 seed sd twice:
its own noise floor is ~0.046 at n=1 before any per-room subsampling noise is
added. Any remapping claim smaller than that is unresolvable, and the existing
single-room evidence says differences of that size are exactly what to expect.

Two things worth fixing before spending cluster time on multi-room:

1. **The spatial probe is unseeded.** `evaluate_spatial_representation` accepts
   `probe_seed` and nothing in `curious_george/` passes it. On ONE frozen
   checkpoint, 6 repeats: sRSA range 0.0596, SWdist range 0.1174. With
   `probe_seed`: sd 0.0000. That estimator noise is stacked on top of the seed
   noise above, and it is free to remove.
2. **Coverage is policy-dependent**, so a metric measured on-policy confounds
   "the map changed" with "the agent stopped exploring". Starting the eval from
   every (cell, head direction) removes it; the rollout is only 13% of the
   eval's 5.81 s, so exhaustive coverage over the L-room's 172 cells x 4
   directions costs about 4.6 s more per analysis event.

Reproduce any arm with `sbatch slurm/action_offset_ab.sh 1 0.003 <seed>`.
