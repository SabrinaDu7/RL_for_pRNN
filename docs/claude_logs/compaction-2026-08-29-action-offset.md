2026-08-29 · branch `sdu/predict-next-obs`

# Compaction: the circuit change works; the policy collapse it causes is the open problem

**The next task is one arm: `action_offset=1` with a RAMPED entropy coefficient**
(`train_policy.entropy_coef_final`, 0.001 → 0.01), at n = 3, because sRSA is the only
metric still unresolved and it has fooled this investigation twice at n = 1.

## State

```
RL_for_pRNN     sdu/predict-next-obs @ 4386aa1, pushed (a9c665a is what the cluster ran)
                gate: 473 passed, 1 deselected   (baseline was 463; +7 new tests in
                tests/test_action_offset.py, +3 auto-parametrised over the new
                curious_george/evaluation/circuit_diagnostics.py)
                tests/golden bitwise green - action_offset=0 is unchanged
prnn            PINNED, UNTOUCHED: LevensteinLab/pRNN sdu/rl-integration @ 163cb578
                No fork. No re-pin. The circuit needed neither.
untracked       docs/claude_logs/rl_tricks_2026-08-29.md (not mine)
```

Throughput, local RTX 4060, mila-parity loop only: **34.15** wm_grad_steps/s at offset 0
(33.94 before the change) and **35.66** at offset 1 — offset 1 is faster, because its
curiosity pass feeds L+1 observations where offset 0 duplicates one to reach L+2.

---

## What the change is

`arch_prnn.action_offset` — one integer.

```
0  row t = (obs[t], a[t])    action chosen AFTER obs[t];   policy acts on h[t-1]   <- was
1  row t = (obs[t], a[t-1])  action that PRODUCED obs[t];  policy acts on h[t]
```

**Verified, not asserted.** Fingerprinting every tensor reaching `pN.predict` under both
settings: **17 of 18 identical** — architecture, encoding, `predOffset`, upstream
`actOffset`, `inMask`, weights, trajectory, observation input, target. Only the action
input differs.

🔴 **Do not use the upstream `actOffset` / `thRNN_5win_prevAct` / `SpeedNextHD`.**
`actOffset` is a `ConstantPad1d` front-pad plus a tail-drop, so it loses `HD[0]` from row 0
AND discards each segment's last action. Building the rows in `PRNNAdapter.action_rows`
keeps both. `SpeedNextHD` exists only to pre-shift HD so `actOffset`'s whole-vector shift
cancels on the HD half — unnecessary once you build the rows yourself. This is why
`pRNNtypes.masked_nextstep` stays retired and `prnn_type` stays a fixed property.

## Two silent bugs found and fixed

Both invisible at offset 0, because `init_sr` returns zeros there and never reads the
observation. Both would have corrupted the new arm without saying so.

- `collect_rollout` reset the tracker BEFORE the environment, so `h[0]` was built from the
  FINISHED episode's last view, at a position already left. Measured:
  `SR[L] == bootstrap(previous episode's last obs)`.
- `init_sr` zeroed the whole action vector, dropping `HD[0]`, so the rollout disagreed with
  the training pass at row 0.

`tests/test_action_offset.py` pins both, plus the shift itself, B=2-batched == two serial
streams, and device == CPU-table at either offset — a combination nothing covered, and the
one the runs used.

## The result

**Mechanism: confirmed decisively.** On each arm's own checkpoint, 12 segments × 256 steps:

| | offset 0 | offset 1 |
|---|---|---|
| `fwd(a[t])` decoded from `h[t]`, held-out balanced accuracy (chance 0.500) | **1.000** | **0.496** |
| end-of-segment MSE ÷ segment median | **3.29x** | **1.14x** |

The pending-action bit is gone from the hidden state. That was the whole point.

**Outcome: the policy collapses.** Reproducible in every offset-1 run at
`entropy_coef=0.001` — 2 local seeds and 1 cluster seed — and in NO offset-0 run:

```
                     off0 s2  off0 s3 | off1 s2  off1 s3     (local, e=0.001)
policy_entropy MIN    1.4490   1.5135 |  0.4910   0.3912     (2.000 = uniform over 4)
loc_entropy    MIN    4.9425   4.8581 |  2.8448   2.7754     (~7 cells)
```

Since the metrics are `spatial_onpolicy`, that degrades both the MEASUREMENT and the world
model's training distribution. `corr(policy_entropy, loc_entropy)` is **+0.94 at lag 0**
under offset 1 and **~0.00 at every lag** under offset 0 — the circuit change created the
coupling.

### The entropy 2×2 settles the loss question

Mila, commit `a9c665a`, seed 2, four L40S jobs in parallel, 27.3–27.9 min each, all at
43,936 pRNN / 175,744 policy gradient steps:

| metric | off0 e0.001 | off0 e0.01 | off1 e0.001 | off1 e0.01 |
|---|---|---|---|---|
| `pRNN loss` | 0.00425 | 0.00468 | 0.00479 | **0.00387** |
| `sRSA_onPolicy` | 0.67249 | 0.54028 | 0.67051 | 0.60696 |
| `SWdist_onPolicy` | 0.09164 | 0.07019 | **0.03957** | 0.05442 |
| `policy_entropy` MIN | 1.5305 | 1.8187 | **0.7386** | **1.8270** |
| `loc_entropy` MIN | 4.7155 | 6.1124 | **3.2205** | **5.9425** |
| `MI_policy` MAX | 0.2111 | 0.0406 | **0.5306** | 0.0520 |

**offset 1 at 0.01 has the LOWEST loss of all four arms.** At 0.001 it had the worst.
Nothing about the circuit differs between those cells. The loss penalty was exploration,
not the circuit — confirming the pre-registered prediction and the original intuition that
pairing `obs[t]` with the action that produced it should HELP prediction.

Corroborating: prediction loss is IDENTICAL before the first collapse (ratio 0.998 seed 2,
1.050 seed 3) and only diverges as coverage falls.

## Why a fresher state collapses the policy

Measured at e=0.001: `advantages_std` 0.05535 vs 0.04595 (+20%), `values_std` 0.01746 vs
0.01545, `MI_policy` max 0.531 vs 0.211. The entropy bonus is a FIXED additive term, so it
covers 1.8% of offset 1's advantage scale against 2.2% of offset 0's — already weaker
against a sharper policy. At 0.01 it becomes ~22% and the collapse disappears.

Inferred, and consistent with all of the above: `h[t]` is trained so `W_out·h[t] ≈ obs[t]`,
so it encodes `position[t]` — an indexable place code for where the agent IS. `h[t-1]`
encodes where it WAS, and the intervening step folds in `obs[t]`, which the policy never
sees (`with_CV=False`). Offset 0's staler state was accidentally blurring place-specific
action selection. The collapse is evidence the change worked: a policy that cannot collapse
on a representation is one that cannot exploit it.

The on-policy loss RISES during collapse (6.63 → 6.99 → 7.58) and `loc_entropy` settles at
an orbit of ~7-9 cells, not one cell — the agent is tracking the moving error maximum, not
parking somewhere easy.

## ⚠️ What is NOT established

- **sRSA is unresolved and has misled twice.** At n = 1 (seed 2) offset 1 looked worse
  (0.664 vs 0.750); adding seed 3 (0.846) showed the means are equal and the SPREAD is the
  difference — 0.18 against offset 0's 0.001. Every 2×2 cell is n = 1 against a band of
  ~0.10. **Do not read any sRSA difference in this document as real.**
- **Cluster sRSA (0.54-0.67) sits below local (0.66-0.85) at the same configuration.**
  Never compare sRSA across machines here.
- `entropy_coef=0.01` OVER-corrects: `MI_policy` max falls to 0.04-0.05 and
  `policy_entropy` pins near the 2.000 ceiling — very nearly a uniform walker, exactly what
  `slurm/train_fast.sh` already warns this coefficient does. sRSA falls for BOTH arms.
- **`SWdist` favours offset 1 in 4 of 4 comparisons** (2 local seeds, 2 cluster entropies).
  This is the one outcome signal with a consistent direction, and it is still n = 1 per cell.
- **Phase 4 (the cleanup) was NOT started**, deliberately: a `Circuit` enum, deleting
  `pastSR`, and the two remaining vacuous asserts. Gated on the science.
- `h[-1]` is `actfun(noise)` in the serial rollout and in every `predict`/`trainStep`, but
  **zeros in `BatchedSRTracker`**. Matching them was implemented and REVERTED — see
  `[[batched-tracker-rng-equality]]`.

## Where things are

```
docs/action-offset-ab-2026-08-29.md    the full result, all three stages
outputs/figures/circuit-{current,desired}.png   the circuit, one timestep per column
curious_george/evaluation/prediction_figures.py   trace_circuit / plot_circuit
curious_george/evaluation/circuit_diagnostics.py  the three mechanism statistics
slurm/action_offset_ab.sh              sbatch [offset] [entropy] [seed] [branch]
wandb blake-richards/curious-george    offset{0,1}-parity{,-s3}_*   (local, e=0.001)
                                       mila-off{0,1}-e{0.001,0.01}-s2_*  (the 2x2)
```

`docs/prnn-io-alignment.md` is SUPERSEDED and still needs rewriting: its claim that "the
action cannot inform the prediction it is paired with", its claim that the reward shift
patches an upstream misalignment, its verdict that `masked_nextstep` is misnamed, and its
whole route-A/route-B framing. Route B as described there (`actOffset=1`) is not the route
taken.
