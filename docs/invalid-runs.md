# Invalidating lines through the run series

**What this is.** Each entry is a commit after which the code computes something
materially different from before it. A run, checkpoint or figure produced on the wrong
side of one of these lines does not measure what its document says it measures.

**Why it has to exist separately.** These were recorded only as prose plus SLURM job
numbers inside two long session documents. To place an existing artifact on a timeline
you had to have written its job number down at launch, and the artifact itself carried
nothing. Every artifact made after `provenance.json` lands (refactor plan, Phase 1)
records its own commits, and then each line below becomes a property of the artifact
rather than something a reader has to reconstruct.

**Rate, for calibration:** four lines in five days, 2026-08-21 to 2026-08-25.

---

## `d275149` — `recurrence` removal · 2026-08-23

**What changed.** Removed a dead `recurrence` parameter and, with it, a code path that
silently dropped one transition on odd epochs. Policy minibatch composition changed, so
the update statistics and the weights changed with them.

**What it invalidates.** Any comparison of policy-update quantities across this line.
The world-model rollout is NOT affected: measured with `tests/golden/capture_golden.py`,
round 0's `curious_rewards`, `advantages`, `actions`, `log_probs`, `SRs` and `locs` are
bitwise unchanged; only the update and what follows it moved (17 leaves).

**How to tell which side an artifact is on.** Before: `37aaa1b` and earlier. After:
`d275149` and later.

**Fixtures.** `tests/golden/golden_v1.pt` is valid up to `37aaa1b`; `golden_v2.pt` is
the baseline from `d275149` on. `../experiment-curiousgeorge/tests/golden_omt/golden_omt_v1.pt` was re-captured at
the time; the training-path fixture was missed and stayed stale for three days because
no test ran it. `tests/golden/test_golden.py` now does.

---

## `0d60df7` — stranded CUDA-graph buffers · 2026-08-23

**What changed.** `on_device` now restores `param.grad` and registered **buffers**, not
just `param.data`. Before it, a captured graph read `thRNN_5win`'s phase masks
(`inMask_f` / `actMask_f` / `outMask_f`) from freed memory after any spatial eval moved
the model to CPU and back, so the graph scaled its inputs by whatever occupied that
memory.

**What it invalidates.** *"Every arm launched before it was found is invalid"* — any
run with `predNet.cuda_graph=True` that also ran a spatial eval. The failure is silent
and the loss curve looks healthy, because the forward reads parameters (correct) and
only the masked inputs were wrong: replayed loss 0.038778 → 0.006764 on identical data,
gradient norm 0.0603 → 0.0156. A graphed 20.48M-step arm held sRSA at 0.0238 against its
eager control's 0.5158.

**How to tell.** Before: `2c83318` and earlier with graphs on. After: `0d60df7` and
later.

---

## `4cfd8cb` — captured floats · 2026-08-24

**What changed.** Recorded (did not yet fix) that a captured CUDA graph freezes every
scalar hyperparameter passed as a Python float at its capture-time value.
`GraphPolicyTrainer._region` and `_GraphWMTrainer._capture` bake them in; the graphed
path never re-reads them.

**What it invalidates.** Any schedule under `rl.cuda_graph=True`. Confirmed dead:
`rl.entropy_coef_final`. Same exposure: `clip_eps`, `value_loss_coef`, and both
optimizers' `lr`. Specific runs: `10482137` ("ramp 0.001→0.01") ran pinned at 0.001;
`10483973` ("ramp 0→0.001") ran pinned at 0, a duplicate of the `ent=0` arm. `10483974`
(constant 0.0003) is valid, because a constant captures correctly.

**Status: OPEN.** The fix — a 0-dim device tensor mutated with `.fill_()` — is proven to
work under capture but has not landed. Until it does, every scalar schedule is dead
under graphing.

---

## `UpdateLogs` epoch averaging — 2026-08-25

**What changed.** The eager path re-created its log accumulators inside the PPO epoch
loop (inherited verbatim from `torch_ac/algos/ppo.py:32-35`), so it reported the mean
over the **last** epoch only — one quarter of the gradient steps at `ppo_epochs=4`. The
graphed path averaged **all** epochs. Eager now matches graphed.

**Measured, and it is reporting-only.** `tests/golden/capture_golden.py` moved
**exactly six leaves** — `policy_loss`, `value_loss` and `grad_norm` in both rounds —
and left every weight and every rollout tensor bit-identical:

```
rounds[0].grad_norm    2.1485 -> 3.6079
rounds[0].value_loss   0.2125 -> 0.2373
rounds[0].policy_loss -0.4792 -> -0.4626
```

All three in the direction the mechanism predicts, so the old convention understated
`grad_norm` by 41% on this fixture.

**What it invalidates.** Policy diagnostics from every eager arm, including the
2026-07 reference `pRNN_curious_26-07-08-16-04-37`: `policy_entropy`, `value_loss`,
`grad_norm`, `policy_loss`. The direction of the bias is systematic, not noise — eager
reports lower entropy, lower `value_loss`, lower `grad_norm` and larger-magnitude
`policy_loss` than graphed for the identical update.

**What it does NOT invalidate.** `pRNN loss`, `sRSA`, `SWdist` and `SI` do not pass
through `UpdateLogs`. The entire world-model line is untouched.

**Fixture.** `golden_v3.pt`. **Gate.** `tests/test_update_logs_semantics.py` recomputes
the statistic from every `LossTerms` the update produced, so it holds for any config and
any device; all 9 of its tests go red on the old placement.

**Already contaminated by it:** `speed-30min` §9's eager-vs-graphed entropy comparison
("g8 EAGER 0.5927 at 60M against g8 graphed 1.3357") mixes an estimator difference with
a real one, in an unmeasured proportion.
