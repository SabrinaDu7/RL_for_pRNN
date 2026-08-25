# Performance baseline (pre-optimization)

Date: 2026-07-07. Branch: sdu/rl-rollout-arch (dirty: timing instrumentation
added, no semantics changed). Config: stock `Configs/main.yaml` (thRNN_5win,
lroom, PPO, curious, B=1, 2048 frames/update, seqdur 256, seed 2), wandb off,
periodic logging/analysis disabled unless stated.

Hardware: RTX 4060 (8 GB), 8-core CPU.

Harness: `tests/perf/benchmark.py` (stage timers in
`curious_george/utils/timing.py`, cuda-synced boundaries). Compare runs with
`tests/perf/compare_metrics.py`. Raw JSON in `tests/perf/results/`.

## CUDA baseline — 3 updates, cProfile on

**FPS: 217.7** (update wall times: 9.76, 9.23, 9.24 s; ~9.3 s steady)

| stage | s/update | ms/call | notes |
|---|---|---|---|
| collect/env_step | 2.37 | 1.16 | dominated by `Shell.step` → MiniGrid step + RGB partial render (cProfile: 1.05 ms/step inside gymnasium) |
| collect/policy_fwd | 2.01 | 0.98 | preprocess + ACModelSR fwd + `action.cpu()` + `probs.cpu()` per step |
| collect/sr_step | 1.62 | 0.79 | `next_sr` → env2pred (CPU) + `.to(cuda)` + predict_single per step |
| update/wm_train | 1.57 | 1573/call | 8 pRNN trainSteps (BPTT over 256) + per-frame dict rebuild in `train_on_episode` |
| collect/curious_rewards | 0.56 | 555/call | 8 episode `predict()` passes |
| collect/gae | 0.26 | 260/call | pure-Python reverse scan, per-element GPU indexing |
| update/policy | 0.19 | 193/call | 32 minibatch steps |
| collect/log_prep | ~0.001 | | (tolist cost shows up under wandb path, not here) |

Sum of stages ≈ 8.6 s ≈ update wall time. Note `log/sample_trajectory`
(plotSampleTrajectory + model device swap, currently every update in real runs
because `log_interval=1`) is NOT included above — measured separately below.

cProfile confirms: `prnn` forward (`Architectures.py:146`) called 6192×/3
updates (once per env step for SR + episode passes), `thetaRNN.forward` 5.3 s
cumulative; `Shell.step` 6.4 s cumulative.

## CPU baseline — 3 updates (CUDA_VISIBLE_DEVICES="")

**FPS: 412.9** (updates: 4.99, 4.91, 4.98 s) — **1.9× faster than CUDA at B=1.**

| stage | s/update | ms/call |
|---|---|---|
| collect/env_step | 1.47 | 0.72 |
| collect/sr_step | 0.97 | 0.47 |
| update/wm_train | 0.92 | 919/call |
| collect/policy_fwd | 0.80 | 0.39 |
| collect/curious_rewards | 0.32 | 322/call |
| collect/gae | 0.10 | 100/call |
| update/policy | 0.10 | 97/call |

Every stage is faster on CPU, including the gradient steps (wm_train 0.92 vs
1.57 s, PPO 0.10 vs 0.19 s): at hidden_size=500 / B=1 the models are too small
to amortize CUDA launch+sync overhead. Caveat: the CUDA run's stage boundaries
call `torch.cuda.synchronize()` (~37k syncs over 3 updates), so uninstrumented
CUDA FPS is somewhat higher than 218 — but the uniform per-stage gap makes the
conclusion robust. **Action: make CPU the default device for this workload**
(and note `Configs/main.yaml hardware.use_gpu` — check whether
`curious_george/utils/common.py:7` even consults it; it auto-selects cuda).

After device flip, the top costs are: (1) MiniGrid RGB render in env.step,
(2) per-step `env2pred` conversion in the SR tracker, (3) pRNN BPTT training,
(4) per-step policy forward — matching the planned Phase 1/2 targets.

## Metric reference (seed 2, 3 updates, CUDA)

Stored in `tests/perf/results/baseline_cuda.json` → `metrics`:
policy/value loss, grad_norm, entropy, curious mean/std, values/advantages
stats, per-segment pRNN losses. This JSON is the comparison reference for all
Phase-1 (exact-math) changes via `compare_metrics.py` (rtol 1e-5 on CUDA-
deterministic metrics; note minor nondeterminism can come from cuDNN — if a
metric wobbles at equal code, rerun baseline twice to bound it).

## Test-suite baseline

`pytest tests/ --ignore=tests/perf --ignore=tests/test_figure3_sRSA.py`:
**194 passed, 16 failed, 7 deselected** (15.8 s).

Both failure groups are PRE-EXISTING (verified by `git stash` + rerun at clean
HEAD 8598f99):
- `tests/test_figure3_sRSA.py` — collection error: imports
  `scripts.figure3_sRSA`, which does not exist (scripts/ has `figure3.py`).
- 12 failures in `tests/test_wandb_data.py` (+4 elsewhere in the full run) —
  assertion failures in plotting/fetch utilities, unrelated to the training
  loop.

## Known config flaw (found during baseline)

`Configs/main.yaml hardware.use_gpu` is DEAD: nothing reads it.
`curious_george/utils/common.py:7` hard-selects cuda-if-available at import
time. Given the CPU>CUDA result above this matters; fixed by honoring a
`CG_DEVICE` env var in common.py (config can't be honored without a lazy
DEVICE refactor - import-time binding).
