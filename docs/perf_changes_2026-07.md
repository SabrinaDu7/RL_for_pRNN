# Perf overhaul, July 2026 — changes, rationale, and flags

Branch `sdu/rl-rollout-arch`, commits `226e1a9..66e9526` (6 commits, local only,
not pushed). Companion docs: `perf_baseline.md` (measurements + methodology),
`perf_log.md` (per-commit running log). This file is the review guide: what
changed, why, and what to watch out for.

## Purpose

Training was ~9.3 s per 2048-frame update (~218 FPS) and each analysis event
took ~10 minutes. Profiling (not guessing) showed the time was NOT model
compute — it was per-timestep CPU↔GPU syncs, Python-loop data conversion,
every-update plotting, and duplicated evaluation rollouts. Goal: make train
and eval fast while gating every change on a **metric comparison harness**
(losses, gradients, reward stats) instead of bitwise equality — per Sabrina's
explicit decision that bitwise goldens may be retired when justified.
(In practice, every change so far turned out bitwise-safe: goldens are still
green and none were retired.)

## Bottom line

| config | before | after |
|---|---|---|
| stock (CUDA default), per update | 9.3 s (218 FPS) | 7.3 s (274 FPS) |
| same run on CPU (`CG_DEVICE=cpu`) | — | 4.9 s (415 FPS) |
| opt-in `exp.num_envs=8` (CPU) | unsupported-ish | 3.2 s (642 FPS) |
| plotSampleTrajectory + model GPU↔CPU swap | **every update** | every 200 (config) |
| spatial analysis event | 293 s ×2 passes | 10.2 s ×1 pass |

## The changes, commit by commit

### `226e1a9` — Phase 0: measure first
- `curious_george/utils/timing.py`: stage timers (no-op unless enabled;
  `CG_TIMING=1` or the benchmark enables them; `sync_cuda` for honest GPU
  attribution). Instrumented: collector stages, PPO update, wm train, plotting,
  analysis.
- `tests/perf/benchmark.py` + `compare_metrics.py`: run N updates fixed-seed,
  record per-stage timings AND learning metrics (PPO losses, grad norm,
  entropy, curiosity stats, per-segment pRNN loss), dump JSON, diff two runs
  with tolerances. **This pair is the gate for all perf work.**
- Key finding: **CPU beats CUDA ~2× at B=1** (413 vs 218 FPS) — hidden_size
  500 / batch 1 can't amortize kernel-launch + sync overhead. Every stage was
  faster on CPU, including gradient steps.

### `221f1cf` — Phase 1: kill the per-step syncs (exact-math)
- `plotSampleTrajectory` was running **every update** (`log_interval=1`),
  each time moving pRNN+ACmodel GPU→CPU→GPU. Now behind new
  `logging.plot_interval` (default 200).
- collector: the four full-rollout `.tolist()` (values/advantages/curious/int,
  2048 elements each) became single `.cpu().numpy()` arrays — consumers only
  compute mean/std. Per-step `rewards[t,b].item()` and per-step subroom
  `.item()` removed (subrooms batched post-rollout in the same (t,b) order).
- GAE (`advantage.py`): Python reverse loop with per-element device indexing →
  float32 numpy scan. 100 ms → 0.7 ms, **bitwise identical** (op order
  preserved; the one reassociated multiply is by a 0/1 mask, hence exact).
- `updater.py`/`losses.py`: grad norm now taken from `clip_grad_norm_`'s
  return (same quantity the per-parameter `.item()` sum computed); LossTerms
  hold detached 0-dim tensors, one sync per metric per update (~160 → 5).
- `adapter.train_on_episode`: per-frame `.cpu()/.item()` (~2048/update) → one
  transfer per segment.
- `CG_DEVICE=cpu|cuda` env var now overrides device selection.

### `87b6335` — Phase 3: skim evaluation (Sabrina's decisions applied)
- `exp.offpolicy_prnn_eval: False` (was True) — halves analysis events.
- `exp.eval_timesteps: 2000` (was hardcoded 15000 inside prnn) and
  `exp.eval_decoder: False` (was an unconditional 5000-batch decoder fit).
- Behavior Plotly figures only at `plot_interval`; the MI scalar stays at
  every `analysis_interval`.

### `6247a5a` — Phase 2: vectorized SpeedHD formatting
- `adapter.py` gains `flat_obs_rows` / `encode_speed_hd_rows|seq`: one numpy
  stack + two vectorized one-hots replacing `env2pred`'s per-item Python
  loops. **Bitwise-equal to prnn's originals** — `tests/test_batch_format.py`
  asserts `torch.equal`, including the `act[0]<0` no-action flag and int64
  dtype.
- Wired into: `next_sr` (B=1 per-step hot path), `BatchedSRTrackerShim.step`
  (was one `env2pred` call per env per step), `episode_prediction_rows`
  (reward pass), `train_on_episode` (tensor-native; images never leave the
  device). Non-SpeedHD encodings fall back to `env2pred` (`fast_speedhd`
  flag checks `encodeAction.__name__`).
- Deliberately did NOT import `../pRNN-new`'s ShellVectorized /
  predictiveNetVectorized: that's a divergent fork (different
  Architectures.py, different remote), and the RL repo's adapter/tracker seam
  already owns batched stepping (`BatchedSRTracker`).

### `482d86b` — spatial eval: single-rollout rewrite
- **Why it was slow** (cProfile evidence): ~85% of the eval was **two serial
  agent rollouts** — one inside prnn's `calculateSpatialRepresentation`, and a
  duplicate inside `compute_sleep_wake_dist`, which existed only because prnn
  computes SWdist internally but returns neither it nor the hidden states.
  Each rollout ≈ 60% MiniGrid RGB render + 40% per-step forwards.
- **Fix**: `evaluate_spatial_representation` collects ONE wake rollout +
  ONE `predict`, then derives SI (pynapple tuning curves, same bins and
  active-time threshold as prnn), sRSA (`RGA.calculateRSA_space`), and SWdist
  (spontaneous rollout + `RGA.calculateSleepWakeDist`) from that shared
  activity. The legacy double-rollout path survives behind `trainDecoder=True`
  (position decoding is the one capability the rewrite drops).
- Verified against the old implementation on the same net: sRSA 0.0906 ==
  0.0906, SWdist 0.0092 vs 0.0096 (statistically equivalent — the two SWdists
  never came from the same rollout even historically).

## Things to flag (review these)

1. **sRSA is rollout-length sensitive.** At `eval_timesteps=2000` vs the
   historical 15000, sRSA on a near-untrained net read 0.091 vs 0.044.
   Within-run trends stay valid (fixed length), but absolute values are NOT
   comparable to old wandb dashboards. Set `exp.eval_timesteps=15000` if you
   need cross-run continuity.
2. **wandb key continuity for spatial metrics.** prnn's internal logging
   (`mean SI`, `sRSA`, `SWdist` + nameext) no longer fires on the fast path;
   `training/logging.log_spatial` now logs the same key names from the
   returned dict. If a dashboard reads other prnn-internal keys (figures,
   `EVs`), those are gone on the fast path.
3. **sRSA/SWdist now share one wake sample.** Estimator-equivalent but not
   the same random draws as the historical two-rollout version; expect
   within-noise differences at equal seeds.
4. **`reshaped_return_per_episode`** now accumulates the env's raw float
   reward instead of its float32-rounded copy — last-bit differences vs
   historical logs (more accurate, not less).
5. **Loss scalars** (entropy/policy/value) are computed in float32 on-device
   rather than float64 after `.item()` — ~1e-8 relative shift.
6. **`hardware.use_gpu` was and remains dead** — nothing ever read it.
   Device control is `CG_DEVICE` (import-time binding prevents a config knob
   without a bigger refactor). Given CPU>GPU at B=1, consider
   `CG_DEVICE=cpu` for standard runs.
7. **B>1 is faster but not yet the default.** 642 FPS at B=8, but per-stream
   rollout length drops to 2048/B and exploration statistics shift. Flip
   `exp.num_envs` only after a multi-seed learning-curve comparison
   (planned, not done). AsyncVectorEnv (parallelizing the MiniGrid render,
   now the dominant cost) is only worth building after that decision.
8. **Pre-existing breakage, named not fixed:** `tests/test_figure3_sRSA.py`
   imports a module that doesn't exist (`scripts.figure3_sRSA`); 16 test
   failures in `tests/test_wandb_data.py` and friends predate this work
   (verified via `git stash` at `8598f99`). Suite went 194→197 passed only
   because of the 3 new formatting tests.
9. **Goldens were kept, not retired.** Everything landed so far is
   bitwise-safe, so the golden fixtures still pass. The approved
   retire-and-regenerate policy applies to FUTURE changes that alter RNG
   order (e.g., making B>1 default).
10. **prnn stays pinned** (uv git dep `2663d9f`). Real fixes that belong
    upstream (return SWdist/h from `calculateSpatialRepresentation`;
    theta-safe `predict_single`) are deferred until a planned repin.

## Not done yet (in priority order)

- B ∈ {1,4,8} multi-seed learning-curve comparison → `num_envs` default
  decision → AsyncVectorEnv (only if adopted).
- Phase 4: torch.compile audit (blockers already catalogued in
  `perf_baseline.md`; needs prnn changes).
- The paused thcycRNN_5win RL integration (`thcyc_rl_integration.md` has the
  full analysis and the four open design questions awaiting Sabrina).
