# Performance log

Running record of each landed perf change, its gate result, and FPS.
Baselines and harness: docs/perf_baseline.md, tests/perf/.
All FPS numbers: 3 updates x 2048 frames, seed 2, stock config, timers on
(CUDA numbers include sync overhead at stage boundaries).

| commit | change | CPU FPS | CUDA FPS | gate |
|---|---|---|---|---|
| 8598f99 (HEAD~2) | pre-existing | 412.9 | 217.7 | baseline captured |
| 226e1a9 | phase 0: timers + harness (no semantics) | = | = | n/a |
| 3cbe602 | phase 1: sync removal, GAE numpy, plot gating | 427.8 | 273.6 | compare_metrics PASS (rollout metrics bitwise; loss scalars ~1e-8); pytest 194P/16F unchanged (all 16 pre-existing) |

## Phase 1 notes (3cbe602)

- GAE 100 ms -> 0.7 ms per update (bitwise-equal float32 numpy scan).
- The biggest *real-run* win isn't in the benchmark numbers: plotSampleTrajectory
  + a GPU<->CPU model round-trip used to run EVERY update (log_interval=1);
  now behind `logging.plot_interval` (default 200).
- `state.ep_reshaped` now accumulates the env's raw float reward instead of
  its float32-rounded copy (`rewards[t,b].item()`): the logged
  reshaped_return_per_episode can differ from historical values in the last
  float32 bit. More accurate, noted for dashboard continuity.
- Loss-scalar logging (entropy/value/policy_loss/value_loss) now divides in
  float32 on-device instead of float64 post-.item(): ~1e-8 relative shift.
- `CG_DEVICE=cpu` makes runs ~2x faster at B=1 (until B>1 lands). The
  `hardware.use_gpu` yaml knob was and remains dead (documented flaw).

## Remaining planned work

- Phase 2: batch_env2pred (vectorized obs/action conversion in adapter),
  B>1 benchmark {1,4,8} + learning-curve check, AsyncVectorEnv only if env
  stepping still dominates.
- Phase 3: eval skimming per Sabrina's decisions (scalar-only at
  analysis_interval, drop off-policy spatial pass, shorter eval rollouts).
- Phase 4: torch.compile audit (needs prnn repo changes; last).
