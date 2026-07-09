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

## Phase 3: evaluation skimming (per Sabrina's decisions 2026-07-07)

Spatial analysis event: **293 s -> 15.5 s (19x)** per pass, measured with
`benchmark.py --include-analysis` (1 update, CPU). Historical events also ran
a second (off-policy) pass -> ~10 min total, now ~15 s.

Changes (all config-revertable):
- `exp.offpolicy_prnn_eval: False` (was True) - drops the duplicate pass.
- `exp.eval_timesteps: 2000` (was hardcoded 15000 inside prnn) - bounds both
  the calculateSpatialRepresentation rollout and compute_sleep_wake_dist's
  second wake rollout.
- `exp.eval_decoder: False` (was True) - skips the 5000-batch position-decoder
  fit; only decode-error analyses need it.
- Behavior figures (3 Plotly builds) only at `logging.plot_interval`;
  MI_policy_eval scalar still every `analysis_interval`.

Stability check (1-update net, seed 2): SWdist stable (0.0094 vs 0.0096);
**sRSA is rollout-length sensitive** (0.0436 @ 15000 vs 0.0906 @ 2000 on a
near-untrained net). Within-run comparisons stay valid (fixed length), but
absolute sRSA is NOT comparable to historical dashboards - set
`exp.eval_timesteps=15000` if you need that.

Pre-existing flaw named, not fixed: evaluate_spatial_representation computes
SWdist twice (prnn logs one internally, then compute_sleep_wake_dist reruns a
wake rollout because prnn doesn't return it). Real fix needs a prnn-repo
change; see curious_george/evaluation/spatial.py docstring.

## Remaining planned work

- Phase 2: batch_env2pred (vectorized obs/action conversion in adapter),
  B>1 benchmark {1,4,8} + learning-curve check, AsyncVectorEnv only if env
  stepping still dominates.
- Phase 3: eval skimming per Sabrina's decisions (scalar-only at
  analysis_interval, drop off-policy spatial pass, shorter eval rollouts).
- Phase 4: torch.compile audit (needs prnn repo changes; last).

## Phase 2: vectorized formatting + B>1 sweep (6247a5a)

- `flat_obs_rows` / `encode_speed_hd_*` in adapter.py replace env2pred's
  per-item Python loops; bitwise-equal (tests/test_batch_format.py), golden
  tests stay green. Non-SpeedHD encodings fall back to env2pred.
- CPU FPS: B=1 415, B=4 556, **B=8 642 (1.55x)**. Remaining floor is the
  serial MiniGrid RGB render in env.step -> AsyncVectorEnv is the next lever,
  only worth it if B>1 becomes the default (learning-curve check still TODO
  before flipping exp.num_envs).

## Spatial eval: single-rollout rewrite (this commit; hash shifts with amends)

- Profile: ~85% of the eval was two serial agent rollouts (the second existed
  only because prnn doesn't return SWdist/h). Now one rollout feeds SI, sRSA
  and SWdist; legacy path behind trainDecoder=True.
- Verified: sRSA 0.0906 == old 0.0906, SWdist 0.0092 vs 0.0096;
  event 15.5s -> 10.2s. wandb keys 'mean SI'/'sRSA'/'SWdist' preserved via
  log_spatial (prnn's internal logging no longer fires on this path).

## Cumulative (stock config, CPU, B=1): ~9.3s/update CUDA-default baseline
## -> 4.9s/update on CPU -> plus every-update plotting removed and analysis
## events 293s -> 10.2s. B=8 opt-in: 3.2s/update (642 FPS).

## Phase A (new plan): multi-trajectory pooled spatial eval

Per Sabrina's redesign (2026-07-09): eval trajectories must match TRAINING
trajectory statistics. evaluate_spatial_representation now collects
`exp.eval_trajs` (8) trajectories of `predNet.seqdur` (256) steps, pools the
theta-mean hidden states + positions (per-traj onset transient dropped), and
computes SI/sRSA/SWdist once on the pooled data under one CPU device move.
Legacy prnn path kept behind `exp.eval_decoder=True` (uses `eval_timesteps`).

- Analysis event: 10.2s -> 8.6s (1-update net). Suite: 101 passed, 0 failed.
- Logging per Sabrina's rule: pRNN metrics are logged BY prnn; the RL repo
  logs only its own SWdist_direct. Until Phase B moves the pooled-metric
  computation+logging into prnn, pooled-path runs do NOT emit
  'mean SI'/'sRSA'/'SWdist' wandb keys (legacy path still does).
- Reference values shift by design: pooled wake states include per-trajectory
  init transients (training-like); e.g. SWdist 0.036 pooled vs 0.009
  single-rollout on the same 1-update net. Trends within a run stay valid.

## New phase order (Sabrina, 2026-07-09)

A. eval restructure (this section) -> B. prnn-repo changes (metrics computed+
logged inside prnn on precomputed activity; trainStep sync trims; compile
groundwork; git-pinned flow, no editable installs) -> C. B>1 + AsyncVectorEnv
last (only step that breaks bitwise; goldens regenerated then).
