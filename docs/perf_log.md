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

## Phase B (part 1): spatial metrics computed + logged inside prnn

- prnn branch sdu/prnn-perf-optim (pushed), commit 3c022a7:
  PredictiveNet.calculateSpatialMetrics(h, agent_pos, env, ...) - SI/sRSA/
  SWdist from precomputed pooled activity, wandb-logged inside prnn with the
  historical 'mean SI'/'sRSA'/'SWdist' keys. Tested in pRNN repo
  (test/test_spatial_metrics.py, 2 passed; test_env.py is stale - minigrid
  drift, per Sabrina).
- RL repo pin moved to that BRANCH (pyproject [tool.uv.sources] prnn
  branch=sdu/prnn-perf-optim; uv lock tracks its head, currently 3c022a7).
- evaluation/spatial.py pooled path prefers the prnn method; the RL-side
  computation remains as a fallback for older pins and gets deleted when
  Phase B is accepted (start of Phase C).
- Gates: RL suite 101 passed / 0 failed (goldens green -> prnn change is
  additive); analysis event through the prnn method reproduces Phase A values
  exactly (sRSA 0.1094, SWdist 0.0362, seed 2, 1-update net).
- Remaining Phase B items: trainStep .item()/TrainingSaver pd.concat trims,
  torch.compile groundwork.

## Mismatch investigation: CLOSED (2026-07-12)

Sabrina's Mila run of the perf tree with old-cadence overrides
(plot_interval=1, offpolicy on, eval_timesteps=15000, eval_decoder=true)
reproduces the pre-optimization curves (92f37d2). Verdict: the (a)/(c)
divergences (advantage extremes, cur_reward_max) between the Jul-5 and Jul-8
runs were eval/plot RNG-cadence + machine/realization effects, NOT a semantic
change from the perf refactor. (b) mean SI was the documented eval_timesteps
change, superseded by the Phase A pooled eval.

## Phase B: CLOSED (part 1 only, by measurement)

The planned trainStep/.item() and TrainingSaver pd.concat trims were measured
and rejected: concat costs ~20s cumulative over a FULL 10k-update run
(~2ms/update); trainStep syncs are 8/update. Not worth churn in a pinned dep.
The remaining single-stream lever is the thetaRNN Python timestep loop
(wm_train 0.92s/update) -> optional torch.compile "Phase D", deferred.

## Phase C (started): num_envs learning-curve gate

- slurm/bsweep.sh: 9 CPU array jobs, B in {1,4,8} x seeds {2,3,4},
  1500 updates each (rl.steps=3072000), exp_name bsweep-B{B}-s{seed}.
  Requires pushing sdu/rl-rollout-arch (cluster clones from $HOME).
- scripts/analysis_bsweep.py: fetches the runs from wandb, bins curves over a
  common frame horizon, reports per-B mean+/-std across seeds and a
  within-2sd-of-B=1 overlap verdict; optional figure.
- Local pilot first: bsweep-pilot-B{1,8}-s2, 300 updates each, validates the
  pipeline + Phase B wandb keys at the update-200 analysis event.
- Decision rule: flip exp.num_envs default only if B>1 curves sit inside the
  B=1 across-seed band; then evaluate AsyncVectorEnv for the serial render.

## Phase C gate result: B>1 is learning-equivalent (2026-07-13)

9/9 bsweep runs completed (B in {1,4,8} x seeds {2,3,4}, 1500 updates).
scripts/analysis_bsweep.py verdict vs the B=1 across-seed band:
- cur_reward_mean 92% in-band (rel diff 0.6-1.0%); value_loss 83-100%
  (1.4-2.4%); advantages_max 100% (2.2%); advantages_min 83-92% (0.4-1.1%);
  loc_entropy 83-92%; cur_reward_max 75-83%.
- policy_loss "22% rel" is the near-zero-metric artifact (bands overlap);
  policy_entropy 67-75% in-band, curves interleave with no consistent
  direction. Figure: bsweep.png.
No systematic separation on any metric -> per the agreed decision rule,
flipping exp.num_envs default to 8 is justified, and AsyncVectorEnv is the
next lever (serial render is ~half the B=8 update).

## Phase C1: AsyncVectorEnv rollout collection (honest result: +6-9% locally)

- curious_george/envs/vector.py: AsyncShellPool - workers run the raw wrapped
  minigrid envs (factory.make_env minus the Shell), positions ride step infos
  (PosInfo), mission stripped for lean IPC (DropMission, re-attached on
  unstack). Collector pool branch: one parallel step per t; synchronized
  seqdur resets (env-signaled dones assert). Eval/analysis use a separate
  eval shell (own seed stream - eval no longer perturbs training envs, a
  deliberate semantic improvement over the sync list mode).
- Gates: transition equivalence EXACT (tests/test_async_envs.py); collector
  metrics bitwise-equal to sync B=8 (compare_metrics PASS, prnn_loss
  0.00e+00); suite 103 passed / 0 failed.
- Speed (CPU, 8 cores, B=8, 3 updates): sync 569 FPS -> async ~604-618 FPS.
  env_step 6.4 -> ~4.6 ms/step only: the per-step lockstep IPC round-trip
  (~0.5ms) + 8 workers contending with torch threads on 8 cores eat most of
  the render parallelism. shared_memory measured slightly WORSE than pickling
  (598); payload size is not the bottleneck, latency is.
- New bottleneck ranking at B=8 async: wm_train (BPTT, ~1.1s/update) >=
  env_step (~1.2s) > curious_rewards (~0.38s). Further env-side gains need
  either more cores (cluster nodes - async may fare better there) or
  decoupled (non-lockstep) collection; further overall gains point back at
  the thetaRNN loop (optional compile work).
- exp.async_envs=True default at B>1; false restores in-process list (same
  transitions, bitwise-equal metrics).

## Phase C1 verdict: async collection OFF by default (2026-07-13)

Cluster benchmark (async_bench_10110503, 16-cpu allocation on cn-h001):
async env_step 6.0 vs sync 4.9 ms/step at B=8 (and worse at B=4); combined
with the dev box's +6-9%, async never earns its complexity ->
exp.async_envs now defaults to False. The pool stays as tested opt-in
infrastructure (transitions exactly equal, metric gate PASSed on cluster).

Bigger finding from the same log: cluster updates were 4-51s (vs 3.2s
local), wm_train 5-12s/update - torch/BLAS thread oversubscription (no
OMP_NUM_THREADS cap, torch spawns per-physical-core threads inside a
16-cpu cgroup). All slurm scripts now export OMP/MKL_NUM_THREADS=
$SLURM_CPUS_PER_TASK. The bsweep runs likely paid this tax too (their
learning-curve CONCLUSIONS are unaffected - equivalence is about values,
not speed - but cluster wall-times should drop a lot).
