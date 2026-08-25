# Refactor progress log

Plan: `curious_george/` refactor (behavior-preserving; legacy paths deleted;
reward-alignment fix behind a flag, default `legacy`). Baselines in
`docs/refactor_baseline.md`. Branch `sdu/refactor`.

## Status by phase

- **Phase 0** (done, `a6eb212`): baselines. pytest 169 passed / 18 failed
  (pre-existing) / 7 deselected; golden fixture `tests/golden/golden_v0.pt`
  (bitwise-deterministic 2-round training oracle); sRSA=0.4391 /
  SWdist=0.0550 on `$PRNN_CUR_CKPT` (seed 2, CPU).
- **Phase 1** (done, `1d27802`): deleted theta algos/models, RecACModel,
  legacy Agent, FakePlaceCells, CANN wiring, their config keys; removed
  import-time env reads (`storage.py`) and dotenv-at-import (lazy in
  `get_env_var`).
- **Phase 2** (done, `8957b83` + `6fceb7f`): `curious_george/` package;
  `RLutils` is a shim. `PredictivePPOAlgo` is a facade over
  `world_model/adapter.py` (PRNNAdapter - the only rollout-time prnn seam),
  `rl/rewards.py`, `rl/buffer.py` (compute_gae), `rl/ppo.py` (ppo_update).
  Intrinsic-reward code intentionally left inline (off in mainline config;
  fixture can't protect a move). Golden fixture bitwise match after each step.
- **Phase 3** (done): `world_model/device.py` on_device/eval_mode
  (PredictiveNet-aware - they move `pN.state` with the weights; a real run
  caught the state-left-on-cuda crash). Agent device is dynamic (follows
  acmodel); no more hidden `.to("cpu")` in `getObservations`; all band-aids
  in trainRL/define_task replaced with contexts. `envs/access.py`
  (base_env/grid_shape/subroom_size/get_goal_loc) replaces every
  `env.env...` reach-in. `rl.reward_alignment` config flag (legacy default;
  `next_obs` implemented + unit-tested; episode-boundary behavior
  documented in `rewards.py`). `evaluation/spatial.py` returns
  {sRSA, SWdist, SI} (SWdist was wandb-only before). Fixed 2 pre-existing
  `test_ckpts` failures: the stub said `with_obs=True` but the `.env`
  checkpoints are from the noObs config.
- **Phase 4** (done): `OnPolicyAnalysis(reuse_last_rollout=True)` analyzes
  the training rollout already in the algo's buffers instead of collecting
  25,000 fresh steps - measured 0.8ms vs ~48s (CPU) per analysis interval;
  trainRL uses it except on the random-agent path (whose buffers are never
  filled). `quantifyObjectLearning` now uses the vectorized
  `get_obs_at_loc_fast` (looped versions kept solely as the tests' reference
  oracle); OMT outputs still bitwise-equal to pre-refactor.
  `subroom_size` cached out of the per-step hot loop. Deferred to Phase 5:
  batching getTestTrial's B predict calls (needs the direct pN.pRNN 4-D
  call machinery Phase 5 builds anyway); `preprocess_images` micro-opt
  skipped (uint8->float conversion forces the copy regardless; the real fix
  is batched collection).
- **Phase 5** (done): `exp.num_envs` (default 1 = untouched serial path).
  `BatchedSRTracker` (batched stateful predict_single equivalent via direct
  `pN.pRNN.rnn(batched=True)` calls; proven exactly equal to serial streams
  at zero noise incl. per-env resets) + `BatchedPredictivePPOAlgo` (lockstep
  B-env collection, env-major flat layout, per-stream GAE). End-to-end runs
  pass at num_envs=1 and 2. No upstream pRNN change was needed.
- **Phase 6** (done): RLutils shim deleted; all imports (trainRL, tasks,
  tests, scripts) migrated to curious_george; comparison harnesses stay
  dual-compatible for worktree runs; pyproject updated;
  docs/refactor_notes.md written (contracts, batched-mode constraints,
  follow-up experiments). Final: 187 passed / 16 failed (all pre-existing
  wandb_data/analysis_omt) / 7 deselected; golden fixture bitwise match.

## Verification evidence

- Test counts now: **181 passed / 16 failed / 7 deselected** (baseline was
  169/18; the +12 = 10 new alignment/device tests + 2 fixed ckpt tests; the
  16 remaining failures are the pre-existing test_wandb_data (12) and
  test_analysis_omt (4) ones).
- Golden fixture: bitwise match at every phase gate (fresh-net training path).
- **Old-vs-new I/O comparison with real checkpoints** (harnesses:
  `tests/golden/compare_io.py`, `tests/golden/compare_omt.py`; old side run
  in a git worktree at `a6eb212`; both sides CPU, per-section seeds):
  - A: ckpt-loaded algo collect+update - curious rewards, advantages,
    actions, log_probs, SRs, locs, losses, post-update weight sums: EQUAL.
  - B: ActorCriticAgent.getObservations 50-step rollout - actions,
    positions, dirs, SRs, obs images: EQUAL.
  - C: on-policy sRSA via calculateSpatialRepresentation: 0.513182 both.
  - D: OMT getTestTrial + quantifyObjectLearning (2 trajs) - obs/pred
    tensors, positions, goalmodulation, inviewtimes, all metrics: EQUAL.
- All 8 `.env` checkpoints (LRoom cur/rand, FourRooms cur/rand/start-rand
  pRNNs; 3 AC status ckpts) load and forward-pass through the refactored
  code.
- Short end-to-end training run (64 frames, analysis every update) exercises
  plot + spatial eval + device contexts without manual `.to()` calls.

## Known intentional behavior notes

- `evaluate_spatial_representation` adds a second short rollout for the
  returned SWdist (the internal one inside calculateSpatialRepresentation
  still wandb-logs as before). Dedupe is a Phase 4 item.
- The decoder trained inside the sRSA eval is discarded by the training
  loop (pre-existing; ~70s/eval on CPU) - Phase 4 candidate: trainDecoder
  flag or smaller numBatches.
- `align_to_next_obs`: the last action of each episode keeps its own
  (legacy) error - the prediction targeting the episode's final obs is not
  produced by the per-episode predict pass.

## Modularity pass (2026-07-05, commits 6f86794 / 60eb7bf / cf11953)

- **rl/ = collect/ + update/ + algo.py.** ONE `PredictivePPOAlgo` for B>=1
  (env or list of envs as first arg; `BatchedPredictivePPOAlgo` deleted).
  B=1 goes through the unified collector with a `SingleSRTracker` that
  delegates to the exact predict_single/reset_state calls - **golden fixture
  bitwise match held**, including the RNG order
  (sample -> env.step -> SR noise -> reset noise -> env.reset).
  update/ holds the RL math: losses.py (`ppo_clip`, `a2c` behind one
  signature + LOSSES registry, selected by `rl.loss`), updater.py
  (loss-agnostic driver), advantage.py, rewards.py, world_model.py.
  collect/ holds collector.py (RolloutConfig/CollectorState/CollectResult
  dataclasses), diagnostics.py, agent.py, format.py.
- **training/ + main_train.py.** setup.py (functions -> RunContext /
  TrainingComponents dataclasses; one construction path), loop.py
  (run_training + analysis/checkpoint functions), logging.py (explicit
  dicts, historical wandb keys; per-action curious_reward_*/avg_adv_* now
  actually forwarded). trainRL_Adel.py is a shim. **A/B gate**: old script
  vs main_train, same seed, ckpt every update -> AC/optimizer/pRNN weights
  bitwise-identical.
- **Configs.** Hydra groups env/model/world_model/algo/rewards composing
  into the historical key paths via @package - zero call-site changes
  (tasks/ needed none as a result); components swap from the CLI.
- **Bit-compat summary**: everything gated bitwise passed - no tolerance
  threshold was needed. Known non-bit-comparable by design (unchanged from
  Phase 5): B>1 vs B=1 RNG streams. Two intentional behavior deltas:
  large-jump debug no longer dumps debug_locs*.pt (prints only), and B>1
  logs now carry real per-episode returns / dist_travelled.

## OMT refactor (2026-07-06, commits 5f6564f..48ab596)

- **tasks/ObjectMemoryTask -> tasks/omt** with main_task.py entry (was broken:
  still composed the deleted Conf1_Adel config), task.py + metrics.py split
  (quantify_object_learning is a pure function), figure.py unchanged.
- **Task template** `curious_george/evaluation/task.py`: FreezeSpec,
  TaskComponents (explicit env_train vs env_eval - the OMT memory probe
  trains WITH the object and evaluates WITHOUT it), setup_task (reuses
  get_pN/get_SR_acmodel/get_agent/setup_algo; historical construction
  order), train_phase (callback hooks), collect_eval_rollouts (serial,
  bitwise-faithful incl. the pN.state carry-over quirk) and
  collect_eval_rollouts_batched (lockstep env copies + BatchedSRTrackerShim;
  same traj_stats_fn as serial). storage.get_algo deleted (setup_algo is
  the single constructor; gained a device param - the golden gate caught it
  defaulting to global cuda). tasks/template_task.py is the documented
  skeleton for new tasks.
- **Batching**: tasks.testing.batched (eval) + exp.num_envs (train phase,
  via the env-list algo). Not bit-comparable to serial by design; zero-noise
  equivalence pinned in tests/test_omt_batched.py.
- **Gates**: tests/golden_omt/ bitwise fixture (real .env ckpts: construction,
  2 train batches incl. lr scaling, eval trial, metrics, pN_control-frozen
  and env-wiring guards) held through every step; training golden_v0 held;
  suite 194 passed / 16 pre-existing failed; serial/batched/num_envs=2
  smoke runs clean.

## Perf overhaul + GPU work (2026-07-07..15, branches sdu/rl-rollout-arch -> sdu/gpu-batched-wm)

Full detail: docs/perf_baseline.md (methodology), docs/perf_log.md (per-commit
log), docs/perf_changes_2026-07.md (review guide), docs/gpu_batched_wm_plan.md
(current branch plan). Condensed here for hand-off.

### What made CPU fast (218 -> ~1200+ FPS potential, ~950 FPS realized in runs)

1. **CPU > CUDA at this model shape** (hidden=500): kernel-launch + sync
   overhead dominates microsecond-scale ops. `CG_DEVICE=cpu`; training sbatch
   scripts are CPU-only. Full 20.5M-frame run: ~6h wall on 16 cluster cpus.
2. Sync removal (bitwise-gated): GAE numpy scan (100ms->0.7ms), no per-step
   .item()/.cpu(), loss scalars aggregated on-device, plotSampleTrajectory
   gated to logging.plot_interval=200 (was EVERY update).
3. Vectorized SpeedHD formatting in world_model/adapter.py (bitwise-equal to
   env2pred; tests/test_batch_format.py).
4. **exp.num_envs=8 default** (curve-gated: 9-run bsweep, B in {1,4,8} x 3
   seeds - B>1 learning-equivalent; docs/perf_log.md + bsweep.png).
5. **Obs bank** (envs/obs_bank.py): partial RGB obs is a pure function of
   (pos, dir, grid) -> precomputed per-grid-fingerprint banks in
   data/obs_bank/, byte-equal to live renders (goldens prove it end-to-end).
   env_step 6.4 -> 3.2 ms/step at B=8.
6. Eval: pooled multi-trajectory spatial eval (exp.eval_trajs x
   predNet.seqdur, matches training statistics), metrics computed AND
   wandb-logged inside prnn (PredictiveNet.calculateSpatialMetrics, prnn
   branch sdu/prnn-perf-optim). Analysis event 293s x2 -> ~9s x1.
7. Slurm: OMP/MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK in every script
   (unbounded torch threads in a cgroup caused 4-51s update-time thrashing).
8. AsyncVectorEnv collection exists (envs/vector.py, exp.async_envs) but is
   OFF: measured no faster locally (+6-9%) and slower on cluster - per-step
   lockstep IPC latency eats the tiny render's parallelism.

### Why the GPU doesn't help (measured, not assumed)

The wm BPTT is ~5-10k tiny ops per trainStep (256 sequential timesteps x ~8
ops x fwd+bwd). Each op pays a fixed tax (GPU kernel launch 5-10us) that
exceeds its ~2us of arithmetic, and the time recurrence forbids queueing
ahead -> GPU ~91% idle, 558 FPS vs 674 CPU (async_bench_10111153). Batching
segments (8,256) cut trainStep CALLS 8x (wm_train 1.07->0.22s/update, CPU
1197 FPS) but CUDA stayed at the same 0.23s: both now sit on the per-op
dispatch floor. The bill is op COUNT, not op size.

### batched_wm curve gate: FAILED (wmsweep, 2026-07-15)

predNet.batched_wm=true (ONE pooled optimizer step vs 8 sequential per
update) changes learning: 73k vs 10k RMSprop steps per run, pRNN loss ~2x
higher at matched frames; wmsweep arms (lr x1, lr x2) fail the band badly
(cur_reward_mean 0% in-band, value_loss 33-40% off). Flag stays OFF.
Wall-clock it IS 1.6x faster (2019 vs 1278 FPS in full runs). Note
wmsweep-B3-s3 died early (unexamined) - horizon truncated to 1.48M frames.

### Next steps (in agreed order)

1. **torch.compile / CUDA graphs on the thetaRNN cell loop** (option 3,
   semantics-preserving, no curve gate - metric harness only). In-flight
   probes: compiled CELL alone on CPU = 1.25x on serial (1,256); CUDA eager
   batched (8,256) fwd+bwd = 187ms measured; CUDA compiled probe with
   mode="reduce-overhead" TIMED OUT >2min (recompile/capture issue inside
   the 256-step loop - unresolved; try default mode first, then piecewise).
   Real prize: fuse the ~8 ops/timestep and/or capture the launch sequence.
   Blockers catalogued in docs/perf_baseline.md (np.mod, asserts, dynamic
   segment lengths); prnn changes go to branch sdu/prnn-perf-optim + repin.
2. **k-steps middle design** for batched_wm (option 2): k pooled steps on
   (8/k, 256) sub-stacks per update - interpolates step count (dynamics)
   vs dispatch count (speed). Needs small adapter change + wmsweep-style
   curve gate.
3. **Free probe** (option 1, no code): rerun wmsweep arms with lr x4 / x8 -
   likely insufficient (step-count deficit, RMSprop second-moment dynamics)
   but nearly free to check.

### Hand-off state (2026-07-15)

- RL branches: sdu/rl-rollout-arch (perf overhaul, pushed), sdu/gpu-batched-wm
  (current, pushed through 10059a4 + local commits - check `git log`).
- prnn: pinned to BRANCH sdu/prnn-perf-optim in pyproject [tool.uv.sources]
  (uv lock --upgrade-package prnn to advance). Contains
  calculateSpatialMetrics + the predict(batched=True) 4-D prep fix (383ae24).
  NOTE: pRNN repo has a PR-required rule that direct pushes bypass.
- Suite: 110 passed / 0 failed / 7 deselected (test_wandb_data +
  test_figure3_sRSA deleted with approval; goldens pinned to B=1 +
  rewards=curious in capture configs and still bitwise-green).
- Gates ladder: bitwise (goldens) -> metric harness (tests/perf/benchmark.py
  + compare_metrics.py, fixed seed) -> curve gate (multi-seed banded
  comparison via scripts/legacy/analysis_bsweep.py; used for bsweep PASS and
  wmsweep FAIL).
- Paused queue: thcycRNN_5win RL integration (docs/thcyc_rl_integration.md,
  4 open design questions for Sabrina).
