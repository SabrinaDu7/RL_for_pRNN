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
- **Phase 4** (next): perf - OnPolicyAnalysis reuse, vectorized
  get_obs_at_loc_fast swap, batched np->torch, batched getTestTrial predict
  (direct pN.pRNN 4-D call, NOT predict(batched=True)).
- **Phase 5** (pending): [T,B] buffer/collector, SyncVectorEnv, B=1 default.
- **Phase 6** (pending): delete RLutils shim, migrate scripts/, docs.

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
