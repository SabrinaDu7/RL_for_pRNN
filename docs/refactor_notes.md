# Curious George refactor notes

Companion to `docs/refactor_baseline.md` (baselines) and
`docs/refactor_progress.md` (phase log). This file records the contracts and
conventions the refactored code relies on, and the follow-up experiments.

## Package map

```
curious_george/
  common.py            DEVICE, seed, synthesize, mean_by_action
  envs/factory.py      make_env + wrappers
  envs/access.py       base_env / grid_shape / subroom_size / get_goal_loc
                       (the ONLY place that reaches through env wrappers)
  models.py            ACModel, ACModelSR
  rl/algo.py           PredictivePPOAlgo (facade; single-env path, golden-pinned)
  rl/collector.py      BatchedPredictivePPOAlgo (num_envs > 1)
  rl/rewards.py        curiosity reward + reward_alignment modes
  rl/buffer.py         compute_gae
  rl/ppo.py            ppo_update + minibatch indexing
  rl/agent.py          ActorCriticAgent (dynamic device, no hidden .to() moves)
  rl/format.py         obs preprocessing
  world_model/adapter.py  PRNNAdapter + BatchedSRTracker - the ONLY rollout-time
                          prnn seam
  world_model/device.py   on_device / eval_mode (PredictiveNet-aware: they move
                          pN.state with the weights)
  evaluation/spatial.py   evaluate_spatial_representation -> {sRSA, SWdist, SI}
  evaluation/on_policy.py OnPolicyAnalysis (reuse_last_rollout=True is free)
  storage.py           paths, checkpoint factories (env vars read lazily)
```

## Temporal-alignment contract (the load-bearing one)

- All `*_5win` architectures set `predOffset=0` (`pRNN/Architectures.py:1347,
  1822, 2408`): `predict()` returns `obs_pred[t]` targeting `obs[t]` - the
  SAME index. Future prediction comes from `inMask` zeroing the obs input on
  5 of 6 steps, not from a +1 shift. prnn docstrings claiming "t+1" describe
  the base-class default and are wrong for these nets.
- **Curiosity reward (`reward_alignment`)**: `legacy` (default) credits
  action i with the error on `obss[i]` - the PRE-action observation. This is
  the historical behavior and is pinned by the golden fixture. `next_obs`
  credits action i with the error on `obss[i+1]`; the last action of each
  episode keeps its own error (its successor's prediction row is not
  produced by the per-episode predict pass). Flip with
  `rl.reward_alignment=next_obs` and re-baseline sRSA / SWdist / OMT.
- **pastSR convention**: `pastSR = not ("prevAct" in str(pN.pRNN))`.
  pastSR=True pairs with `SpeedHD` (HD from current step, actOffset=0, SR
  aligns to current position); pastSR=False pairs with `SpeedNextHD`.
  Validated by `world_model.adapter.validate_action_encoding`.
- `int_rewards[0]` is always 0 (duplicated first error) - preserved quirk.
- `pRNN.k` exists only on `thcyc*` nets; use `pN.phase_k` or hasattr guards.

## Device policy

No code moves models across devices behind the caller's back. The agent's
`device` follows its AC model. Callers that need CPU (numpy-based analysis:
`calculateSpatialRepresentation`, `plotSampleTrajectory`, OMT test trials)
wrap the call in `on_device([...], "cpu")`, passing **PredictiveNet objects,
not bare modules** - the context also moves `pN.state`, and a hidden state
left on the old device crashes `predict_single`.

## Batched mode (num_envs > 1)

- `BatchedSRTracker` steps B pRNN streams in one forward pass by calling
  `pN.pRNN.rnn(..., batched=True)` directly (the 4-D trailing-batch layout).
  Do NOT use `pN.predict(batched=True)` - known permutation bug (worked
  around the same way in `scripts/analysis_OMT.py:246`).
- Exactly equal to serial `predict_single` at zero noise (tests); with noise
  the streams are distributionally identical but consume the RNG in a
  different order than serial stepping - B>1 runs are not bit-comparable to
  B=1 runs, by design.
- Constraints (asserted): masked (`thRNN_*win`) pastSR nets only, no
  intrinsic rewards, `num_frames % B == 0`, per-env T divisible by seqdur.
- Flat experience layout is env-major (`index = b*T + t`); episode segments
  never span env boundaries; GAE runs per env stream.

## Verification artifacts

- `tests/golden/golden_v0.pt` + `capture_golden.py`: bitwise oracle of the
  pre-refactor training path (fresh nets, CPU, seed 2). Regenerate and
  compare after any change to the B=1 path.
- `tests/golden/compare_io.py` / `compare_omt.py`: cross-version harnesses
  (run the old side in a `git worktree` at the pre-refactor commit; both
  import RLutils or curious_george, whichever exists). All sections matched
  bitwise at handover, using the real `.env` checkpoints.

## Follow-up experiments / known debts

1. Flip `rl.reward_alignment=next_obs`; re-baseline the three metrics
   (sRSA high, SWdist low, OMT novel-object approach).
2. The decoder trained inside the sRSA eval is discarded by the training
   loop (~70s/eval on CPU); consider `trainDecoder=False` or fewer batches
   when only sRSA/SWdist are wanted.
3. `evaluate_spatial_representation` runs a second short rollout for its
   returned SWdist (the internal wandb-logged one uses different wake data);
   dedupe by having prnn return SWdist.
4. Batch `getTestTrial`'s B predict calls with the direct `pN.pRNN` 4-D
   pattern (collection stays serial; prediction batches).
5. `init_SR` on episode done uses the pre-reset final obs for non-pastSR
   nets (stale-obs quirk, irrelevant for pastSR mainline) - preserved.
6. `docs/*` is gitignored; refactor docs are force-added. Decide whether
   docs should be tracked.
