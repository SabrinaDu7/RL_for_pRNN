# Curious George refactor notes

Companion to `docs/refactor_baseline.md` (baselines) and
`docs/refactor_progress.md` (phase log). This file records the contracts and
conventions the refactored code relies on, and the follow-up experiments.
Last updated 2026-07-05 (post modularity pass + utils move).

## Package map

```
main_train.py            training entry (hydra); trainRL_Adel.py is a shim
curious_george/
  utils/
    common.py            DEVICE, seed, synthesize, mean_by_action
    enums.py             AgentType, AgentInputType
    checkpoints.py       StatusCkptKeys, load_statedict_from_acmodel_status
    dev_env.py           get_env_var + ckpt/wandb env-var helpers (lazy dotenv)
  envs/factory.py        make_env + wrappers
  envs/access.py         base_env / grid_shape / subroom_size / get_new_obj_pos
                         (the ONLY place that reaches through env wrappers)
  models.py              ACModel, ACModelSR
  rl/algo.py             ONE PredictivePPOAlgo for B>=1 (env or list of envs);
                         thin composition over collect/ + update/
  rl/collect/
    collector.py         unified rollout loop (RolloutConfig / CollectorState /
                         CollectResult dataclasses; env-major flat layout,
                         per-stream GAE)
    diagnostics.py       joint-dist + location-entropy bookkeeping (no RNG)
    agent.py             ActorCriticAgent (dynamic device, no hidden .to())
    format.py            obs preprocessing
  rl/update/             ALL the RL math lives here
    losses.py            ppo_clip + a2c behind one signature; LOSSES registry,
                         selected via rl.loss / algo= config group
    updater.py           loss-agnostic epoch/minibatch/optimizer driver
    advantage.py         compute_gae
    rewards.py           curiosity reward + reward_alignment modes
    world_model.py       per-episode pRNN training schedule
  training/
    setup.py             construction functions -> RunContext / TrainingComponents
    loop.py              run_training + analysis/checkpoint functions
    logging.py           explicit wandb dicts (historical key names)
  world_model/adapter.py PRNNAdapter + SR trackers (Single/Batched/Null via
                         make_sr_tracker) - the ONLY rollout-time prnn seam
  world_model/device.py  on_device / eval_mode (PredictiveNet-aware: they move
                         pN.state with the weights)
  evaluation/spatial.py  evaluate_spatial_representation -> {sRSA, SWdist, SI}
  evaluation/on_policy.py OnPolicyAnalysis (reuse_last_rollout=True is free)
  storage.py             paths, checkpoint factories (env vars read lazily)
```

## Configs (hydra groups)

Groups compose into the historical key paths (`exp` / `predNet` / `rl`) via
`@package` directives, so call sites are unchanged. Swap components from the
CLI:

```
env=lroom|fourrooms          model=sr_ac|plain_ac
world_model=thrnn5win|thrnn5win_prevact   (pairs pRNNtype with its action encoding)
algo=ppo|a2c                 (sets rl.loss - the LOSSES registry key)
rewards=curious|curious_next_obs          (sets rl.reward_alignment)
```

## Temporal-alignment contract (the load-bearing one)

- All `*_5win` architectures set `predOffset=0` (`pRNN/Architectures.py:1347,
  1822, 2408`): `predict()` returns `obs_pred[t]` targeting `obs[t]` - the
  SAME index. Future prediction comes from `inMask` zeroing the obs input on
  5 of 6 steps, not from a +1 shift. prnn docstrings claiming "t+1" describe
  the base-class default and are wrong for these nets.
- **Curiosity reward (`rl.reward_alignment` / `rewards=` group)**: `legacy`
  (default) credits action i with the error on `obss[i]` - the PRE-action
  observation. This is the historical behavior and is pinned by the golden
  fixture. `next_obs` credits every action i - including the last of each
  episode - with the error on the observation it produced: the per-episode
  predict pass is extended by one zero-action step (the init_sr convention)
  so the final observation is a real prediction target. No boundary special
  case. First wandb probe run: docs/exp_reward_alignment_next_obs.md
  (predates this; its runs used the old duplicate-last-error shift).
- **pastSR convention**: `pastSR = not ("prevAct" in str(pN.pRNN))`.
  pastSR=True pairs with `SpeedHD` (HD from current step, actOffset=0, SR
  aligns to current position); pastSR=False pairs with `SpeedNextHD`.
  Validated by `world_model.adapter.validate_action_encoding`; the
  world_model config group keeps the pairs together.
- `int_rewards[0]` is always 0 (duplicated first error) - preserved quirk.
- `pRNN.k` exists only on `thcyc*` nets; use `pN.phase_k` or hasattr guards.

## Rollout RNG order (what keeps B=1 bitwise)

Per step: policy `sample` -> `env.step` -> SR noise (predict_single) ->
on done: reset noise (`reset_state(randInit=True)`) -> `env.reset`.
End of collect: one more `reset_state`. The unified collector preserves this
exactly at B=1 by delegating SR stepping to `SingleSRTracker` (the literal
predict_single/reset_state calls). Any reordering breaks the golden gate.

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
  B=1 runs, by design. B>1 episode resets are to ZERO state/phase, not the
  serial randInit draw (documented Phase 5 semantic).
- Constraints (asserted): masked (`thRNN_*win`) pastSR nets only, no
  intrinsic rewards, `num_frames % B == 0`, per-env T divisible by seqdur.
- Flat experience layout is env-major (`index = b*T + t`); episode segments
  never span env boundaries; GAE runs per env stream.

## Verification artifacts

- `tests/golden/golden_v0.pt` + `capture_golden.py`: bitwise oracle of the
  pre-refactor training path (fresh nets, CPU, seed 2). Regenerate and
  compare after any change to the B=1 path. Held through: package
  extraction, algo split, collector unification, training/ split, config
  regrouping, utils move.
- `tests/golden/compare_io.py` / `compare_omt.py`: cross-version harnesses
  (run the old side in a `git worktree` at the pre-refactor commit
  `a6eb212`; dual-compatible imports). All sections matched bitwise, using
  the real `.env` checkpoints.
- A/B entry-point gate: old trainRL script vs `main_train.py`, same seed,
  checkpoint every update -> AC/optimizer/pRNN weights bitwise-identical.
- Unit tests: `tests/test_reward_alignment.py` (alignment contract, device
  contexts, conventions), `tests/test_batched_tracker.py` (batched ==
  serial streams), `tests/test_batched_collector.py` (B=2 invariants).

## Follow-up experiments / known debts

1. Full-length legacy-vs-next_obs comparison (matched seeds); re-baseline
   sRSA / SWdist / OMT. Probe run done (see
   docs/exp_reward_alignment_next_obs.md).
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
7. `hardware.use_gpu` / `which_gpu` config keys are ignored (DEVICE global
   always picks cuda:0 when available) - honor or drop.
8. `curious_george/utils/` has no `__init__.py` (works as an implicit
   namespace package); add one for convention if desired.

## 2026-07-15 update (perf overhaul + GPU work) - corrections to the above

Several notes above are now STALE; current truth:

- **predict(batched=True) is FIXED** (prnn 383ae24 on branch
  sdu/prnn-perf-optim, pinned in pyproject): input prep now emits the 4-D
  (1, L, X, B) layout. trainStep(batched=True) works end-to-end and is used
  by adapter.train_on_episodes_batched (predNet.batched_wm flag, default OFF
  - failed its curve gate, see refactor_progress.md). BatchedSRTracker's
  direct rnn() call remains for single-step SR. Old "do NOT use" note above
  is obsolete; baseline flaw #4 resolved upstream.
- **Defaults changed**: exp.num_envs=8 (bsweep curve-gated),
  exp.async_envs=False (AsyncShellPool in envs/vector.py exists, tested,
  measured not-faster), CG_DEVICE controls device (hardware.* config block
  REMOVED - debt #7 done). CPU is the training device by verdict.
- **Spatial eval**: pooled multi-trajectory (exp.eval_trajs x predNet.seqdur)
  via PredictiveNet.calculateSpatialMetrics - computed AND wandb-logged
  inside prnn (Sabrina's rule: pRNN metrics belong to prnn). Debts #2 and #3
  above are done (no decoder by default, no duplicate SWdist rollout).
  Legacy prnn path behind exp.eval_decoder=True. Reference values shifted by
  design (training-matched statistics); sRSA/SI are rollout-length
  sensitive.
- **Obs bank** (envs/obs_bank.py): BankedRGBPartialObsWrapper replaces
  RGBImgPartialObsWrapper_HD in factory + async thunks; per-grid-fingerprint
  banks live in data/obs_bank/ (committed). Byte-equal to live renders;
  rebuilt automatically for new layouts.
- **Golden capture configs are PINNED** to exp.num_envs=1 + rewards=curious
  (main.yaml defaults moved on; fixtures unchanged and green).
- **Tests deleted with approval** (2026-07-09): test_wandb_data.py (12
  failing + 102 passing) and test_figure3_sRSA.py. Suite: 110 passed /
  0 failed. Sabrina's standing rule: NEVER change a failing test without
  consulting her first.
- Perf docs: perf_baseline.md / perf_log.md / perf_changes_2026-07.md /
  gpu_batched_wm_plan.md. Next steps live in refactor_progress.md's
  hand-off section (torch.compile on the cell loop, then k-steps batched_wm
  middle design, then lr free-probe).

## 2026-07-17 update (prnn-new migration) - supersedes pin/golden notes above

- **prnn pin**: now LevensteinLab/pRNN branch `sdu/rl-integration` (curated
  re-port of the fork's RL surface; see docs/migration_prnn_new.md +
  migration_baseline.md). SabrinaDu7/pRNN is retired.
- **Goldens**: `tests/golden/golden_v1.pt` pins the new stack (capture
  config unchanged: B=1, rewards=curious, seed 2, CPU). golden_v0.pt kept
  for the legacy stack only. golden_omt_v0 still valid (ckpt-based, passed
  bitwise unchanged).
- **Arch detection**: pastSR/thcyc now key off `pN.pRNNtype` (upstream
  partial factories erased "prevAct" from str(pN.pRNN)).
- **Dynamics**: identical to the old stack EXCEPT biases now train
  (bias_lr 0.01; the fork's int(0.1)=0 froze them - bug). Round-0 rollouts
  are bitwise identical across stacks.
- predict(batched=True) fixed on the new branch; the direct rnn() call in
  BatchedSRTracker remains only because forward(single=True) doesn't
  forward `batched`.
