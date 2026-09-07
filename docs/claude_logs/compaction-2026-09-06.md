2026-09-06 · branch `sdu/clean-sept` (from `sdu/optim-pred` @ `71890bd`) · audit: `docs/claude_logs/audit-2026-09-05-optim-pred.md` · running log: `docs/claude_logs/cleanup-2026-09-06.md`

# Compaction: the cleanup is mid-flight; here is exactly where

## The task, verbatim from the user

Perform the cleanup in the audit on a new branch (`sdu/clean-sept`) while
preserving bitwise equality in tests and runs for the impassable-objects
multienv and the single L-Room (standard environment) with walkable objects.
When done, the USER runs fresh runs of both to compare against recent runs.
Decisions the user made in the audit session: a2c goes; the B=1/serial
`reference` path stays live (baseline); MSE-on-multienv stays live; goldens
are to be revised to what production runs; anything imported downstream by
`../experiment-curiousgeorge` is flagged, not deleted; `throwaway/` is out
of scope. The user does not want the circuit changed (`action_offset` stays 0;
audit C1 is a methods decision, not cleanup).

## The gate, and how to run it

1. Tests: `CG_DEVICE=cuda uv run pytest -q`. Baseline before any change:
   628 passed, 1 deselected (`throwaway/2026-09-05/pytest_baseline_cuda.txt`).
   Tests deleted WITH the code they pinned so far: `tests/test_async_envs.py`.
   Tests added: `tests/test_audit_2026_09_fixes.py` (C4, C5, C6, C15), plus
   new cases in `tests/test_configs.py` (C2, C8) and
   `tests/test_training_schedule.py` (C3).
2. Fingerprints: `throwaway/2026-09-06/fingerprint.py`. Baseline captured from a
   FROZEN WORKTREE of the audit commit, `../RL_for_pRNN_base` (own `.venv`, run
   with `uv run --project ../RL_for_pRNN_base`), labels `base1` (all four cases)
   and `base2` (prod again; identical to base1, so the GPU path is deterministic).
   After every step:

       RL_STORAGE=$PWD/throwaway/2026-09-06/outputs CG_DEVICE=cuda \
         uv run python throwaway/2026-09-06/fingerprint.py --label stepN --cases all
       uv run python throwaway/2026-09-06/fingerprint.py --diff base1 stepN

   Every case must print IDENTICAL. Cases: `prod` (multienv-fast, Selected
   impassable positions 0 1 2 3 5 6 7 8, CE + focal 5 + MLP + reward norm),
   `walkable` (multienv-fast as shipped, MSE), `parity`, `reference`. A pass is
   about 3.5 minutes. Outputs under `throwaway/2026-09-06/fp/<label>/`.

## Commits so far on `sdu/clean-sept`

- `920c8f8` audit report + baseline gate output.
- `fd75b77` cleanup 1: correctness fixes C2 C3 C4 C5 C6 C7 C8 C15. Gate: touched
  tests pass; fingerprint IDENTICAL on all four cases (`fp/step1`).
- (in progress, uncommitted) cleanup 2: the deletion pass. Full gate +
  fingerprint were launched in the background writing
  `throwaway/2026-09-06/pytest_step2.txt` and `fp_step2.log`. READ THOSE FIRST.
  If both are green, commit everything in the working tree except
  `throwaway/count_params.py` and `throwaway/survey_params.py` (the user's own,
  untracked) with a message listing the deletions below.

### What cleanup 2 deleted or changed (all in the working tree now)

- `rl/algo.py`: `IntrinsicReference`, `compare_trajs`, the `intrinsic`/`k_int`/
  `noise_mu`/`noise_std`/`loss` constructor args, `self.cuda_graph`, async-pool
  handling, subroom plumbing; loss is `ppo_clip_loss` directly.
- `rl/collect/collector.py`: `AsyncShellPool` branch, intrinsic tail, `int_rewards`,
  `check_large_jump` + DEBUG prints, `subroom_ids`, `last_post_obs`; `RolloutConfig`
  lost `intrinsic`/`k_int`. `diagnostics.py` lost `check_large_jump`.
- `rl/update/advantage.py::compute_gae` lost `int_rewards`/`k_int` (x + 0.0 == x, so
  bitwise-safe). `losses.py` lost `a2c_loss` and `LOSSES`; `policy.py` lost the
  string indirection. `world_model.py` lost the `fast_speedhd` condition.
- `models/prnn_adapter.py`: theta-cycle branches and every non-SpeedHD fallback
  gone; the constructor asserts SpeedHD. `models/policy.py` lost the `SR.ndim > 2`
  branch.
- `configs.py`: `MinigridEnv`, `SingleLayout`/`FrozenLayouts`/`LayoutPool`/
  `EnvLayoutSpec`, `EnvBackend.ASYNC/ASYNC_TABLE`, `SpatialEvalPath`,
  `TrainPolicyCfg.intrinsic/k_intrinsic`, `ArchPolicyCfg.input_type`,
  `EvalCfg.spatial_path/legacy_decoder_timesteps/behaviour_timesteps`,
  `RunCfg.video_every_episodes`; `ArchPrnnCfg.action_encoding` is now a fixed
  property (SpeedHD). The `multienv` RETIRED preset is still there (its removal
  needs the setup-golden recapture; see step 5).
- `envs/vector.py`: `AsyncShellPool`, `DropMission`, `PosInfo`, `_make_worker_thunk`.
  `envs/factory.py` rewritten: banked wrapper only (no `wrappers` dict, video,
  `ResetWrapper`, `HDObsWrapper`, `Visual_FO`). `envs/access.py`: viz helpers,
  `get_subroom_id`, `subroom_size`, `get_new_obj_pos` gone. `envs/layouts.py`:
  `generate_layouts` and `LandmarkKind.size` gone; `Layout.min_cell_gap` is a
  property. `utils/enums.py::AgentInputType` is `H_PO`, `H` only.
- `evaluation/spatial.py`: legacy decoder path, `compute_sleep_wake_dist`,
  `_sleep_wake_dist` gone. `evaluation/on_policy.py`: `EnvironmentFeaturesAnalysis`,
  `_compute_deltas`, `plot_deltas` gone; the clone path stays (downstream OMT).
- `log_and_store/wandb.py`: 1,488 -> 455 lines; only `fetch_occupancy_grids` and
  what it needs (downstream Q1) plus `_history_rows` (one test). `storage.py`:
  `get_tmp_dir`, `get_tmp_model_dir`, `get_goal_loc`, `get_video_dir`, the ndarray
  `RAND_ACT_PROBA` copy gone (everyone imports `configs.RAND_ACT_PROBA`).
  `utils/dev_env.py`: `get_wandb_env_vars`, `get_logdir_env_var`,
  `get_root_dir_env_var` gone; filenames live in `utils/checkpoints.py`
  (`PRNN_CKPT_FILENAME`, `POLICY_CKPT_FILENAME`), the one home.
- `rl/collect/format.py`: image + direction only (no mission tokenizer, no `HD`
  branch). `training/logging.py`: no `subroom_ids` histogram.
- `__init__.py` files: the top-level surface is the 16 names the tests and the
  questions repo import; `rl/`, `envs/`, `evaluation/`, `rl/update/` have `__all__`.
- Moved to `throwaway/2026-09-06/`: `tests/golden/compare_io.py` (cross-tree harness
  for a tree that no longer exists) and `tests/perf/`.
- `check/config_keys.py`: removed keys moved to `GONE` with reasons.
- Tests updated for the removed parameters: `test_advantage.py`, `test_reward_norm.py`,
  `test_batched_collector.py`, `test_configs.py`, `test_advantage_normalization.py`,
  `test_update_logs_semantics.py`, `test_cuda_graph_policy.py`, `test_batch_format.py`,
  `test_occupancy_counts.py`, `test_ckpts.py`, `test_exploration_evals.py`,
  `test_random_agent_baseline.py`; `tests/golden/capture_golden*.py` lost the inert
  kwargs (no fixture change: they were inert).

## Remaining steps, in order

3. **Naming** (audit §2): stale Hydra keys in error messages and docstrings
   (`exp.*`, `rl.*`, `predNet.*`, `logging.*`: `rollout_graph.py:1`, `world_model.py`
   docstring, `prnn_adapter.py` docstrings, `models/device.py:46`, `loop.py`
   early-stop message, `policy_graph.py` "rl.frames"); `world_model.device._move()`
   in `prnn_adapter.py`; the fourth action's three names (`access.ACTION_NAMES` is
   gone; make `utils/common.py::mean_by_action`'s names the one home and import them
   in `prediction_figures.py`); `ACModelSR.SR_size`/`SR_single` rename (internal
   only; constructor signature unchanged, downstream constructs it);
   `prediction_mses*` -> `prediction_errors*` (adapter methods; callers:
   collector, rewards.py, 3 tests); `BANK_DIR` climbing path -> under
   `get_storage_dir()`; `agent.py:97` TODO; `provenance.py:184` stale accusation;
   `train_fast.sh:35-36` stale "does not use cuda_graph"; `wandb_compare.py`
   ENTITY/PROJECT -> `RunCfg` defaults + `--project`.
4. **Tooling fixes** (audit §1.2): C9 `checkpoint_series.run_config` reads
   `provenance.json["argv"]` like `prediction_figures.plot_run_predictions` does,
   with the flags as fallback for pre-provenance runs; C10 `Selected`: drop `n`,
   make `positions` the field (default `(0,1,2,3,4)`), update `multienv.sh`
   (`N` -> positions 0..N-1 when no positions given), `checkpoint_series`,
   `action_graph.main`, `tests/test_selected_rooms.py`, `test_slurm_invocations.py`;
   C11 `surprisal_timing.main` / `readout_probe.main` read `action_offset` from the
   run's `provenance.json` when present; C17 one `classes_of(pixels)` in
   `envs/palette.py` used by `error_decomposition.py` and `readout_probe.py`
   (the adapter keeps `predCE.targets_for`); O5 vectorize `LocationStats.update`
   with `np.add.at` (no RNG; fingerprint must stay identical); C14 route
   `ActorCriticAgent.next_SR` through `PRNNAdapter.next_sr` and seed row 0 with
   `adapter.init_sr` (fingerprint `eval_lines` must stay identical; if they move,
   revert this one). Delete the orphaned fixtures `golden_v0..v4.pt`,
   `golden_eval_v1.pt`, `golden_eval_offset1_v1.pt`.
5. **Goldens**: remove `RewardAlignment.LEGACY` and the `"legacy"` defaults
   (`algo.py:127`, `collector.py:117`, `rewards.py`), then recapture
   `capture_golden.py` at `next_obs` (fixture v6) and `capture_golden_eval.py`
   offset-0 at `next_obs` (eval v3); remove the RETIRED `multienv` preset, drop it
   from `tests/test_configs.py::EXPECTED` and `capture_golden_setup.COMPOSITIONS`,
   recapture `golden_setup` (v2). Consider a CUDA-only production fixture
   (prod composition, two rollouts, graphs on) as a third golden; the fingerprint
   harness already shows the GPU path is bitwise reproducible.
6. **Docs**: README presets (five, not three; the `multienv` example), README:160
   boundary claim, `.env` stale `slurm/omt_task.sh` reference; finish
   `cleanup-2026-09-06.md` with one entry per commit (commit, gate, fingerprint);
   update the memory file `audit-2026-09-05-state.md`.
7. Final: full gate + fingerprint IDENTICAL on all four cases; then hand over for
   the user's fresh runs. Remove the worktree with `git worktree remove
   ../RL_for_pRNN_base` only after the last fingerprint.

## Things learned that the next session should not re-derive

- The editable install is a meta-path finder: `PYTHONPATH` cannot redirect
  `import curious_george` to another checkout; a worktree needs its own `uv sync`
  and `uv run --project <worktree>`.
- Removing an always-zero additive term (`+ k_int * int_rewards`) is bitwise-safe
  (verified by fingerprint); removing dropout/noise-consuming calls is NOT.
- `getObservations` (the eval probe) at offset 0 starts from zeros, which equals
  `init_sr`; only offset 1 would differ.
- The launcher test pins the half budget as a shell-variable binding
  (`tests/test_slurm_invocations.py::BINDINGS["BUDGET"]`), now 21,984 / 87,936.
