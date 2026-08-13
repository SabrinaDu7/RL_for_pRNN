# OMT on the post-migration stack: three object locations (2026-07-30)

> **STATUS (2026-08-03): SUPERSEDED, kept for the refactor/runnability record.**
> The three runs described here predate `86f31be` and kept only checkpoints 0 and 2992.
> They were re-run with dense checkpoints on 2026-07-30 (jobs 10252642/10252658/10252659).
> The "Next steps" below are all obsolete: `scripts/legacy/analysis_OMT_h.py` was NOT patched — new
> tooling was written instead (`scripts/trace_*.py`). For the science see
> `exp_object_trace_cells_2026-07-30.md`; for the interventions see
> `exp_object_into_hidden_state_2026-08-01.md`.

## Purpose

Two things at once:

1. **Runnability gate.** Confirm `tasks/omt/main_task.py` still works after the
   `curious_george/` reorganisation and the `SabrinaDu7/pRNN →
   LevensteinLab/pRNN` migration (`migration_prnn_new.md`,
   `migration_baseline.md`).
2. **The experiment proper.** Run OMT three times, each with a different novel
   object location, so the resulting pRNN checkpoints can be analysed for
   hidden-state spatial tuning localised near the object. Tuning that *follows*
   the object across locations is the result; tuning fixed in room coordinates
   is not.

## Methodology

Seed checkpoint for all runs: `pRNN_curious_26-07-23-10-06-25` (a finished
`main_train.py` run on the post-migration stack).

- Local: `outputs/ckpts/pRNN_curious_26-07-23-10-06-25/`
- Mila: `$SCRATCH/pRNN/RL_for_pRNN_10178850/pRNN_curious_26-07-23-10-06-25/`

Three cluster jobs, **same seed (5200) so object location is the only
variable**, via `slurm/omt_task.sh`:

| job | object loc | seed |
|---|---|---|
| 10251399 | `[7, 11]` | 5200 |
| 10251423 | `[14, 7]` | 5200 |
| 10251424 | `[7, 2]` | 5200 |

```bash
OBJ_LOC='[7,11]' SEED_START=5200 SEED_END=5200 \
  sbatch --job-name=OMT_cur_obj_7_11 --time=4:00:00 slurm/omt_task.sh
```

Comparison baseline: wandb run `omt-cur-dot-0218-213634`
(`blake-richards/curious-george-omt`, id `apb8yuc5`) — pre-migration, and
conveniently also object location `[7, 11]`, so `[7,11]` is a direct
like-for-like comparison.

## Changes made to enable this

1. **One output root.** OMT passed a *relative* `save_path`, so checkpoints
   landed in the CWD instead of `RL_STORAGE`. A cluster job that rsynced only
   `$RL_STORAGE` silently lost every OMT checkpoint. Now
   `$RL_STORAGE/<run_name>/<traj_count>/`.
2. **Portable checkpoint resolution.** `get_ckpt_env_vars` now prefers a single
   directory variable (`CUR_CKPT_DIR` / `RAND_CKPT_DIR` / `FOURROOM_*`) and
   derives `predictiveNet_state.pt` + `status.pt` from it. Same variable name
   locally and on Mila; only the value differs.
3. `tasks.testing.trajs` constraint corrected to "smaller than **or equal to**"
   `rl.trajs_per_batch` (both are 8 at the defaults).

Commits `0f67998`, `ec35f34` on `sdu/omt-and-trace-cells`.

## Results

**Runnability: confirmed.** Local CUDA smoke run completed end-to-end
(construction, training, eval probe, `quantify_object_learning`, figure path,
checkpoint writes). Cluster job 10251399 ran at a steady ~2.1 s/batch over 375
batches — roughly 20 min/run, well under the historical ~1h.

**Test gate:** `uv run pytest` → **114 passed, 0 failed, 7 deselected**,
identical to the pre-change baseline.

All three jobs COMPLETED, exit `0:0`, elapsed 21:45 / 34:48 / 33:01 (node
variation; a single run is ~20-35 min vs the baseline's 63:37).

**Final metrics, whole-run means:**

| object loc | job | pRNN loss | cur_rewards | goal mod | goal−ctrl | neg |
|---|---|---|---|---|---|---|
| `[14, 7]` | 10251423 | 0.00772 | 0.00595 | +0.0362 | +0.0353 | 0/11 |
| `[7, 11]` | 10251399 | 0.00736 | 0.00569 | +0.0401 | +0.0240 | 2/11 |
| `[7, 2]` | 10251424 | 0.00749 | 0.00575 | +0.0575 | +0.0419 | 1/11 |
| baseline `[7,11]` | — | 0.00768 | 0.00594 | +0.0352 | +0.0417 | 0/10 |

`pRNN loss` and `cur_rewards` land within a few percent of baseline (new runs
slightly *lower*). `goal−ctrl` is positive in all three.

**Retracted:** an interim reading of job 10251399 at n=200/n=50 showed
`pRNN loss` +18% (~3.3σ) and `cur_rewards` +25%, and `Goal Minus Ctrl` opening
negative (`[-0.0003, -0.0536, ...]`) against a baseline with zero negatives.
Both dissolved with the full run — the early window was not representative and
should not have been characterised as a settled shift. The `[7,11]` run's
`goal−ctrl` mean (+0.0240) is still below baseline (+0.0417), but `[14,7]` and
`[7,2]` bracket the baseline, so there is no evidence of a systematic problem.

**Confound in the baseline comparison (important).** The 0218 baseline predates
`reward_alignment` and therefore ran the `legacy` alignment (curiosity reward
credits an action with the error on the PRE-action observation). All three new
runs use `next_obs` (`Configs/rewards/curious_next_obs.yaml`, the `main.yaml`
default; confirmed in each run's wandb config and traced through
`training/setup.py:219` → `algo.py:257` → `collector.py:326` →
`rewards.py::REWARD_ALIGNMENTS`). So new-vs-baseline compares two different
reward definitions, not just two code versions. It is a sanity check, not a
controlled comparison.

**Checkpoints:** only traj 0 and 2992 exist for these three runs, because
`slurm/omt_task.sh` passed `saving_interval=1000000`. Fixed afterwards (see
below); re-run if the trajectory-resolved series is needed.

## Discoveries along the way (not all fixed)

- **Committed merge-conflict markers.** `slurm/train_prnn.sh` on branch
  `sdu/prnn-new-migration` has unresolved `<<<<<<< HEAD` / `>>>>>>> parent of
  971ff0b` markers *committed into it*. A hand-fix existed in the cluster
  working tree but was never committed; it is now stashed on Mila
  (`git stash list`). **Not fixed** — that branch needs cleaning.
- **`tests/golden_omt` had no pinned checkpoint.** It called
  `get_ckpt_env_vars()`, inheriting whatever `.env` pointed at, so repointing
  `CUR_CKPT_DIR` broke 3 bitwise tests. The diff was exactly the difference
  between the two checkpoint *files*, not numeric drift. Fixed by pinning the
  fixture to the 02-15 run that captured it; no assertions changed.
- **`--export` cannot carry a hydra list.** `sbatch --export=ALL,OBJ_LOC=[7,2]`
  splits on the comma inside the brackets and tears the literal in half. Set
  the vars in the submitting shell instead. Caught before any submission.
- **`scripts/legacy/analysis_OMT_h.py` path drift (not fixed).** `get_ckpts` hardcodes
  `omt-cur-dot-noObs-goal{i}{j}/{step}/pN-{step}.pt`, which is neither the name
  nor the location `main_task.py` produces. Needs patching to
  `$RL_STORAGE/<run_name>/<step>/pN-<step>.pt` before the hidden-state
  analysis can consume these checkpoints. ~~**This is the next blocker.**~~ It was not — `scripts/trace/trace_probe.py` and friends replaced that path entirely; `analysis_OMT_h.py` remains unpatched.
- **`main_task.py` hardcodes `DEVICE = torch.device("cuda")`** with no CPU
  fallback — OMT jobs must request a GPU.
- **Only first+last checkpoints were saved.** `slurm/omt_task.sh` overrode
  `saving_interval=1000000`, which disabled the intermediate saves the config
  already defaulted to. Fixed: the override is gone, and the key is now
  `saving_interval_trajs` (default 200), denominated in TRAJECTORIES and
  converted to batches in `task.py`, so the spacing no longer silently depends
  on `rl.trajs_per_batch`. Verified on a 600-traj run: saves at 0, 200, 400,
  592. The three runs above predate this fix.
- **Output path simplified** from `$RL_STORAGE/omt/<run_name>/` to
  `$RL_STORAGE/<run_name>/`.

## Next steps

1. Collect the three finished runs' checkpoints from `$SCRATCH/pRNN/<JOB_ID>/`.
2. Patch `scripts/legacy/analysis_OMT_h.py::get_ckpts` for the new output layout.
3. Hidden-state analysis: PCA / spatial tuning near the object, compared across
   the three locations.
4. Follow-up recorded in `tests/golden_omt/test_golden_omt.py`: add a
   bound-based test (finite outputs, `pN_post` diverges from `pN_control`,
   non-degenerate `curious_rewards`) that runs on `CUR_CKPT_DIR` and answers
   "does a *new* checkpoint still behave on OMT" — a question the bitwise
   golden does not ask.
