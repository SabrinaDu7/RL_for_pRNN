2026-08-25 · branch `sdu/refactor-after-speed30`

# Compaction: the library/questions split, and the prune

## State

```
RL_for_pRNN        sdu/refactor-after-speed30, ~25 commits, pushed
                   gate: 170 passed, 0 skipped, 1 deselected
../experiment-curiousgeorge   130 passed, 18 skipped, ruff clean
                   exp check Q0 and Q1 both reproduce every value exactly
```

The plan this executed: `docs/claude_logs/refactor_plan_2026-08-25.md`. Phases 0–7 are
done except Phase 6 (deleting `throwaway/ported/`), which is deliberately last.

## The shape now

`RL_for_pRNN` is the library. `tasks/` and most of `scripts/` are gone — moved to the
questions repo or staged in `throwaway/ported/`. Nothing was deleted: you cannot prove a
port bitwise-equal to an original you deleted.

```
curious_george/
  rl/            algo.py is the hub; collect/, update/{policy,world_model,losses,...}
  models/        policy.py (actor-critic), prnn_adapter.py (the only prnn import), device.py
  envs/          factory, access, vector (DeviceTableShellPool), obs_bank, layouts
  evaluation/    spatial, on_policy, probe, task
  training/      setup, loop, schedule, logging
  log_and_store/ storage, provenance, wandb
  check/         wandb_compare
  utils/
docs/            claude_logs/, invalid-runs.md, sab_context/
```

Renames this session, each named for **what the thing is**:
`rl/update/updater.py`→`policy.py`, `world_model/`→`models/`, `models.py`→`models/policy.py`,
`adapter.py`→`prnn_adapter.py`, `io/`→`log_and_store/` (absorbing `storage.py`,
`provenance.py`).

## Bitwise coverage — the thing the prune depends on

```
tests/golden/test_golden.py       seed -> 2 updates -> every tensor + both state dicts
tests/golden/test_golden_eval.py  REFERENCE WEIGHTS -> 1 collect -> 5 metrics
```

The second is new and answers a different question: `test_golden` pins "same seed => same
trajectory" (catches moved RNG); `test_golden_eval` pins "same weights => same metrics"
(catches a metric whose *meaning* changed). Pinned values:

```
prnn_loss 4.307033e-03  mi_policy 0.251099  sRSA 0.479099  SWdist 0.044688
SI 1/500 zeroed, mean(active) 0.946904
```

Both proven to fail on deliberately broken code. **Neither covers the pooled/device path**
— that has equivalence tests (A vs B in one process), not fixtures, so a change breaking
both sides identically passes.

## Findings worth keeping

- **`stay_put` is MiniGrid's `pickup`.** Tested: a `Ball` in front of the agent vanishes
  into `carrying`. `action_space` is `Discrete(4)` and index 3 is `Actions.pickup`. The
  name comes from `utils/common.py:99` and is a wandb key on every run. Inert to date
  (every object used is `Floor`-derived, `can_pickup=False`) but it refuted Q4's premise.
- **`encode_speed_hd_rows` gives the world model a FORWARD BIT ONLY** — every non-forward
  action leaves the action block zero. So a no-op is "not forward, HD unchanged", and a
  *blocked* forward is "you moved" when nothing moved.
- **Reusing a training rollout for spatial metrics needs the onset trim.** The tracker's SR
  is exactly zero on the first step of every segment; `calculateRSA_space` uses cosine,
  undefined on a zero vector, and one such row NaNs the whole matrix.
- **The obs bank is load-bearing, its committed cache was not.** Rebuild is 0.49 s / 0.15 MB
  per grid; it was 246 files with 2 tracked. Untracked now. Fixed on the way: a non-atomic
  write, and `bank_dir=BANK_DIR` captured as a default argument (which made a measurement
  read 0.01 s because it had silently loaded instead of built).
- **`np.savez_compressed` appends `.npz`** if the name doesn't end in it — a `.tmp` path
  becomes `.npz.tmp.npz`. Broke four tests before I saw it.
- **`_reset_streams` draws i.i.d. per stream per episode**, so multi-room training is
  simultaneous *mixing*, not alternation. There is no rate knob.
- **`run_spatial_analysis` returns before the on/off-policy loop when `layouts` is set**, so
  `exp.offpolicy_prnn_eval` is silently ignored on the multi-room path.

## Open

1. **The path gate's blind spots.** `LIVE_GLOBS` doesn't cover `docs/**/*.md`, and the regex
   needs a directory prefix so a bare filename is invisible. `docs/invalid-runs.md` and
   `docs/sab_context/open_choices.md` both still name `tests/golden_omt/`, which moved.
2. **No fixture for the pooled/device path** (see above). This is the biggest coverage gap
   for the pruning ahead.
3. `Configs/main.yaml:8` still has `- tasks: omt` and `Configs/tasks/omt.yaml` still exists,
   configuring code that left. Hydra composes, so nothing breaks.
4. 62 ruff findings and 10 basedpyright errors, both surfaced by widening the configs, both
   recorded rather than fixed.
5. Phase 6: delete `throwaway/ported/` once every port is gated.
6. The eval fixture is pinned against the tracked February checkpoint, not the run you
   named (`fast-single-e0.001to0.01-g8-...-26-08-24-19-30-37`) — that one is on Mila's
   scratch. `GOLDEN_EVAL_CKPT_DIR=... --recapture` re-pins it, and the new checkpoint must
   be tracked.

## Questions repo

Eleven plans in `docs/claude_logs/` (Q2–Q11) plus a methods note. Q1 is answered — the
agent goes to the novel object (approach index 0.0442 ± 0.0130, present > absent at 3 of 3
locations), from the twelve `omt-cur-dot-0730-*` runs with no new training.

**Q6 should probably go first**, as a methods fix rather than a control: Q2–Q5 all inherit
a readout whose sampling is unrecorded and whose off-policy arm does not run.
