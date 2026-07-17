# Experiment: curiosity reward alignment `next_obs` — first wandb probe

**Date:** 2026-07-04
**Purpose:** First run of the corrected curiosity-reward indexing
(`rl.reward_alignment=next_obs`) to verify the flag works end-to-end and to
inspect what lands in wandb, before committing to full-scale re-baselining of
sRSA / SWdist / OMT.

## Background

All `*_5win` pRNN architectures set `predOffset=0`, so `predict()` returns
`obs_pred[t]` targeting `obs[t]` — the same timestep (the "prediction" comes
from `inMask` zeroing the obs input on 5 of 6 steps, not a +1 offset). The
historical (`legacy`) curiosity reward therefore credits action *i* with the
prediction error on `obss[i]`, the observation seen **before** the action —
surprise the action didn't cause. Full contract: `docs/refactor_notes.md`,
implementation: `curious_george/rl/rewards.py`.

## Methodology

- New config flag `rl.reward_alignment: {legacy, next_obs}` (default
  `legacy`, which is bitwise-pinned to pre-refactor behavior by
  `tests/golden/golden_v0.pt`).
- `next_obs` mode: `align_to_next_obs()` shifts the per-step MSEs within each
  episode so action *i* is credited with the error on `obss[i+1]`. Episode
  boundary: the **last action of each episode keeps its own (legacy) error**,
  because the prediction row targeting the episode's final observation is not
  produced by the per-episode predict pass. Shift never crosses episode
  boundaries (unit-tested with a stub net in
  `tests/test_reward_alignment.py`, including single-step episodes).
- Probe run command:

  ```bash
  uv run python trainRL_Adel.py exp.exp_name=next_obs_check \
      rl.reward_alignment=next_obs rl.steps=20480 \
      logging.analysis_interval=5 logging.save_interval=0
  ```

  20,480 steps = 10 updates × 2048 frames, fresh pRNN + fresh ACModelSR
  (mainline `thRNN_5win`, `SpeedHD`, curious agent, pRNN co-trained,
  `num_envs=1`), spatial analysis at updates 5 and 10, no checkpoint saving.
- wandb run: `blake-richards/curious-george/next_obs_check_curious_26-07-04-16-12-52`
  (`rl.reward_alignment` is recorded in the run config).

## Discoveries along the way

- **First launch failed with wandb 403** (`permission denied` on
  `upsertBucket` for entity `blake-richards`): the local machine was logged
  into the wrong wandb account. Fixed by re-running `wandb login` with the
  correct account; rerun succeeded. Nothing code-related.
- The expected logged keys for this run (post-refactor):
  - per update: `cur_reward_{mean,std,abs_mean}`, per-action
    `curious_reward_{left,right,forward,stay}`, `avg_adv_*`, returns,
    values, advantages, `policy_loss`/`value_loss`/`grad_norm`, `MI_policy`,
    `loc_entropy`/`loc_entropy_5`, `subroom_ids`, `dist_travelled`, sample
    trajectory figure;
  - at analysis intervals: `sRSA_onPolicy`/`sRSA_offPolicy` (+ internal
    prnn-side SWdist logging), the new `SWdist_direct_{on,off}Policy`
    scalars from `evaluation/spatial.py`, and the on-policy analysis plots
    (now computed from the training rollout itself — Phase 4 change — so
    they reflect the sampled policy on the last 2048 training frames, not a
    fresh 25k-step eval collection).
- Caveat for interpreting this probe: 10 updates from a **fresh** pRNN is far
  too short for meaningful sRSA/SWdist — this run validates plumbing and
  logging, not the science. The scientific comparison (legacy vs next_obs at
  full training length, matched seeds) is the follow-up.

## Results

Run finished cleanly (exit 0, no tracebacks; ~8 min wall clock, FPS ≈ 282,
run config confirms `rl.reward_alignment: next_obs`).

Final-update summary (update 10, fresh nets — plumbing check, not science):

| key | value |
|---|---|
| `cur_reward_mean` / `std` / `max` / `min` | 0.0209 / 0.0054 / 0.0344 / 0.0117 |
| `advantages_mean` / `std` | -0.0004 / 0.159 |
| `policy_loss` / `value_loss` | -0.00103 / 0.0242 |
| `policy_entropy` | 1.92 (near-uniform 4-action policy, expected this early) |
| `pRNN loss` | 0.0218 |
| `loc_entropy` / `loc_entropy_5` | 6.56 / 7.26 |
| `sRSA_onPolicy` / `sRSA_offPolicy` | 0.050 / 0.080 (update 10; update 5 was 0.053 / 0.106) |
| `SWdist_direct_onPolicy` / `offPolicy` | 0.0050 / 0.0065 (agree with prnn-internal `SWdist_onPolicy` 0.0050 / 0.0064) |
| `MI_policy` = `MI_policy_eval` | 0.0034 (equal by construction — Phase 4 rollout reuse) |

Checklist:

- [x] Run completed without errors; keys present in wandb, including both
  SWdist variants and the analysis plots.
- [x] sRSA / SWdist plumbing works at both analysis events. Values are tiny
  as expected for a 20k-step fresh pRNN (baseline trained ckpt: sRSA≈0.44).
- [x] `SWdist_direct_*` (new return-value path) matches the prnn-internal
  wandb numbers to ~1e-4 — the two computations use different wake rollouts,
  so exact equality is not expected.
- [ ] `cur_reward_*` vs a matched-seed `legacy` run — not run yet; needed
  before interpreting the alignment's effect.
- [ ] Decision: full-length legacy-vs-next_obs comparison.

**Discovery (logging gap, pre-existing):** the per-action curiosity keys
(`curious_reward_{left,right,forward,stay}`) and `avg_adv_*` are computed in
`collect_experiences` logs but only the OMT task (`define_task.py`) forwards
them to wandb; `trainRL_Adel.py` logs only the aggregate `cur_reward_*`
stats. If per-action curiosity is wanted in training runs, trainRL's header/
data block needs a one-line addition.

## Update 2026-07-06: boundary special case removed

The original `next_obs` implementation shifted the legacy MSE vector within
each episode and let the LAST action keep its own (legacy) error, because the
per-episode predict pass produced no prediction row for the episode's final
observation. This is now fixed at the source: the adapter extends each
episode's predict pass by one step that feeds `last_obs` with a zeroed action
row (the same zero-action convention `init_sr` uses), so the final
observation is a real prediction target and EVERY action's reward is computed
identically - no boundary case. Verified with a real-net test:
`next_obs[:-1] == legacy[1:]` within episodes (causality: the extra step
cannot affect earlier rows) and the final reward is a genuine, distinct
error. The probe run above predates this change (its last-step-per-episode
rewards used the duplicate); rerun before drawing conclusions from
per-episode-boundary steps (1 in 256 frames at seqdur=256).
