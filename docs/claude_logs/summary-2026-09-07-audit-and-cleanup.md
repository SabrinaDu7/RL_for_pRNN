# Summary: the 2026-09-05 audit and the 2026-09-06/07 cleanup

One page for a reader who was not there. The detail lives in two records and
this page only points at them:

- the audit report: `docs/claude_logs/audit-2026-09-05-optim-pred.md`
  (read-only, of `sdu/optim-pred` @ `71890bd`, fork `prnn` @ `852cc7d2`)
- the cleanup log, step by step with every gate: `docs/claude_logs/cleanup-2026-09-06.md`

Branch: `sdu/clean-sept`, commits `fd75b77`..`aea0ebf` ("cleanup 1".."cleanup 7")
on top of the audit commit `920c8f8`. Not pushed. Nothing submitted to Mila.

## What was asked

1. Audit the branch, without changing it, on four axes - naming, redundancy,
   optimization, correctness - leading with the RL wiring (rewards to
   experiences, the hidden state and head direction the policy consumes,
   batching equivalence, the two gradient-step counters).
2. Perform the cleanup on a new branch while preserving BITWISE equality, in
   tests and in runs, for the impassable-objects multi-room arm and the single
   L-room with walkable objects.
3. Then run fresh runs of both arms and check they hold up against the most
   recent runs.

## How equality was held

Every code step was gated the same way before it was committed
(`throwaway/2026-09-06/fingerprint.py`): SHA-256 digests of the weights, both
optimizers' moments, every rollout buffer, all three RNG streams and the
printed eval lines, for four cases - the production command line (8 rooms,
impassable, focal-5 CE, MLP readout), the walkable 5-room preset, the
single-room `parity` preset and the serial B=1 `reference` - against a frozen
worktree of the audit commit. Result, every step: IDENTICAL (digests under
`throwaway/2026-09-06/fp/`). The full pytest gate ran at the start and the end.

| gate | before | after |
|---|---|---|
| fingerprint, four cases | baseline captured from `920c8f8` | IDENTICAL after each of steps 1-5; steps 6-7 touched prose only |
| full pytest (`throwaway/2026-09-05/pytest_baseline_cuda.txt`, `throwaway/2026-09-06/pytest_final_cuda.txt`) | 628 passed, 1 deselected | 647 passed, 1 deselected |

## What the cleanup did, by audit item

| audit item | what it was | what happened | commit |
|---|---|---|---|
| C1 circuit (`action_offset` 0 in production, 1 in the figure) | a methods decision | left as is, by the user's decision; still open | - |
| C2 budgets the rollout does not divide | realized steps overshot the stated budget | the config refuses such a budget; the launcher's half budget is a multiple of a rollout | fd75b77 |
| C3 gradient-step axes for a non-training learner | logged from the schedule | zero for a learner that does not train | fd75b77 |
| C4 wandb degrade path killed the run | the fork's `wandb_log` never flipped | one home for "is wandb on", after the init that may fail | fd75b77 |
| C5 `prnn_seqdur` zeroed against its own comment | | reads the episode length | fd75b77 |
| C6, C7 `evaluation/task.py` (downstream OMT) | a `TypeError` and a double-logged key | fixed; the downstream repo still imports what it did | fd75b77 |
| C8 resume with a CUDA graph on | failed after the allocation | refused at parse | fd75b77 |
| C15 multi-room eval left the shell in the last room | | landmarks restored | fd75b77 |
| a2c, the process pool, the intrinsic reference-SR reward, the legacy decoder eval, unused config surfaces, `MinigridEnv`, seven superseded golden fixtures | selected by nothing any run launched | deleted; `check/config_keys.py` lists every removed key under GONE | c8bc5fb, fbe5e57 |
| naming (Hydra-era keys in messages, `prediction_mses` under CE, a `SR_size` hiding a +4, four spellings of the action names, a climbing cache path) | | one name per fact, one home each | 5ed54ee |
| C9, C10, C11 offline tools re-typed the run's config; `Selected.n` shadowed by `positions` | an 8-room run scored as 5 rooms, `action_offset` hard-coded | `Config.from_dict` / `Config.of_run` rebuild a run's EFFECTIVE config from `provenance.json`; every offline tool scores under it; `Selected` is positions-only | fbe5e57 |
| C16 deltas across stream boundaries | | removed with the analysis that computed them | c8bc5fb |
| C17 pixel-to-class in two places | | `palette.classes_of`, one home | fbe5e57 |
| the "legacy" reward alignment | a constructor default no config ever selected; the serial golden pinned it | gone; `rewards.py::REWARD_TARGET_OFFSET` is the one home; goldens re-pinned with each delta measured and recorded in the capture scripts' ledgers | 724bbfa |
| C12 what "pRNN loss" means on focal arms, C13 initial states at B>1, C14 the probe bypassing the adapter | | not changed (C14 optional in the audit; C13 is a design note in memory) | - |
| C18 axes assume full rollouts | | holds by construction once C2 refuses partial ones | fd75b77 |
| fork-side findings | | reported in the audit, not changed (the fork is pinned) | - |
| README, `.env.example`, `docs/invalid-runs.md` claims that had drifted | | corrected | 681b829 |

Retired but kept on purpose: the `multienv` preset (`slurm/train_fast.sh`
launches it) and `storage.save_pN_and_acmodel` (imported by
`../experiment-curiousgeorge`).

## The fresh runs

Run locally on the RTX 4060 (validated on 2026-08-26 to reproduce the L40S
curves at matched env steps) from the EXACT config of the most recent
finished run of each arm, rebuilt from that run's own record
(`throwaway/2026-09-07/rerun.py`), and compared on matched env steps against
the reference's own adjacent-sample band (`curious_george.check.wandb_compare`;
outputs in `throwaway/2026-09-07/`).

| arm | reference | fresh run | matched points inside the band |
|---|---|---|---|
| single L-room, `parity` preset | `parity-s2-mseB_curious_26-08-31-01-32-31` | `clean-sept-parity-s2-mseB_curious_26-09-07-04-51-25` | 51 of 64 |
| 8 rooms, impassable, focal-5 CE, MLP readout | `mx-impassable-n8-s2-focal5mlp_curious_26-08-31-23-18-20` | `clean-sept-mx-impassable-n8-s2-focal5mlp_curious_26-09-07-05-16-40` | 35 of 60 |
| control: the 8-room reference's own sibling seed against it | same reference | `mx-impassable-n8-s3-focal5mlp` | 23 of 48 |

The world-model curves are inside the band at every point on both arms; the
8-room headline (mean room sRSA) ends at 0.796 against 0.791. Every
out-of-band point is a policy-behaviour metric that the reference's own second
seed moves by more. Recorded and not explained: SWdist above the reference at
13 of 14 matched points across the two arms, with the code lineage traced
commit by commit and nothing in it accounting for that; a second fresh seed on
the cluster would settle it. The single-room reference predates two 2026-08-31
correctness fixes, so that pair is not a same-code replication and is not
claimed as one.

## What is left to the user

- Push `sdu/clean-sept` and, if wanted, the cluster seeds (the launchers fetch
  `origin/<branch>` and default to another branch).
- The circuit decision (C1) and the mixed-count design are methods questions,
  untouched.
- Untracked and untouched: `data/` (the observation-bank cache at its old
  location; safe to delete), `throwaway/count_params.py`,
  `throwaway/survey_params.py`, two stale worktrees `git worktree prune` would
  clear.
