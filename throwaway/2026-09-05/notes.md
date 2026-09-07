# Audit scratch notes (2026-09-05) — raw observations, to be verified before the report

## Wiring trace (device path, action_offset=1) — CONFIRMED by reading collector.py / prnn_adapter.py
- policy at t: acmodel(obs[t].direction, SR=state.sr) with state.sr = h[t] (tracker.step_device at t-1 used post-step obs; reset_all_envs bootstraps h[0] from post-reset obs with action -1)
- SRs[t]=h[t], actions[t]=a[t], values[t], masks[t]=pre-step mask
- env step banks obs[t], HD[t], pos[t]; post=obs[t+1]
- SR step offset1: tracker.step_device(a[t], obs[t+1], HD[t+1]) -> h[t+1] row t+1=(obs[t+1], fwd a[t], HD[t+1])  ✓ figure
- training rows (train_on_episodes_batched): L+1 rows, row0=(obs0, no act, HD0), row t=(obs t, fwd a[t-1], HD t) ✓; target obs[t] (predOffset 0 — VERIFY in fork)
- curiosity device: same L+1 rows, errors[:,1:] -> column i = row i+1 error -> reward for a[i] ✓ ; flat order b*T+t ✓
- GAE: masks[t+1]=0 after segment close; final_mask=0 ✓ no cross-segment bootstrap
- PPO log_prob from stored normalized logits ✓ bit-exact w/ Categorical.log_prob
- update: _index_policy_batch gives SR + direction ✓ same inputs as rollout

## 🔴 CHECK: production preset action_offset
- ArchPrnnCfg.action_offset default 0; _parity/_multienv_fast don't set it; multienv.sh passes no flag
- => production likely runs offset 0 (policy acts on h[t-1]) while circuit-desired.png shows offset 1. GREP docs/slurm.

## Naming / two-homes defaults
- PredictivePPOAlgo.__init__ action_offset=1 default vs ArchPrnnCfg 0 vs ActorCriticAgent 0 vs RolloutConfig 0
- reward_alignment="legacy" default in algo/RolloutConfig/compute_curious_rewards vs config NEXT_OBS; rewards.py docstring says legacy is "default"
- stale hydra keys in code/errors: exp.rollout_cuda_graph, exp.device_env, rl.cuda_graph, rl.loss, predNet.* (algo, collector, rollout_graph, world_model, losses, prnn_adapter, models/device)
- algo docstring "world_model.adapter"; _fingerprint docstring "world_model.device._move()" -> models/
- tests/golden_omt references (advantage.py, models/policy.py x2, collector) -> moved to questions repo
- RolloutConfig docstring RANDOM_ACTION_PROBS vs configs.RAND_ACT_PROBA
- randomAgent_collect_exp_and_update called "retired" (collector.py:110, loop.py:242) but exists and is called by evaluation/task.py:171; bypasses adapter (pN.trainStep direct)
- ACModelSR.SR_size includes +4 HD; SR_single is real SR size — misleading
- prediction_mses under CE = surprisal (noted historical)
- _GraphWMTrainer docstring "MSE loss" but code uses pN.loss_fn (CE-capable)

## Dead / redundancy candidates
- IntrinsicReference + intrinsic/k_intrinsic + int_rewards plumbing (B=1 only, off)
- compare_trajs (exported, no callers)
- a2c_loss + LOSSES/loss_fn string indirection (no config field `loss`? VERIFY setup.py)
- wm_segment_stride (no TrainPrnnCfg field? VERIFY setup.py) + long docstring in world_model.py
- algo.noise_mu/noise_std/lr/k_int/cuda_graph stored only for OnPolicyAnalysis rebuild
- theta (thcyc) branches in adapter/policy (prnn_type fixed to masked)
- non-SpeedHD env2pred fallbacks (action_encoding configurable? which are used?)
- ACModel (non-SR) if unused; ACModelSR SR.ndim>2 branch
- ActorCriticAgent.next_SR duplicates adapter.next_sr (non-fast path) and bypasses adapter; starts SR=zeros regardless of offset (offset1 mismatch at row 0)
- check_large_jump DEBUG prints (serial/async paths)
- AsyncShellPool: any preset/launcher uses it? VERIFY
- SpatialEvalPath.LEGACY_DECODER + legacy_decoder_timesteps (user: ask if used)

## Optimization notes
- _GraphWMTrainer.train_batch: .item() sync per WM step (32/rollout)
- wasted tracker step at last step of every segment (h[L] discarded) — negligible
- k_int * zeros each GAE — negligible

## Correctness to verify in fork
- predOffset=0 in MaskedRNN; clip_mask minsize; initial hidden state in predict/trainStep (noise vs zeros) vs BatchedSRTracker zeros
- trainStep extras vs _GraphWMTrainer region (bptt trunc? grad clip? homeostat?)
- predCE focal formula; targets_for; render
- phase alignment: predict starts phase 0 per segment?
