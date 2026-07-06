# Refactor baseline (Phase 0)

Captured 2026-07-03 before any refactor changes. Plan: `curious_george/` refactor
(see approved plan; behavior-preserving, legacy paths deleted, reward-alignment
fix deferred behind a future `reward_alignment` flag).

> **Status vs this baseline (2026-07-05, end of modularity pass):**
> suite is at **187 passed / 16 failed / 7 deselected** (baseline 169/18;
> +10 new alignment/device tests, +3 batched-tracker, +3 batched-collector,
> 2 pre-existing ckpt-test failures fixed; the 16 remaining failures are the
> untouched pre-existing test_wandb_data (12) and test_analysis_omt (4)).
> The golden fixture below has held BITWISE through every stage: package
> extraction, algo split into adapter/rewards/buffer/ppo, collector
> unification (B=1 through the batched code path), training/ split
> (old script vs main_train A/B: identical checkpoints), hydra config
> groups, and the utils move. Flaw #1 below is now implemented behind
> `rl.reward_alignment` (`rewards=curious_next_obs`); #2-#6 are fixed or
> quarantined as documented in docs/refactor_notes.md and
> docs/refactor_progress.md.

## Ground

- Base commit: `807dab9748331af97aac5edfc86099d5eed570cc` on branch
  `sdu/refactor` (clean working tree at capture time).
- Python 3.10.15, torch per `uv.lock`, all runs CPU-forced for determinism.

## Test suite baseline

`uv run pytest -q --ignore=tests/test_figure3_sRSA.py`
→ **169 passed, 18 failed, 7 deselected (slow)** in ~10s.

- `tests/test_figure3_sRSA.py` cannot be collected at all: it imports
  `scripts.figure3_sRSA`, whose `.py` source is missing (only a stale `.pyc`
  exists in `scripts/__pycache__`). **Pre-existing breakage**, ignored to get
  counts for the rest.
- Pre-existing failures (all present before refactor):
  - `tests/test_wandb_data.py` — 12 failures
  - `tests/test_analysis_omt.py` — 4 failures (`TestEvalTrajectoryConfig::test_defaults`,
    `TestEvalModeContextManager::test_restores_actor_critic`,
    `TestEvalModeContextManager::test_restores_on_exception`,
    `TestSaveLoadRoundtrip::test_roundtrip`)
  - `tests/test_ckpts.py` — 2 failures (`test_load_acmodel_from_checkpoint[AC_agent]`,
    `test_load_actor_critic_agent[AC_agent]`, RuntimeError)

Refactor gate: passed count must stay ≥ 169 and no NEW failures beyond these 18.

## Golden fixture (behavior-preservation oracle)

- Generator: `tests/golden/capture_golden.py` → `tests/golden/golden_v0.pt`.
  (The generator's imports track the current package layout; the .pt file is
  frozen pre-refactor truth and is never regenerated as the oracle.)
- Mirrors the historical mainline construction (MiniGrid-LRoom-v0, `pRNN`
  input, `SpeedHD`, fresh `thRNN_5win` hidden=500, ACModelSR, PredictivePPOAlgo,
  `curious_agent=True`, `intrinsic=False`, `train_pN=True`), CPU, seed=2,
  frames=64, seqdur=32, 2 collect+update rounds.
- Saves per round: `curious_rewards`, `rewards`, `advantages`, `values`,
  `actions`, `log_probs`, `SRs`, `locs`, PPO losses/grad_norm; plus final
  acmodel and pRNN state_dicts.
- **Confirmed bitwise deterministic**: two independent runs compared
  `torch.equal` on every tensor incl. post-update weights → identical.
  Round-0 signature: `curious_rewards.mean()=3.875239e-02`,
  `advantages.mean()=4.402453e-01`.

Refactor gate (Phases 1–4, and Phase 5 at B=1): regenerating via the same
entry must reproduce `golden_v0.pt` exactly.

## Spatial metrics baseline (real checkpoint)

Checkpoint: `$PRNN_CUR_CKPT` from `.env` (curious-agent pRNN), `thRNN_5win`,
seed=2, random-action agent (off-policy eval, action probs [.15,.15,.6,.1]), CPU.

- **sRSA = 0.4391** via `calculateSpatialRepresentation(..., calculatesRSA=True,
  sleepstd=0.03)`; runtime 69s (dominated by decoder training, 5000 batches).
- **SWdist = 0.0550** computed directly: 5000-step wake sequence → `predict` →
  theta-mean h; `spontaneous(500, 0, noisestd=0.03)` sleep h;
  `RGA.calculateSleepWakeDist(wake, sleep, metric="cosine")[0]`
  (the scalar is the FIRST return, `SleepSimilarity` = median nearest-wake
  cosine distance; `predictiveNet.py:951` unpacks it under the name SWdist).
  Runtime 7.3s. Note: this is a close approximation of the in-library
  computation (which uses the wake activity from `calculateSpatialRepresentation`
  after transient trimming); methodology is what `evaluation/spatial.py` will
  standardize in Phase 3.

These are stochastic (agent RNG) — treat as ballpark references, not exact
gates; the exact gate is the golden fixture.

## Known pre-existing flaws recorded (not fixed; see plan findings)

1. Curiosity reward credits `a_t` with prediction error on `obs_t` (pre-action),
   not `obs_{t+1}` — `predOffset=0` on all `*_5win` nets. Fix deferred behind
   `reward_alignment` flag (default `legacy`).
2. `int_rewards[0]` always 0 (duplicated first error, `algo.py:367`).
3. `ActorCriticAgent.getObservations` forces pRNN to CPU (`agent.py:161`).
4. `pN.predict(batched=True)` permutation bug (workaround: direct `pN.pRNN` call).
5. `predictiveNet.pRNN.k` missing on `thRNN_5win*` (theta-only attribute).
6. Import-time env-var read in `RLutils/storage.py:24`.
