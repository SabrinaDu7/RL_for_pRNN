# Migration: SabrinaDu7/pRNN → LevensteinLab/pRNN (2026-07-17)

Purpose: retire the fork pin (`SabrinaDu7/pRNN` @ `sdu/prnn-perf-optim`,
383ae24) and make the lab's `LevensteinLab/pRNN` the source of truth, via a
new branch `sdu/rl-integration` there that carries the fork's RL-facing
features. Companion baseline doc: `migration_baseline.md`.

## Decisions (Sabrina, 2026-07-17)

1. Adopt upstream model core (dynamics changes allowed, judged by curve gate
   + full run).
2. Curated re-port (clean commits), not a git merge.
3. Adopt upstream's SI `minmax` fix.
4. Branch pushed directly to LevensteinLab/pRNN.

## What was ported to `sdu/rl-integration` (7 commits, head a2545f5e)

- `pyproject.toml` (uv/hatchling; setup.py kept; numpy<2 from upstream;
  minigrid from SabrinaDu7/minigrid).
- `prnn/utils/enums.py` verbatim; `prnn/utils/checkpoints.py`
  (load_pN/save_pN/CkptKeys) with TWO deltas: `PredictiveNet.pRNNtype` is a
  property (upstream stores ctor args in `trainArgs`), and **load_pN now
  honors the checkpoint's HIDDEN_SIZE** (fork version ignored it — only
  worked for hidden_size=500; pre-existing flaw, fixed).
- `make_minigrid_env` in `prnn/utils/env.py` = the fork's `make_env` renamed
  (upstream's package-dispatch `make_env` is load-bearing for its examples/
  tests/OMT and was left untouched). `HDObsWrapper` came along.
- Curated `prnn/utils/__init__.py` exports (the whole RL import surface);
  `get_env_var`/`set_seed` added to eg_utils.
- Shell: `get_goal_loc`/`get_new_obj_pos` (envs/access.py contract),
  `render_size`, `unwrapped` grid access.
- `PredictiveNet.calculateSpatialMetrics` (pooled SI/sRSA/SWdist, wandb
  logging inside prnn — Sabrina's rule).
- `predict(batched=True)` input-prep fix re-applied (upstream's permute
  produced 3-D and crashed; now 4-D `(1, L, X, B)` — port of fork 383ae24).
- Tests: `test_batched_predict`, `test_spatial_metrics`,
  `test_temporal_alignment` (pins predOffset=0 / same-index masking that
  reward alignment depends on). pRNN suite: 33 → **41 passed, 0 failed**.

## RL repo changes (branch `sdu/prnn-new-migration`)

- Pin swap in pyproject `[tool.uv.sources]` + relock. numpy 2.2.6 → 1.26.4
  (upstream's `numpy<2`).
- Arch detection `"prevAct"/"thcyc" in str(pN.pRNN)` → `pN.pRNNtype`
  (upstream `partial(MaskedRNN, ...)` factories erased class-name markers).
  Sites: adapter.py, rl/collect/agent.py, training/setup.py,
  tests/golden/capture_golden.py. `_StubPN` in test_reward_alignment.py
  gained a `pRNNtype` attr — the ONE test change, required by this approved
  API change (flagged per the never-touch-failing-tests rule).
- Stale permutation-bug comments updated (adapter, scripts/analysis_OMT.py).
- `f=0.5` constructor arg needed NO change: MaskedRNN still accepts `f`
  (the "f removed upstream" exploration claim was wrong for MaskedRNN).
- BatchedSRTracker needed NO change: its `(B,1,H)` state already matches
  upstream's cell contract (confirmed batched==serial 3.6e-7 + tests green).

## Corrections to earlier claims (important)

The exploration-phase claim "upstream thRNN_5win is a different model
(LayerNorm, Xavier, vs fork's plain ReLU + uniform)" was WRONG on two of
three counts: the fork's thRNN_5win ALSO uses LayerNormRNNCell (fork
Architectures.py:1331), and upstream's `xavier_init` is actually the same
uniform U(-1/√k, 1/√k) scheme, renamed. Confirmed empirically:

- `tests/golden_omt/` passes **bitwise** on the new stack (checkpoint-loaded
  forward math identical).
- golden_v0 vs golden_v1 **round 0 is bitwise identical** (same init draws,
  RNG order, encodings, collector).

## The ONE dynamics change: bias_lr

The fork has `bias_lr=int(bias_lr)` (fork predictiveNet.py:238) which turns
the 0.1 default into **0** — all fork-era training ran with frozen pRNN
biases. Upstream (PR #69) uses bias_lr=0.01 with no int(): biases now train.
This is the sole source of divergence between stacks (starts at the first
optimizer step) and is a bug fix, not a regression.

## Gate results

| Gate | Result |
|---|---|
| RL suite (baseline 110/0) | **110 passed, 0 failed, 7 deselected** |
| pRNN_new suite (baseline 33/0) | **41 passed, 0 failed** |
| golden_omt_v0 (bitwise, ckpt-based) | PASS unchanged |
| golden_v1 capture ×2 (determinism) | bitwise identical |
| golden_v0 vs v1 round 0 (invariance) | bitwise identical |
| Perf (benchmark, CPU, 5 updates) | 952 → 914 FPS (0.96x; run-variance range; wm_train 4.60→4.71s) |
| compare_metrics strict per-update gate | FAILs as expected (trajectories diverge from bias training); means comparable (value_loss 0.0694→0.0679, prnn_loss 0.0228→0.0227) |
| Curve gate (multi-seed banded) | **PENDING — Sabrina's full-run launch** |

Artifacts: `tests/perf/results/migration_{old,new}_stack.json`,
`tests/golden/golden_v1.pt` (v0 kept for the legacy stack).

## Follow-ups

1. Sabrina: launch full RL training run; compare vs old-stack wandb band
   (`scripts/analysis_bsweep.py`); expect small shifts from bias training.
2. Re-baseline sRSA/SI/SWdist references on the new stack (SI shifts from
   the minmax fix; expected).
3. Open a PR from `sdu/rl-integration` → LevensteinLab main when ready
   (branch was pushed directly; repo may prefer PR review).
4. The fork's `sdu/prnn-perf-optim` extras NOT ported (deliberately):
   hydra `prnn/config/` dir, justfile/pRNN.sh, LayerNormRNN.py (upstream has
   its own cell), trainNet.py rewrite (upstream's is maintained), fork's
   `test_env.py` (machine-specific path).
5. `benchmark.py` logged-scalar dtype drift vs golden fixtures (float vs
   np.float64, ~1e-7) — pre-existing, see migration_baseline.md.
