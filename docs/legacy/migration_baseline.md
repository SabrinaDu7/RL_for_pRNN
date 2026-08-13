# Migration baseline (old stack) — captured 2026-07-17

Baseline for the SabrinaDu7/pRNN → LevensteinLab/pRNN migration
(plan: prnn-new migration; companion doc: migration_prnn_new.md, written at
the end). All numbers below were RUN, not recalled.

## Environment

- RL repo: main @ `cc4dcaf` (clean tree)
- prnn: SabrinaDu7/pRNN branch `sdu/prnn-perf-optim` @ `383ae24`
  (pin: pyproject `[tool.uv.sources]` + uv.lock)
- torch 2.8.0+cu128, CPU runs (`CUDA_VISIBLE_DEVICES=""`)

## Test suite

`uv run pytest`: **110 passed, 0 failed, 7 deselected** (17.7s).

## Golden gate (tests/golden/golden_v0.pt)

Regenerated via `capture_golden.py` (seed 2, frames 64, seqdur 32, 2 updates,
CPU) into scratchpad and compared bitwise:

- All tensors bitwise identical: curious_rewards, rewards, advantages,
  values, actions, log_probs, SRs, locs, acmodel_state, prnn_state.
- KNOWN BENIGN DIFF: `rounds[*].{policy_loss,value_loss,grad_norm}` scalars —
  fixture stored `np.float64`, current code logs Python `float` from a
  float32 tensor; values agree to ~1e-7 (float32 eps). A logging-dtype drift
  vs the fixture, not a numeric regression (all weights bitwise-equal proves
  the math is unchanged). Pre-existing; flagged, not fixed here.

`tests/golden_omt/` runs inside pytest (green, included in the 110).

## Perf / metric harness

`CUDA_VISIBLE_DEVICES="" uv run python tests/perf/benchmark.py --updates 5
--out tests/perf/results/migration_old_stack.json`

- **fps = 952.3** (CPU, main.yaml defaults: exp.num_envs=8). Full stage
  timings + learning metrics in `tests/perf/results/migration_old_stack.json`.
- Note: docs/perf_log.md cites ~1197 FPS for this config on an earlier run;
  today's same-machine number is the one the post-migration comparison uses.

## Curve-gate reference

The banded multi-seed reference is the existing wandb `bsweep-B*`/`wmsweep`
runs consumed by `scripts/legacy/analysis_bsweep.py` (old-stack dynamics).
Post-migration short runs will be compared against that band; the full-run
verdict is Sabrina's training launch. NOTE (by design, decisions 2026-07-17):
the new stack adopts upstream's model core (LayerNorm cell, Xavier init,
bias_lr 0.01), so curves are expected to shift; the gate is "learns
comparably", not "identical".

## Upstream/fork gap facts confirmed while baselining

- Upstream pRNN_new @ `316ac4cf` ALREADY has batched `predict`/`trainStep`
  and `predict_single`/`reset_state`/`phase_k`/`trainNoiseMeanStd`/
  `numTrainingEpochs`/`generate_noise` (predictiveNet.py:240-303) and
  `MaskedRNN(predOffset=0)` (Architectures.py:733) — the temporal-alignment
  contract holds structurally upstream.
- SUSPECT: upstream batched predict input prep (predictiveNet.py:253-259)
  permutes without the fork's 4-D `.unsqueeze(0)` fix (fork 383ae24) — to be
  settled by an equivalence test on the new branch.
- Missing upstream (to port): prnn/utils/enums.py, checkpoints.py
  (load_pN/save_pN/CkptKeys), PredictiveNet.calculateSpatialMetrics,
  populated prnn/utils/__init__.py exports, fork's make_env signature,
  pyproject.toml packaging.
