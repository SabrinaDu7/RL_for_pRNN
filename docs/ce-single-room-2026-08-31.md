2026-08-31 · branch `sdu/optim-pred` · prnn pin `sdu/ce-loss` @ `a83dede4` · overnight, autonomous

# Phase 1: the CE prediction loss on the bright single L-room

Executing `docs/claude_logs/plan-ce-multienv-overnight-2026-08-31.md`. This doc
is the Phase-1 record: what launched, what came back, what was decided.
STATUS: IN PROGRESS — sections fill as runs land.

## What the CE switch is (implementation, all landed and gated)

One config field: `arch_prnn.loss ∈ {MSE, CE}` (`configs.py::PredLoss`).
- Upstream (`prnn` branch `sdu/ce-loss`, pinned `a83dede4`): `predCE(vocab,
  focal_gamma=None)` in `lossFuns`, a `readout="logits"` architecture kwarg
  (the historical sigmoid head untouched byte-for-byte at the default), and a
  `loss_kwargs` pass-through on `PredictiveNet`. Fork gate: 48 passed
  (7 new in `tests/test_ce_loss.py`).
- Here: `envs/palette.py::TILE_VOCABULARY` — the committed 7-class alphabet
  (floor, wall, agent, blue, green, red, yellow), measured closed over
  1,053,696 bank tiles and re-derived live by `tests/test_palette.py`;
  `storage.py::prediction_loss_kwargs` (the ONE constructor home — MSE
  returns `{}` so the pre-CE call is byte-identical); the adapter's
  `_prediction_errors` (curiosity = per-step SUMMED per-tile surprisal in
  nats under CE, the pinned pixel MSE under MSE) and
  `render_prediction_rows` (argmax class → palette RGB, for by-eye figures).

## Pre-launch measurements (local RTX 4060, `throwaway/`-grade probes)

3 rollouts + updates per arm, parity shape, tiny budget
(`scratchpad ce_probe.log`, reproduction: the probe snippet in the session log):

```
[MSE] cur_reward mean 0.066 -> 0.051   wm loss 0.0635 -> 0.0558 (96 steps)
[CE]  cur_reward mean 99.4  -> 64.9    wm loss 2.0257 -> 1.2678 (96 steps)
      returns ~900-1400, value_mean -0.3 -> 5.2
```

- CE loss starts at ln(7)=1.95 per tile (chance) — the arithmetic sanity.
- Bright-MSE reward (~0.05-0.066) is ~3x the pale era's (~0.016-0.036):
  the 2026-08-30 brightness moved the reward scale as designed.
- 🔴 CE's raw reward scale (~65-99 nats/step, summed over 49 tiles) leaves
  the critic ~200x behind its return target after 3 rollouts. **Decision:
  CE arms run `--train-policy.normalize-reward`** (the Phase-0 flag, built
  for exactly this); one CE arm runs WITHOUT it as the ablation that
  documents the necessity.
- Decision: all Phase-1 arms at `action_offset=0` (the preset default and
  the Phase-2 multienv shape). The historical anchor
  `mila-off1-e0.003-s2_curious_26-08-29-21-09-36` is offset 1, raw
  advantages, pale era — a BALLPARK reference only (its own values:
  sRSA 0.64 last-2 [series 0.34→0.68→0.60 — unseeded-probe noise, which is
  why `probe_seed` now exists], SWdist 0.049, mean SI 0.90, MI 0.042).

## Arms (launched via `slurm/parity.sh`, project `curious-george`)

| arm | label | command tail | wandb run |
|---|---|---|---|
| A bright-MSE control | mseB | *(pure preset: whitened, e=0.035, probe_seed on)* | *(fills on launch)* |
| C bright-CE | ceRN | `--arch-prnn.loss CE --train-policy.normalize-reward` | |
| C-e024 | ceRN (e=0.024) | same + entropy arg 0.024 | |
| C-e07 | ceRN (e=0.07) | same + entropy arg 0.07 | |
| C-raw ablation | ceRaw | `--arch-prnn.loss CE` (no reward norm) | |
| D CE + random fwd | ceRN-rndfwd | C's flags + `--arch-policy.agent RANDOM` | |
| E MSE + random fwd | mse-rndfwd | `--arch-policy.agent RANDOM` | |

Reading order per the plan: A vs anchor (era ballpark), D vs E (loss effect
at fixed data — the most diagnostic pair), C vs A and C vs D.

## Results

*(pending — runs land ~30 min after launch)*

## Deviations from the plan

- Golden fixtures did NOT need a v6 recapture for the defaults flip: the flip
  landed at PRESET level, dataclass defaults untouched, so the train/eval
  fixtures stay bitwise; only the setup-composition fixture (its own
  in-place convention) was recaptured. Recorded in `docs/invalid-runs.md`.
- The fork needed one unplanned fix: `pRNN.__init__` forwarded `output_size`
  to `create_layers` both positionally and inside `**cell_kwargs` — a
  TypeError the first time the documented override was actually used.
  Filtered (not popped — `trainArgs` holds the dict by reference).
