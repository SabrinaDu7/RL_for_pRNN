2026-08-31 · branch `sdu/optim-pred` · prnn pin `sdu/ce-loss` @ `600c7e09` · overnight, autonomous

*(Audit corrections, 2026-08-31 evening: the pin above originally read
`a83dede4` - the runs actually installed `600c7e09`, whose only delta is the
argmax RENDER path, no dynamics. The repo has since re-pinned `1939006e`
(audit refactors: `targets_for` as the one pixel-to-class home, `readout`
popped in `create_layers` - construction and math unchanged, goldens gate
it). These runs also pre-date the seeded-probe RNG fix (docs/invalid-runs.md
top entry): cross-run orderings stand, seed-variance is understated.)*

# Phase 1: the CE prediction loss on the bright single L-room

Executing `docs/claude_logs/plan-ce-multienv-overnight-2026-08-31.md`. This doc
is the Phase-1 record: what launched, what came back, what was decided.
STATUS: IN PROGRESS — sections fill as runs land.

## What the CE switch is (implementation, all landed and gated)

One config field: `arch_prnn.loss ∈ {MSE, CE}` (`configs.py::PredLoss`).
- Upstream (`prnn` branch `sdu/ce-loss`, pinned `600c7e09`): `predCE(vocab,
  focal_gamma=None)` in `lossFuns`, a `readout="logits"` architecture kwarg
  (the historical sigmoid head untouched byte-for-byte at the default), and a
  `loss_kwargs` pass-through on `PredictiveNet`. Fork gate: 48 passed
  (7 new in the fork's test_ce_loss.py).
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

## Arms (relaunched 2026-08-31 ~02:00 via `slurm/parity.sh`, project `curious-george`)

Repo commit `b1763ae` (branch `sdu/optim-pred`), all arms
`sbatch slurm/parity.sh '' [ent] 2 sdu/optim-pred '' <label> <extra...>`:

| arm | label / run-name stem | slurm job | command tail |
|---|---|---|---|
| A bright-MSE control | parity-s2-mseB | 10576559 | *(pure preset: whitened, e=0.035, probe_seed on)* |
| C bright-CE | parity-s2-ceRN | 10576560 | `--arch-prnn.loss CE --train-policy.normalize-reward` |
| C-raw ablation | parity-s2-ceRaw | 10576561 | `--arch-prnn.loss CE` (no reward norm) |
| D CE + random fwd | parity-s2-ceRN-rndfwd | 10576562 | C's flags + `--arch-policy.agent RANDOM` |
| E MSE + random fwd | parity-s2-mse-rndfwd | 10576563 | `--arch-policy.agent RANDOM` |
| C-e024 | parity-e0.024-s2-ceRN | 10576564 | C's flags, entropy 0.024 |
| C-e07 | parity-e0.07-s2-ceRN | 10576565 | C's flags, entropy 0.07 |

Reading order per the plan: A vs anchor (era ballpark), D vs E (loss effect
at fixed data — the most diagnostic pair), C vs A and C vs D.

## Results (all 7 arms COMPLETED 0:0, 28:52-29:59 elapsed; read ~02:45)

Final summary values, seeded probe (10007), `scratchpad/phase1_readout.py`:

| arm | wm loss | sRSA | SWdist | SI | MI | entropy | locH | coverage |
|---|---|---|---|---|---|---|---|---|
| A  mse-bright | 0.0109 | 0.652 | 0.106 | 0.996 | 0.075 | 1.790 | 7.12 | 0.252 |
| C  ce+rnorm e.035 | 0.159 | **0.871** | **0.087** | 1.036 | **0.358** | 0.956 | 5.13 | 0.142 |
| C  ce+rnorm e.024 | 0.192 | 0.782 | 0.217 | 1.277 | 0.449 | 0.937 | 5.74 | 0.159 |
| C  ce+rnorm e.07 | 0.205 | 0.814 | 0.107 | 1.138 | 0.334 | 0.980 | 4.76 | 0.118 |
| C' ce raw-reward | 0.192 | 0.808 | 0.138 | 1.204 | 0.366 | 1.133 | 6.58 | 0.151 |
| D  ce random-fwd | 0.063 | 0.682 | 0.135 | 1.031 | — | 1.832 | 7.12 | 0.391 |
| E  mse random-fwd | 0.0078 | 0.700 | 0.214 | 1.159 | — | 1.887 | 7.12 | 0.391 |

(wm loss units differ by design: pixel MSE vs nats/tile - never compare
across the loss column. Anchor, ballpark only: sRSA 0.64, SWdist 0.049,
SI 0.90, MI 0.042.)

**Verdict: the Phase-1 gate PASSES, and the plan's success criterion is met
in Phase 1 already.** The headline CE arm (e0.035) posts sRSA 0.871 with a
COMMITTED policy (MI 0.358, entropy 0.96 bits) - above its own random
baseline (D, 0.682), above the bright-MSE control (A, 0.652), above the
anchor. **The MI-sRSA anticorrelation is broken**: under MSE the curious arm
still sits BELOW its random baseline (0.652 vs 0.700 - the pale-era
pathology, reproduced in the bright era); under CE it sits far above.

Supporting readings:
- **D vs E** (loss effect at fixed data): 0.682 vs 0.700 - the loss swap
  alone does not degrade the spatial code. The user's stated risk, retired.
- **Both random arms hit coverage 0.391** vs the calibration's 0.395 -
  the exploration wiring sanity rule passes.
- **Reward normalization earns its default-for-CE**: ceRaw is worse on sRSA
  (0.808) and SWdist (0.138) at otherwise identical settings.
- **Entropy pick for Phase 2: 0.035** (best sRSA and SWdist of the scan).
- SWdist is elevated across ALL bright-era arms relative to the pale anchor
  (0.09-0.22 vs 0.049); within-era, C-e.035 is the lowest. Carried per the
  user's instruction: both bright-MSE AND pale-MSE stay as controls.
- **By eye** (final in-run Observation Sequence figures, saved to
  `docs/figures/phase1_{ceRN,mseB}_observation_vs_prediction.png`, also
  in each run's wandb media): the CE run's argmax predictions match the
  observation nearly tile-for-tile (landmark shapes, wall band, the blue
  triangle corner); the MSE run's predictions are the familiar blur - a
  muddy blue-yellow gradient with barely a hint of structure. Same
  architecture, same room, same budget; only the loss differs.
- ⚠️ For the morning: the CE policies CONCENTRATE (locH ~4.8-5.7,
  coverage ~0.12-0.16 vs random's 0.39). The representation is better while
  the coverage is lower - where the policy actually goes (occupancy maps,
  trajectory stats) is the next behavioural question, not a blocker.

## Deviations from the plan

- 🔴 **The first launch wave (jobs 10576255-61, 10576276-77, ~01:35) was
  mislaunched and cancelled.** `sbatch` reads the launch script from the
  shared cluster checkout's WORKING TREE at submission time, and that tree
  was still on `sdu/multienv` @ `b583405` — a `parity.sh` without label or
  EXTRA support, so every label and every extra flag (including
  `--arch-prnn.loss CE`) was silently dropped: the "CE" jobs were running
  plain bright-MSE under junk names, and the mx8 wave ignored both the
  positions argument (room 83 back in) and the project override. The
  RUNBOOK's preflight step 3/4 names exactly this trap; the preflight here
  checked the ORIGIN ref and not the working tree. Fixed by advancing the
  checkout to `origin/sdu/optim-pred` and resubmitting; the six junk wandb
  runs are tagged `mislaunched-drop` with an explanatory note. Job 10576255
  additionally died at 8m11s (FAILED 1:0) — not diagnosed further; it was
  junk either way.

- Golden fixtures did NOT need a v6 recapture for the defaults flip: the flip
  landed at PRESET level, dataclass defaults untouched, so the train/eval
  fixtures stay bitwise; only the setup-composition fixture (its own
  in-place convention) was recaptured. Recorded in `docs/invalid-runs.md`.
- The fork needed one unplanned fix: `pRNN.__init__` forwarded `output_size`
  to `create_layers` both positionally and inside `**cell_kwargs` — a
  TypeError the first time the documented override was actually used.
  Filtered (not popped — `trainArgs` holds the dict by reference).
