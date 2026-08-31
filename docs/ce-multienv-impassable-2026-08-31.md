2026-08-31 · branch `sdu/optim-pred` · prnn pin `sdu/ce-loss` @ `600c7e09` · overnight, autonomous

# Phase 2: multienv training on 8 impassable-object rooms, MSE vs CE

Executing `docs/claude_logs/plan-ce-multienv-overnight-2026-08-31.md` Phase 2.
Project: **`curious-george-multienv`** (new; every run in it is bright-era,
whitened, seeded-probe). STATUS: IN PROGRESS.

## The room set

`ROOMS_SELECTED` positions (0, 1, 2, 3, 5, 6, 7, 8) = source-pool indices
0, 14, 31, 35, 126, 144, 169, 191 — source index 83 dropped (one cell from
index 0, the recorded near-duplicate; user call 2026-08-31), 191 promoted
from the spares. Impassable affordance; anchors identical to the walkable
twin by construction (`Selected`). Standable cells per room: 152.

## Exploration calibration at the launch tree (regenerated, per the handoff rule)

`uv run python -m curious_george.envs.action_graph` at `5cf2e65`:

```
=== impassable, denominator 152 ===
agent                     cov@256   nAUC     T25          T50          T90
uniform                     0.182  0.104   228 ( 11%)   censored     censored
forward-weighted            0.360  0.206   152 ( 90%)   234 (  4%)   censored
greedy sweeper              1.000  0.611    42 (100%)    92 (100%)   198 (100%)
```

Sanity rule: a forward-weighted RANDOM arm's online `exploration/coverage`
must sit near 0.36 from its first log points, else the wiring is wrong.

## Arms

Launcher: `slurm/multienv.sh true 8 <seed> sdu/optim-pred 21968 [agent] '' ''
0,1,2,3,5,6,7,8 <label> <extra...>` — half budget (~30 min L40S), whitened
preset defaults, `probe_seed` on.

| wave | arm | label / run stem | slurm job | extra flags |
|---|---|---|---|---|
| 2a | MSE curious s2 | mx-impassable-n8-s2-wm21968-mse8 | 10576566 | `--run.wandb-project curious-george-multienv` |
| 2a | MSE curious s3 | mx-impassable-n8-s3-wm21968-mse8 | 10576567 | same |
| 2b | CE curious | *(pending Phase-1 gate + entropy pick)* | | `+ --arch-prnn.loss CE --train-policy.normalize-reward` |
| 2b | CE random fwd | | | 2b's flags, agent arg `random` |
| 2b | CE random uniform | | | + `--arch-policy.random-action-probs 0.25 0.25 0.25 0.25` |
| 2b | CE count-based | | | `--train-policy.no-curious --train-policy.k-count 0.1` + CE flags |

🔴 Ordering honoured: wave 2a (bright-MSE) submitted 2026-08-31 ~02:00 (first wave ~01:50 mislaunched and cancelled - see the Phase-1 doc's deviations),
BEFORE any CE run exists in the project. Wave 2b launches after the Phase-1
gate (`docs/ce-single-room-2026-08-31.md`) reads out.

## Results

*(pending)*

## Reading list when runs land

- Per-room sRSA (seeded probe), pooled sRSA, remapping index, SWdist, SI —
  CE vs MSE within-project.
- `exploration/*` vs the calibration rows above and vs the three behavioral
  baselines.
- Prediction figures by eye (argmax renders under CE).
- Bump events: prediction error at table-refused forward poses vs free moves
  (offline; the impassable-specific affordance-learning probe).
- Trajectory stats/plots per arm: wall-hugging, spinning, object-edge
  shuffling are the named pathologies to look for.
