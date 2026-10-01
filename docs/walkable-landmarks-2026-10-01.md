# The MSE1024 recipe with walkable landmarks — 2026-10-01

## Purpose

The questions repository's Q15 and Q16 found the object-vector cells to be the world
model's obstacle code (recruited as the error at object bumps falls; clamping them raises
that error alone; present under a random walker; allocated by the object-to-wall balance
of what stops the agent). The user's necessity test: the same rooms with the landmarks
walkable — no bump to predict — should grow none (Q17).

## Launch

`multienv.sh`'s walkable arm, otherwise the MSE1024 recipe. `Selected` pins the anchors
and applies the affordance afterwards, so the rooms are the impassable runs' rooms with
the landmarks painted as `Floor`:

```bash
sbatch slurm/multienv.sh false 8 <seed> sdu/mixed-count-mse '' '' '' '' 0,1,2,3,5,6,7,8 mse-h1024 '' \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

- 2026-10-01: seed 2 → job **11028232**, seed 3 → job **11028233**, from commit `0d76dc9`.
  Both started within a minute (cn-l003, cn-l047).
- Outcome: both **COMPLETED**, exit 0, ~60 min each — the walkable arm ran at ~32k env
  steps/s against the impassable arm's ~95k on the same node class, so a walkable run needs
  the 1:30 limit's second half. Run directories
  `mx-walkable-n8-s2-mse-h1024_curious_26-10-01-15-58-44` and
  `mx-walkable-n8-s3-mse-h1024_curious_26-10-01-15-58-44`, fetched to `outputs/fetched/`
  (wandb excluded); registered as `WALK1024` and `WALK1024S3` in the questions repository.
- Q17's first numbers (final checkpoint, Q4's detector): 36 and 45 object-vector cells —
  the same as the random walker's impassable runs (45, 56), a third of the curious
  impassable runs (116 / 99 / 83) — with the same adjacent tuning; interior place cells 169
  and 224 against ~50 in every impassable run; border place cells 220 and 226. The cells
  are not an obstacle code only; obstruction (with the curious policy) triples them.
