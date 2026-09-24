# Second and third seeds of the MSE1024 recipe — 2026-09-24

## Purpose

The questions repository's 2026-09-24 exploration found, on `MSE1024`
(`mx-impassable-n8-s2-mse-h1024_curious_26-09-09-15-01-18`), that the actor reads the
object-vector cells for a steer-away signal near objects, and that the cells are the world
model's obstacle model. The second finding replicated on MSE2048 and PLUS1024; the first did
not replicate on MSE2048. Two more seeds of the same recipe say whether the actor's readout
comes with the recipe or with the run.

## Launch

The recipe is the original run's provenance with only the seed changed:

```bash
sbatch slurm/multienv.sh true 8 <seed> sdu/mixed-count-mse '' '' '' '' 0,1,2,3,5,6,7,8 mse-h1024 '' \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

- 2026-09-24 11:05: seed 3 → job **10919254** (`mx-impassable-n8-s3-mse-h1024`), seed 4 →
  job **10919255** (`mx-impassable-n8-s4-mse-h1024`); both PENDING on priority at submission,
  from commit `1308758`.
