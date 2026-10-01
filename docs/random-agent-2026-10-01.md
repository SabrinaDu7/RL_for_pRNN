# The MSE1024 recipe under a random walker — 2026-10-01

## Purpose

The questions repository's Q15 (the time course over MSE1024's archived checkpoints) found
the object-vector cells to be the world model's obstacle code first — absent at 8.4M
frames, recruited from 25M to the end — and the actor's steer-away and brake a late, weak
readout. The curious agent presses into objects far more than a random walker would, so
the open question is whether the cells are the loss's or the behaviour's: train the same
world model on a random walker's experience and ask whether they emerge, in what number,
with what tuning, and with the same obstacle role (the questions repository's Q16).

## Launch

The MSE1024 recipe (`mx-impassable-n8-s2-mse-h1024_curious_26-09-09-15-01-18`'s
provenance) with `multienv.sh`'s `agent` argument set to `random`
(`--arch-policy.agent RANDOM`: actions from `RAND_ACT_PROBA`, no policy updates; the
policy flags are inert):

```bash
sbatch slurm/multienv.sh true 8 <seed> sdu/mixed-count-mse '' random '' '' 0,1,2,3,5,6,7,8 mse-h1024 '' \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

Run names `mx-impassable-n8-s<seed>-random-mse-h1024`, wandb project
`curious-george-multienv`.

- 2026-10-01: seed 2 → job **11013546**, seed 3 → job **11013547**, both PENDING at
  submission, from commit `0f00caf` (the launcher on the cluster checkout matches it).
- Smoke-tested locally at a one-rollout budget before submission; outcome appended below.

Outcomes appended below once the jobs finish.

- Smoke test: the recipe with `--arch-policy.agent RANDOM` at a one-rollout budget ran to
  exit 0 locally (the random agent's policy archive carries no weights, as `loop.py`
  says). Both jobs started within a minute of submission (cn-l021, cn-l062), at ~95k env
  steps/s, and archived their first checkpoint pair at 8,388,608 frames after 8 min. Run
  directories: `mx-impassable-n8-s2-random-mse-h1024_random_26-10-01-01-42-43` and
  `mx-impassable-n8-s3-random-mse-h1024_random_26-10-01-01-42-43`.
