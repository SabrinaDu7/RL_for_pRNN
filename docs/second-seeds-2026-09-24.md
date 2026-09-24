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

## The same day: ablate-and-retrain and bump-penalty arms (commit `274f119`)

Library changes: `arch_prnn.clamp_units` (a forward hook on the pRNN cell holding named
units at fixed values, `models/unit_clamp.py`) and `train_policy.bump_penalty` (a reward
of −penalty on every forward that did not move, `envs/vector.py::blocked_forward_rewards`);
launcher `slurm/resume_clamp.sh`. Clamp sets in `data/clamps/` (the 116 object-vector
cells of MSE1024 at their on-policy means; the questions repository's Q13 first random
draw of 116 non-object units at theirs), copied to `$SCRATCH/pRNN/clamps/`.

Resume arms, from the MSE1024 checkpoint on the cluster
(`$SCRATCH/pRNN/multienv_10724967/outputs/mx-impassable-n8-s2-mse-h1024_curious_26-09-09-15-01-18`,
policy.pt at 83,886,080 frames = 40,960 world-model steps), +8,192 world-model steps
(+32,768 policy steps), graphs and layer compile off, the recipe's preset flags repeated:

```bash
sbatch slurm/resume_clamp.sh $RUN <clamp|none> <label> 8192 sdu/mixed-count-mse \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

- job **10919484** `mx-impassable-n8-s2-resume-ovcclamp` (object-vector cells at their means)
- job **10919485** `mx-impassable-n8-s2-resume-rand0clamp` (116 random non-object units)
- job **10919486** `mx-impassable-n8-s2-resume-noclamp` (the resume-only control)

Bump-penalty runs, the MSE1024 recipe from scratch (seed 2) with
`--train-policy.bump-penalty P` appended to `multienv.sh`'s extra flags (labels
`mse-h1024-bump0.002`, `mse-h1024-bump0.01`; the intact run's mean per-step curiosity
reward is ~0.007 pixel-MSE, so these are ~0.3× and ~1.4× of it before normalisation):

- job **10919487** `mx-impassable-n8-s2-mse-h1024-bump0.002`
- job **10919488** `mx-impassable-n8-s2-mse-h1024-bump0.01`

Smoke-tested locally before submission (resume + clamp + penalty, one rollout, exit 0).
Outcomes appended below once the jobs finish.
