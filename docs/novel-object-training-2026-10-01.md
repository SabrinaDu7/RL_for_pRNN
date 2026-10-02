# Training on with a novel object — 2026-10-01

Two designs for the questions repository's Q19 (Q1's continued-exposure paradigm in the
eight-room setup) and its from-scratch counterpart.

## 1. Resume the curious runs with a dot in every room (Q19)

`Selected.extra_anchors` (one extra landmark per room, the one-cell `dot` stencil
registered at import, green, impassable, at Q18's spot per room - the floor cell farthest
from the room's objects) and `resume_clamp.sh`'s `SOURCE_EXTRA` / `SEED` passthroughs.
Each of the three curious runs resumes from its final checkpoint for another 40,960
world-model steps (a second full budget, archives every 8,388,608 frames), the policy and
the world model both training on, graphs and compile off as for every resume:

```bash
export WM_DONE=40960 SOURCE_EXTRA="--env.source.extra-anchors 9,13 13,2 2,10 9,10 9,10 2,9 9,9 2,13"
SEED=<s> sbatch slurm/resume_clamp.sh $SCRATCH/pRNN/<job>/outputs/<run> none dot 40960 sdu/mixed-count-mse \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

- 2026-10-01: seed 2 (from `MSE1024`) → job **11033972**; seed 3 (from `MSE1024S3`) →
  **11033973**; seed 4 (from `MSE1024S4`) → **11033974**; from commit `3e07c18`.
  Run names `mx-impassable-n8-s<seed>-resume-dot`. Smoke-tested locally (one rollout,
  exit 0).

## 2. From scratch with the dot somewhere new in every episode

`Scattered` (`envs/layouts.py`) and `slurm/scattered.sh`: the eight selected rooms, each
repeated `n_placements` times with the dot at a different admissible cell (wall clearance
2, two cells from the room's landmarks, Q18's spot held out), 128 layouts the device pool
draws from per stream at every episode boundary. The spatial evaluations run over all
128 layouts, so their cadence is lowered for this run.

```bash
sbatch slurm/scattered.sh 16 '9,13 13,2 2,10 9,10 9,10 2,9 9,9 2,13' <seed> sdu/mixed-count-mse mse-h1024 \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 20971520
```

Outcomes appended below once the jobs finish.

### 2026-10-02: the position truly random

The 16-placement version shuffles the dot among 16 fixed cells per room, which is not the
design asked for. `Scattered.n_placements = 0` (now the default) takes EVERY admissible
cell in every room: 477 layouts for the eight rooms with Q18's spots held out, the dot's
position random over the whole room at every episode boundary. The spatial evaluation
scores `rooms_max` rooms whatever the layout count, so the cost is the observation banks
at start-up and nothing else.

```bash
sbatch slurm/scattered.sh 0 '9,13 13,2 2,10 9,10 9,10 2,9 9,9 2,13' <seed> sdu/mixed-count-mse mse-h1024 \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

The 16-placement runs (jobs 11034007, 11034008) were left to finish as the coarse version.
