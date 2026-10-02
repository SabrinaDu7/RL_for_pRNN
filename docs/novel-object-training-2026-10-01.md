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

## Outcomes (2026-10-02 02:10)

- **Resumes** (11033972–11033974): a resume runs with the CUDA graphs and the layer
  compile off and trains at ~15k env steps/s against a fresh run's ~100k, so the three hit
  `resume_clamp.sh`'s 90-minute limit at ~96% of the second budget (TIMEOUT); the exit
  trap saved 9 of the 10 archived checkpoint pairs each (92.3M to 159.4M frames). Fetched
  to `outputs/fetched/mx-impassable-n8-s{2,3,4}-resume-dot_curious_26-10-01-23-45-2*`.
  Q19: the dot's squares end at 1.76× their dot-free visits (seeds 1.66–1.92; first
  encounter 1.06), the late distance to the dot lower with it than without in every
  seed; the agent's near-dot time itself stays at uniform while its dot-free visits to
  the same squares halve — the dot retains the agent rather than draws it.
- **Scattered, 16 placements** (11034007, 11034008): COMPLETED in 25:47 / 25:xx. **Scattered,
  every cell** (11034398, 11034399; 477 layouts): COMPLETED in ~25 min each, the spatial
  evaluation scoring `rooms_max` rooms. Q19_exp3: no pull toward a dot at the held-out
  spot (ratios 1.28 / 1.08 for the all-cell runs, distance differences −0.05 / +0.14).

## 3. Swap the green plus for a yellow X (2026-10-02)

The novel-object-recognition design proper: a familiar object taken out and a new one put
in its place. `Selected.swap_landmark` replaces landmark 1 of every committed room (the
green plus, in all eight) with `swap_shape` in `swap_color` - an `x` in `yellow`, both
already in the fork's stencil table and the landmark palette - at the same anchor and the
same affordance; the X's five cells sit on the floor and clear of the other two objects
in all eight rooms (checked by `resolve_rooms`, pinned by
`tests/test_selected_rooms.py::test_swap_landmark_replaces_the_plus_with_an_x_at_the_same_anchor`).
The three curious runs resume from their final checkpoints as in section 1, the same
second budget; the walltime is raised to two hours on the command line so the resumes
(no graphs, no compile, ~6x slower than a fresh run) reach the tenth archive this time.

```bash
export WM_DONE=40960 SOURCE_EXTRA="--env.source.swap-landmark 1 --env.source.swap-shape x --env.source.swap-color yellow"
SEED=<s> sbatch --time=02:00:00 slurm/resume_clamp.sh $SCRATCH/pRNN/<job>/outputs/<run> none swapx 40960 sdu/mixed-count-mse \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

Run names `mx-impassable-n8-s<seed>-resume-swapx`. Jobs and outcomes appended below.

- 2026-10-02 10:xx: seed 2 (from `MSE1024`) → job **11039173**; seed 3 (from `MSE1024S3`) →
  **11039174**; seed 4 (from `MSE1024S4`) → **11039175**; from commit `57fdedd`, two-hour
  walltime. The swapped room renders as intended (checked locally: the plus's anchor and
  corners are yellow obstacles, its old arm cells floor).
