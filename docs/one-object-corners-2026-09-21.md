# One object per room, near the corners — 2026-09-21

## Purpose

Q7–Q10 in the questions repository found the trained agent turning toward the middle of
the room rather than toward objects, on rooms whose three objects all sit mid-room. This
run breaks that confound the direct way: eight rooms, **one** object each, placed near the
room's corners so that "toward the object" and "toward the middle" point in different
directions from most squares. Same network recipe as the MSE1024 checkpoint those questions
analysed, so the only change is the room set.

## The recipe, read off the MSE1024 run's provenance

`outputs/fetched/mx-impassable-n8-s2-mse-h1024_curious_26-09-09-15-01-18/provenance.json`
records:

```
main_train.py multienv-fast --run.seed 2 --run.exp-name mx-impassable-n8-s2-mse-h1024
  --run.wandb-project curious-george-multienv --arch-prnn.loss MSE
  --train-policy.normalize-reward --arch-prnn.hidden-size 1024
  --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
  env.source:selected --env.source.impassable --env.source.positions 0 1 2 3 5 6 7 8
```

Everything before `env.source:` is repeated verbatim through `slurm/placed.sh`'s `extra`
arguments; the source is the new `Placed` (`curious_george/envs/layouts.py`), which takes
one "x,y" anchor per room and a kind index, and applies the affordance on top — impassable
here, as in the recipe.

## The rooms

`throwaway/2026-09-21/one_object_rooms.py` draws them and asserts, per room, that every
cell the landmark paints is floor and sits at wall clearance 2 (`RoomRules.min_wall_distance`:
one floor cell between the object and any wall — "never flush against a corner, so a turn
toward it is testable"). The picture is `throwaway/2026-09-21/one_object_rooms.png`.

| room | anchor | where |
| --- | --- | --- |
| 0 | (3, 3) | top-left corner |
| 1 | (12, 3) | top-right corner |
| 2 | (12, 6) | right wall, above the L's cut-out |
| 3 | (9, 7) | the inner (concave) corner |
| 4 | (8, 12) | bottom-right of the foot |
| 5 | (5, 10) | the foot, off its top-left |
| 6 | (7, 4) | top wall, middle, one row in |
| 7 | (3, 7) | left wall, middle |

One green plus (`EnvContent.kinds[1]`) in every room, impassable; 167 standable squares per
room against 152 in the three-object rooms.

## What changed in the library

- `envs/layouts.py::Placed`: a CLI-reachable room source — `Committed` cannot be built by
  tyro, and every generated source applies wall clearance, so neither could say "one object
  here". `RoomRules` are deliberately not applied by `Placed`; the clearance is the caller's
  choice and the picture shows it.
- `slurm/placed.sh`: `multienv.sh`'s body with the placed source; parsed by
  `tests/test_slurm_invocations.py` like the others.
- Tests: `tests/test_env_layouts.py` (resolves, refuses off-floor cells, reachable from the
  command line), `tests/test_slurm_invocations.py`.

## Launch

```bash
sbatch slurm/placed.sh '3,3 12,3 12,6 9,7 8,12 5,10 7,4 3,7' 1 2 sdu/mixed-count-mse mse-h1024-plus \
    --arch-prnn.loss MSE --train-policy.normalize-reward --arch-prnn.hidden-size 1024 \
    --eval.evals BEHAVIOUR SPATIAL_MULTIROOM TRAJECTORY_PLOT --eval.plot-every-steps 3333328
```

Run name `mx-placed-n8-k1-s2-mse-h1024-plus`, wandb project `curious-george-multienv`.
Job id and outcome: appended below once submitted.

- 2026-09-21: submitted as Mila job **10879573** from commit `1755d49` (queued PENDING on priority at submission).
- 2026-09-21 11:36: **COMPLETED**, exit 0, 25 min 34 s on an L40S, the full budget (43,936
  world-model and 175,744 policy gradient steps; 10 archived checkpoint pairs up to env step
  83,886,080) - the same wall time the MSE1024 run took by its own timestamps. wandb:
  `mx-placed-n8-k1-s2-mse-h1024-plus_curious_26-09-21-11-10-15` in `curious-george-multienv`.
  The run's `provenance.json` records `config.env.source` as `Placed` with exactly the eight
  anchors above, kind 1, impassable. Fetched to
  `outputs/fetched/mx-placed-n8-k1-s2-mse-h1024-plus_curious_26-09-21-11-10-15/` (131 MB,
  wandb directory excluded); the cluster copy is `$SCRATCH/pRNN/placed_10879573/`.
- Not yet done: a checkpoint label for it in the questions repository
  (`src/experiments/Q4/collect.py::CHECKPOINTS`), which is what Q9/Q10's collectors key on.
