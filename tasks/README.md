# `tasks/` — evaluation tasks on a pre-trained agent

A **task** takes a *finished* training run (a pRNN checkpoint + an AC-model
checkpoint) and runs a short, controlled experiment on it. This is the
counterpart to `main_train.py`, which trains an agent from scratch: nothing in
here trains a network from random init.

Everything task-generic lives in `curious_george.evaluation.task`
(`setup_task` / `train_phase` / `collect_eval_rollouts[_batched]` /
`FreezeSpec`). A task in this directory supplies only three things:

1. which environments to use for the exploration phase vs. the eval phase
   (they may deliberately differ),
2. what is frozen during the exploration phase (`FreezeSpec`),
3. which per-trajectory statistics to pull out of the frozen eval rollouts
   (`traj_stats_fn`).

`template_task.py` is the annotated skeleton to copy for a new task.

## Layout

```
tasks/
├── template_task.py      skeleton + instructions for a new task
└── omt/                  Object Memory Task (the NOR paradigm)
    ├── main_task.py      hydra entry point (env construction, wandb, run name)
    ├── task.py           ObjectMemoryTask: train phase + eval probe
    ├── metrics.py        view-coordinate projection, quantify_object_learning
    └── figure.py         the object-learning figure + paper figures
```

## The Object Memory Task (OMT)

Models the Novel Object Recognition paradigm. Two environments, and **they
must not be swapped** (`task.py` asserts this):

| phase | env | object | what learns |
|---|---|---|---|
| exploration (`trainNovelObject`) | `env_novel` | novel object **present** | pRNN **and** AC model |
| eval probe (`getTestTrial`) | `env_orig` | object **absent** | nothing (all frozen) |

The probe question: after exposure, does the pRNN still *predict* the object
at the location where it used to be? The signal is the trained net's
prediction **minus** the prediction of `pN_control` — a frozen copy of the
pre-task pRNN (`setup_task(..., control_copy=True)`) — evaluated from the same
fixed noise state, so the two nets differ only by the exposure
(`task.py::_dual_net_predictions`). `quantify_object_learning` turns that into
`goalmodulation` at the object location and `ctlmodulation_diffloc` at the
control locations (`tasks.testing.ctrl_locs`); the headline metric logged to
wandb is `Analysis/Goal Minus Ctrl Vs. Step Count`.

`room_type_green` picks the novel stimulus: `dot` (a ball at
`tasks.new_obj_loc` — the configuration used for the current experiments),
`line`, `plus`, `goal`. `tasks.control=True` makes the "novel" env a plain
copy of `env_orig`, i.e. exposure without any object — the visitation control.

During exposure the pRNN learning rate is multiplied by
`tasks.training.lr_trials` (default 2) for the param groups in
`tasks.training.lrgroups`, and restored afterwards.

## Running it

Checkpoints come from **environment variables**, not from the config
(`curious_george/utils/dev_env.py::get_ckpt_env_vars`). Point one variable at
the finished training run's **directory** — the two filenames inside it
(`predictiveNet_state.pt`, `status.pt`) are always the same, so they're
derived:

| agent / env | variable |
|---|---|
| actor–critic, LRoom | `CUR_CKPT_DIR` |
| random action, LRoom | `RAND_CKPT_DIR` |
| actor–critic, FourRooms | `FOURROOM_CUR_CKPT_DIR` |
| random action, FourRooms | `FOURROOM_RAND_CKPT_DIR` |

This is deliberately **the same variable name locally and on Mila** — only
the value changes, so nothing in the code or configs branches on the machine.
Locally it lives in `.env`; on the cluster `slurm/omt_task.sh` exports it, and
an exported variable wins over `.env` (`load_dotenv` does not override the
existing environment).

The older per-file variables (`PRNN_CUR_CKPT` / `ACMODEL_CUR_CKPT` and their
`RAND` / `FOURROOM_` counterparts) still work as a fallback when the matching
`*_CKPT_DIR` is unset. Note a pre-existing inconsistency preserved in that
fallback: the AC variables are env-specific but the RANDOM ones are not, so a
random-agent FourRooms lookup silently returns the LRoom checkpoint. The
`*_CKPT_DIR` form is env-specific for both agent types and does not have this
problem.

The `justfile` holds the canonical invocations; they differ only in where the
eval trajectories start:

```bash
just omt-start-rand   # random start anywhere
just omt-start-near   # start box tasks.testing.start_{low,up}_bound
just omt-start-away   # start box on the far side
just omt-start-rand-ctrl   # no-object control
# extra hydra overrides pass straight through:
just omt-start-rand tasks.new_obj_loc=[14,7] exp.seed=5200
```

On the cluster: `slurm/omt_task.sh` (see its header for the seed loop and the
checkpoint directory it reads).

### Where the outputs go

`RL_STORAGE` is the single output root for the whole project, and OMT writes
under it:

```
$RL_STORAGE/<run_name>/<traj_count>/predictiveNet_state.pt
$RL_STORAGE/<run_name>/<traj_count>/status.pt
```

Those are deliberately the **same two filenames `main_train.py` writes**, so any
step directory can be handed straight to `CUR_CKPT_DIR` and re-loaded by
`get_ckpt_env_vars` — that is how a task run is chained off another task run.
`status.pt` likewise uses the same keys, with the pRNN's optimizer under its own
`prnn_optimizer_state` (see below).

with `run_name = "{exp.exp_name}-{cur|rand}-{room_type_green}-{MMDD-HHMMSS}"`,
e.g. `omt-cur-dot-0730-120749`. Syncing `$RL_STORAGE` off a cluster node is
therefore sufficient. (Before 2026-07-30 `save_path` was relative and these
landed in the *current working directory* instead — a cluster job that only
rsynced `$RL_STORAGE` silently lost every OMT checkpoint. Old runs sitting at
the repo root are from that era.)

**Checkpoint cadence.** `tasks.training.saving_interval_trajs` (default 200)
is denominated in **trajectories**; `task.py` converts it to the batch count
`train_phase` actually uses, so the spacing stays 200 even if
`rl.trajs_per_batch` changes. A final checkpoint is always written at
`(num_batches - 1) * trajs_per_batch`. A 3000-trajectory run therefore yields
`0, 200, 400, ..., 2800, 2992`. Do not disable this from a launch script —
`slurm/omt_task.sh` once passed `saving_interval=1000000` and kept only the
first and last, which is useless for trajectory-resolved analysis.

Figures are written under the run directory when `figure.py` is called with
`save=True`; the in-training analysis path calls it with `save=False` and logs
the figure straight to wandb.

### Checkpoint interop (fixed 2026-07-30)

Task checkpoints used to be unloadable by the code that loads `main_train`
ones, in two ways. Both are fixed; old checkpoints still load.

1. **Filename.** Steps were written as `pN-<count>.pt`, which
   `get_ckpt_env_vars` cannot find (it looks for `predictiveNet_state.pt`).
   They now use the canonical name — the count is already the directory. For
   reading, `curious_george.resolve_prnn_ckpt(dir)` accepts either name, so
   pre-existing runs on scratch keep working.
2. **Optimizer key collision.** `status.pt["optimizer_state"]` meant the **AC
   Adam** (1 param group) when written by `main_train` and the **pRNN RMSprop**
   (4 param groups) when written by a task. `setup_algo` loads that key into
   the AC Adam, so a task `status.pt` blew up on reload. The pRNN's optimizer
   now has its own key, `prnn_optimizer_state`; `optimizer_state` means the AC
   optimizer everywhere. `load_prnn_optimizer_state` reads the new key and
   falls back to the old one when its shape matches, and `setup_algo` raises a
   named error rather than an opaque torch one if it is handed a legacy file.

**Known naming drift (still not fixed):** `scripts/analysis_OMT_h.py::get_ckpts`,
`scripts/isomap.py` and `scripts/analysis_reward_map.py` hardcode
`omt-cur-dot-noObs-goal{i}{j}/{step}/pN-{step}.pt`, which is neither the name
nor the location `main_task.py` produces today. Point them at
`$RL_STORAGE/<run_name>/<step>/` and use `resolve_prnn_ckpt`.

## Post-refactor status (2026-07-30)

`curious_george/` was reorganised during the
`SabrinaDu7/pRNN → LevensteinLab/pRNN` migration (`docs/migration_prnn_new.md`,
`docs/migration_baseline.md`); most of what used to live inside the task now
comes from `curious_george.evaluation.task`. **Confirmed by running it**: the
OMT entry point is intact on the new stack — a 16-trajectory run
(`tasks.new_obj_loc=[14,7]`, `exp.seed=5200`, seeded from
`outputs/ckpts/pRNN_curious_26-07-23-10-06-25/`) completed on CUDA, exercised
the analysis + figure path, and wrote both checkpoint sets. Not verified:
whether the metric *values* match pre-refactor runs — the migration changed
`bias_lr` from an effective 0 to 0.01, so trajectories are expected to diverge.

Two things to know before scaling up:

- `main_task.py` hardcodes `DEVICE = torch.device("cuda")`. There is no CPU
  fallback; the job needs a GPU.
- `tasks.testing.trajs` must be **smaller than or equal to**
  `rl.trajs_per_batch`. Both are 8 at the defaults, which is fine.

## Where this is going

The experiment these tasks feed: run OMT three times, each with a **different
`tasks.new_obj_loc`**, on the cluster. Take the pRNN checkpoints from those
runs — networks that have now seen a novel object at a known place — and
analyse the **hidden state** for spatial tuning localised near that object.
The per-location comparison is the point: tuning that follows the object
across the three locations is the result; tuning fixed in room coordinates is
not. `scripts/analysis_OMT_h.py` (PCA of hidden states coloured by
object-in-view) is the current starting point for that analysis, subject to
the path drift noted above.
