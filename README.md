# RL for pRNN

**The point of the repo.** This repository is meant to couple the training of a predictive recurrent neural network (pRNN) , the model from [Levenstein et al. 2024](https://www.biorxiv.org/content/10.1101/2024.04.28.591528v2) and a policy network learned with curiosity-driven RL. The the [original pRNN repository](https://github.com/LevensteinLab/pRNN) already solved its task, but did not yet introduce RL.

**Novelty in this repo: Learning a policy.** In the [original pRNN repository](https://github.com/LevensteinLab/pRNN), a fixed random action selection was used instead. Here, we learn a policy network via PPO, where the agent's reward is the pRNN's own prediction error, so the policy learns to seek out what the prnn cannot yet predict, and the prnn learns from the
observations that policy generates.

**What this repo is and is not.** This repo holds the components you assemble to launch a training run or an evaluation: environments, an actor–critic policy, the link to the world model (i.e., prnn), etc. This repo is not an experiment, so does not make any specific claims or provide results beyond the pRNN "learning" and the RL "working".

Examples: 
- `main_train.py` is *an example* of assembling those components into a training run, not
the only way to use them. 
- `run_spatial_analysis`, `run_behavior_analysis` in `curious_george/training/loop.py` are examples of online analysis done during training.

---

## Setup

Managed with [`uv`](https://docs.astral.sh/uv/). Two dependencies are custom forks pinned
in `pyproject.toml` under `[tool.uv.sources]` and resolved to exact commits in `uv.lock` —
`uv sync` is the source of truth for which revision you are running.

```bash
git clone https://github.com/SabrinaDu7/RL_for_pRNN.git
cd RL_for_pRNN
uv sync
```

### IO and logging
- Inputs (what you have to add): (1) Copy `.env.example` to `.env` and fill it in. (2) Runs log to wandb, so `wandb login` first.
- Outputs: `RL_STORAGE` is the single source of truth for where run outputs land on this machine. Other outputs might simply be logged to wandb. 

### Launching a training run
```bash
uv run python main_train.py                              # defaults from Configs/main.yaml
uv run python main_train.py exp.seed=3 rl.frames=2048    # override single keys
uv run python main_train.py env=lroom_multi run=multienv # swap whole config groups
```

Hydra composes `Configs/main.yaml` from the groups beside it (`env/`, `model/`, `algo/`,
`rewards/`, `world_model/`, `run/`, `performance/`). Run length is set in episodes;
environment steps and both optimizer-step budgets are **derived** — read the schedule
printed at startup rather than a comment (`curious_george/training/schedule.py`).

`uv run pypatree` prints the module tree with every public signature.

---

## How the modules fit together

Six subpackages, and a handful of files that are the joints between them. If you are
reading the code for the first time, read the **anchors** in this order.

```
main_train.py
      │
      ├─ training/setup.py     setup_run  -> a run directory + provenance.json
      │                        setup_training -> TrainingComponents (envs, pRNN,
      │                        acmodel, algo, agents) in ONE construction order
      │
      └─ training/loop.py      run_training: the while-loop over gradient updates
                               + run_spatial_analysis / run_behavior_analysis
                               + save_checkpoint, all fired by StepCadence
                                          │
                                          ▼
                               rl/algo.py  PredictivePPOAlgo
                                 collect_experiences() ──► rl/collect/collector.py
                                 update_parameters()   ──► rl/update/policy.py
                                                       └─► rl/update/world_model.py
                                                                    │
                                                                    ▼
                                                    world_model/adapter.py
                                                       PRNNAdapter: the only
                                                       interface to pRNN and
                                                       only `prnn` import
```

### `curious_george/rl` — the agent

**`rl/algo.py` is the hub.** `PredictivePPOAlgo` owns the environments, the actor–critic,
the world-model adapter and the optimizer, and exposes exactly two calls the loop uses:
`collect_experiences()` and `update_parameters()`. Everything below it is a plain function
it delegates to, which is what makes the pieces reusable outside this loop.

- `rl/collect/collector.py` — `collect_rollout`, the rollout itself. One implementation for
  B ≥ 1 environments. Returns a `CollectResult` carrying the transitions, the per-frame
  curiosity rewards, positions, and the diagnostics the loop logs.
- `rl/collect/agent.py` — the actor–critic agent used to act; `format.py` — observation
  preprocessing; `diagnostics.py` — location statistics and the policy/space joint
  distribution accumulated during the rollout.
- `rl/update/policy.py` — `update_policy`, loss-agnostic: it drives the PPO epochs and
  minibatching and calls whichever loss `rl.loss` names.
- `rl/update/losses.py` — the losses (`ppo_clip`, `a2c`); `advantage.py` — GAE;
  `rewards.py` — the curiosity reward, i.e. the pRNN's prediction error.
- `rl/update/world_model.py` — `train_world_model_on_episodes`, the pRNN's gradient steps
  for a rollout. This is where the two learners meet.
- `rl/collect/rollout_graph.py`, `rl/update/policy_graph.py` — CUDA-graph capture of the
  rollout timestep and the PPO minibatch step. Optional, off by default, and gated against
  the eager path.

### `curious_george/world_model` — the pRNN seam

**`world_model/adapter.py` is the boundary.** `PRNNAdapter` is the only module that imports
`prnn`; everything else talks to it. It owns the SR (hidden-state) trackers, the action and
observation encoding, the training step, and the prediction-error computation the curiosity
reward is built from. If the upstream pRNN API changes, this is the file that changes.

`world_model/device.py` — `on_device` / `eval_mode`, the context managers that move models
between CPU and accelerator without losing their identity. Load-bearing: the spatial
evaluation runs on CPU, and a naive `.to()` there silently invalidated captured CUDA
graphs.

### `curious_george/envs` — the world

- `envs/factory.py` — `make_env`, the one way to build an environment shell.
- `envs/access.py` — accessors that reach into the wrapped MiniGrid env (walkable mask,
  grid shape, subroom ids). Use these rather than reaching through wrappers yourself.
- `envs/vector.py` — `DeviceTableShellPool` and `AsyncShellPool`, the multi-environment
  backends. The device pool keeps observations and transitions resident on the accelerator
  and is what the fast configurations use.
- `envs/obs_bank.py` — precomputed `(position, direction) → observation` tables that make
  the device pool possible.
- `envs/layouts.py` — seeded pools of landmark layouts for multi-room training.

### `curious_george/evaluation` — the online metrics

Called by `training/loop.py` on a cadence, but importable on their own.

- `evaluation/spatial.py` — `evaluate_spatial_representation` and
  `evaluate_multi_room_representation`: spatial information, sRSA, and sleep–wake distance.
  The metrics themselves are computed by `prnn`; this module collects the activity and
  reports the coverage of the SI estimate alongside it.
- `evaluation/on_policy.py` — `OnPolicyAnalysis`, `occupancy_counts`, `mutual_info_policy`:
  what the policy did, as arrays and as figures.
- `evaluation/probe.py` — the one way to turn a checkpoint into (hidden state, position),
  with a fixed seed and fixed actions so repeated scoring of a checkpoint agrees.
- `evaluation/task.py` — reusable machinery for evaluations that train further from a
  checkpoint (`setup_task`, `train_phase`, `collect_eval_rollouts`).

### `curious_george/training` — assembly and the loop

- `training/setup.py` — construction, in a fixed order that matters for reproducibility.
- `training/loop.py` — the loop and the interval-triggered sections.
- `training/schedule.py` — every derived count for a run, from one ground truth (episodes).
  Read `TrainingSchedule.summary()`, not a comment.
- `training/logging.py` — the wandb surface.

### Supporting

- `models.py` — `ACModel` / `ACModelSR`, the actor–critic. `ACModelSR` is the one that takes
  the pRNN hidden state as input.
- `provenance.py` — every artifact records what produced it: the resolved commits of this
  repo, `prnn`, `minigrid` and the caller, plus the config and its input artifacts.
- `storage.py` — where things are written; `io/wandb.py` — reading runs back out;
  `check/wandb_compare.py` — comparing two runs on matched environment steps.
- `utils/` — the device handle, seeding, timing, checkpoint keys, enums.

---

## `slurm/` — how this was run on the Mila cluster

Each script is a self-contained sbatch job: it clones this repo at a named branch into
`$SLURM_TMPDIR`, syncs the environment, runs training, and rsyncs the outputs to
`$SCRATCH`. They are the record of how a result was actually produced, and their headers
carry the reasoning for the settings they pass.

| script | what it launches |
|---|---|
| `train_prnn.sh` | a single training run |
| `train_fast.sh` | the tuned production configuration, parameterised by positional arguments (layout set, seed, entropy coefficient, budget, and the CUDA-graph switches) |
| `multienv.sh` | multi-room training |
| `bsweep.sh` | a job array sweeping parallel-environment counts against seeds |
| `async_bench.sh` | a benchmark of async against synchronous rollout collection |

Two things worth knowing before adapting one: the GPU type is load-bearing for anything
with a wall-clock target, and every job fetches into the same shared checkout, so the fetch
is serialised under `flock` and a lost race is non-fatal.

---

## Not documented here yet

**Outputs layout and the test suite are deliberately absent.** Both are mid-prune: this
codebase is larger than the science needs, and the next step is stripping it to essentials
— launching wandb runs throughout and holding bitwise equality against the golden fixtures
while doing it. Documenting the current shape would only have to be rewritten.

What that pruning has to preserve is already gated. `tests/golden/test_golden.py`
compares the training path's tensors against a pinned fixture, and `docs/invalid-runs.md`
records the commits after which previously-reported numbers no longer mean what they did.
The Object Memory Task's equivalent gate moved to the questions repository with the task
itself, and passes there against the same fixture captured here.
