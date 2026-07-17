# RL for pRNN

This project is focused on modelling rat exploratory behavior in the Novel Object Recognition (NOR) paradigm, leveraging curiosity as an intrinsic reward signal for RL.
An overview of the code layout and training flow is found at the end of the README.

## Project Setup

This project is managed using [`uv`](https://docs.astral.sh/uv/).
See [workspace documentation](https://docs.astral.sh/uv/concepts/projects/workspaces).
A `justfile` is defined to automate common tasks like running files with [`just`](https://github.com/casey/just).

This project utilizes a custom [minigrid](https://github.com/SabrinaDu7/minigrid) package and a custom [pRNN](https://github.com/SabrinaDu7/pRNN) package. Please have both of these packages in the same directory as RL_for_pRNN.

```bash
# Clone the pRNN and minigrid repos beforehand.
git clone https://github.com/SabrinaDu7/RL_for_pRNN.git
cd RL_for_pRNN/

# Create and activate venv
uv venv --python 3.10.15
source .venv/bin/activate

# Download dependencies from pyproject.toml
uv sync
```

Then, set the environment variables in ```.env``` (checkpoint paths, wandb
entity/project, and `RL_STORAGE` — the single source of truth for where run
outputs land; it points at the repo-local `./storage` by default).

## Running training

The entry point is ```main_train.py``` (```trainRL_Adel.py``` is a deprecated
shim to the same thing). Construction, the loop, and wandb logging live in
```curious_george/training/{setup,loop,logging}.py```. Configs use
[Hydra](https://hydra.cc/docs/intro/): `Configs/Conf1_Adel.yaml` composes
swappable groups, and any key can be overridden from the CLI:

```bash
# default run (PPO, curiosity reward, legacy alignment)
uv run main_train.py

# override single keys
uv run main_train.py rl.steps=10000 logging.save_interval=0

# swap whole components (Configs/<group>/*.yaml)
uv run main_train.py algo=a2c                      # loss function (rl.loss)
uv run main_train.py rewards=curious_next_obs      # corrected t+1 curiosity alignment
uv run main_train.py world_model=thrnn5win_prevact # pRNN arch + matching action encoding
uv run main_train.py env=fourrooms model=plain_ac  # environment / AC architecture
uv run main_train.py exp.num_envs=4                # parallel rollout collection (B=1 default)
```

Possible inputs the agent can receive (config group `model=`):

- FO: full observation (often used as a positive control)
- PO: partial observation (the same type of input as the pRNN)
- h: the hidden state of the pRNN
- h+PO: the hidden state of the pRNN and a partial observation

## Object Memory Task

```bash
uv run tasks/ObjectMemoryTask/run_task.py
```

## Setup on Mila's cluster

On the login node:

1. Clone the repo.
2. Create a virtual environment in your desired directory. (Ex: ```uv venv —python 3.10 ~/venvs/venv-pRNN```)
3. Activate and sync the venv: ```source ~/venvs/venv-pRNN/bin/activate``` then ```uv sync --active```
4. Venv is ready to be used on compute nodes. You can deactivate it for now: ```deactivate```

When using ```salloc/srun``` or ```sbatch```, you must activate the venv on the compute node
and use the option ```--active``` to use the active venv. Example run command:

```bash
uv run --active main_train.py rl.steps=10000
```

# Overview of the code

```
main_train.py                hydra entry: setup -> loop
curious_george/
  training/                  setup.py (build everything), loop.py, logging.py (wandb)
  rl/algo.py                 ONE PredictivePPOAlgo for B>=1 envs
  rl/collect/                rollout loop, diagnostics, agent, obs preprocessing
  rl/update/                 losses (ppo_clip/a2c registry), updater, GAE,
                             curiosity rewards (+ reward_alignment), pRNN training
  world_model/               PRNNAdapter + SR trackers (the only prnn seam),
                             on_device/eval_mode context managers
  evaluation/                sRSA + SWdist (spatial.py), on-policy analysis
  envs/                      make_env + wrapper accessors
  models.py                  ACModel, ACModelSR
  utils/                     enums, checkpoints, env vars, DEVICE/seed
  storage.py                 run-output paths + checkpoint factories
tasks/ObjectMemoryTask/      NOR task: train near novel object, quantify learning
tests/golden/                bitwise behavior oracle + cross-version harnesses
docs/                        refactor notes/baseline/progress, experiment logs
```

## Training flow

```
main_train.py
    │
    ├─> setup_training(cfg)  →  envs, pRNN, ACModelSR, PredictivePPOAlgo, agents
    │
    └─> run_training():
            │
            ├─> algo.collect_experiences()          [rl/collect/collector.py]
            │       ├─> per step (batched over B envs):
            │       │   ├─> acmodel(obs, SR) → dist, value; action = dist.sample()
            │       │   ├─> obs, reward = env.step(action)
            │       │   └─> SR tracker step (pRNN predict_single / batched rnn)
            │       ├─> curiosity rewards from pRNN prediction error
            │       │   (rl/update/rewards.py; rl.reward_alignment: legacy|next_obs)
            │       └─> GAE per env stream → experiences
            │
            ├─> algo.update_parameters(exps)        [rl/update/]
            │       ├─> updater: epochs/minibatches; loss from LOSSES[rl.loss]
            │       └─> pRNN trained per episode segment (if predNet.train)
            │
            ├─> every log_interval: wandb metrics + sample-trajectory figure
            ├─> every analysis_interval: sRSA + SWdist (evaluation/spatial.py)
            │   and on-policy analysis (reuses the training rollout - free)
            └─> every save_interval: status.pt + predictiveNet_state.pt
                (0 = no checkpoints; everything lands under RL_STORAGE)
```

## Behavior guarantees

The refactor is pinned by `tests/golden/golden_v0.pt` — a bitwise oracle of
the pre-refactor training path. `uv run pytest` runs the suite;
`docs/refactor_notes.md` documents the temporal-alignment contract, device
policy, and batched-mode constraints.
