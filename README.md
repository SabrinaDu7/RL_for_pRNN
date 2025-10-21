# RL for pRNN

This project is focused on modelling rat exploratory behavior in the Novel Object Recognition (NOR) paradigm, leveraging curiosity as an intrinsic reward signal for RL.
An overview of the file system is found at the end of the README.

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
Then, you should change the environment variables in ```.env```.

## Running pRNN training

Default values for the training pipeline can be found in ```trainRL_Adel.py```. Configs utilize [Hydra](https://hydra.cc/docs/intro/) .and .yaml files. Here is an example run that alters a config value:

```bash
uv run trainRL_Adel.py rl.steps=10000
```


## Running RL training

Possible inputs the agent can receive:

- FO: full observation (often used as a positive control)
- PO: partial observation (the same type of input as the pRNN)
- h: the hidden state of the pRNN
- h+PO: the hidden state of the pRNN and a partial observation

## Setup on Mila's cluster

On the login node:

1. Clone the repo.
2. Create a virtual environment in your desired directory. (Ex: ```uv venv —python 3.10 ~/venvs/venv-pRNN```)
3. Activate and sync the venv: ```source ~/venvs/venv-pRNN/bin/activate``` then ```uv sync --active```
4. Venv is ready to be used on compute nodes. You can deactivate it for now: ```deactivate```

When using ```salloc/srun``` or ```sbatch```, you must activate the venv on the compute node
and use the option ```--active``` to use the active venv. Example run command:

```bash
uv run --active trainRL_Adel.py rl.steps=10000
```

## Overview of file system

# Training
```bash
RL_Trainer
    │
    ├─> Creates: env, acmodel (ACModelSR), predictiveNet (pRNN)
    │
    ├─> Creates: algo = PredictivePPOAlgo(env, acmodel, predictiveNet, ...)
    │
    └─> Training loop:
            │
            ├─> algo.collect_experiences()
            │       │
            │       ├─> For each step:
            │       │   ├─> acmodel.forward(obs, SR) → get action dist
            │       │   ├─> action = dist.sample()
            │       │   ├─> obs, reward = env.step(action)
            │       │   └─> SR = pRNN.predict(obs, action)
            │       │
            │       └─> Returns experiences (obs, actions, rewards, SRs, advantages)
            │
            └─> algo.update_parameters(exps)
                    │
                    ├─> For each batch:
                    │   ├─> acmodel.forward(obs, SR) → get new dist, value
                    │   ├─> Compute PPO loss
                    │   └─> optimizer.step() → update acmodel weights
                    │
                    └─> Optionally train pRNN
```

# Analysis
```bash
RL_Trainer (analysis interval)
    │
    └─> Creates: analysisagent = ActorCriticAgent(env.action_space, acmodel, predictiveNet, DEVICE)
            │
            └─> predictiveNet.calculateSpatialRepresentation(env, analysisagent, ...)
                    │
                    └─> Internally calls: analysisagent.getObservations(env, tsteps)
                            │
                            ├─> For each step:
                            │   ├─> acmodel.forward(obs, SR) → get action dist
                            │   ├─> action = dist.sample()
                            │   ├─> obs = env.step(action)
                            │   └─> SR = pRNN.predict(obs, action)
                            │
                            └─> Returns: observations, actions, states (for analysis)
```