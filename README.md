# RL for pRNN

This project is focused on modelling rat exploratory behavior, leveraging curiosity as an intrinsic reward signal for RL. We use a learned policy instead of a fixed one to feed environmental observations
to the pRNN. The agent only uses the hidden state of the pRNN to make decisions within an episode.
An overview of the code layout and training flow is found at the end of the README.

To get started, run `uv run pypatree` to see the directory structure.

## Project Setup

This project is managed using [`uv`](https://docs.astral.sh/uv/).
See [workspace documentation](https://docs.astral.sh/uv/concepts/projects/workspaces).
A `justfile` is defined to automate common tasks like running files with [`just`](https://github.com/casey/just).

This project utilizes a custom [minigrid](https://github.com/SabrinaDu7/minigrid) package and a custom
[pRNN](https://github.com/LevensteinLab/pRNN) package. Both are pinned in `pyproject.toml`
(`[tool.uv.sources]`) and resolved to exact commits in `uv.lock`; `uv sync` is the source of truth for
which revision you are running.

```bash
# Clone the pRNN and minigrid repos beforehand.
git clone https://github.com/SabrinaDu7/RL_for_pRNN.git
cd RL_for_pRNN/

# Create and activate venv
uv sync
source .venv/bin/activate
```

Then, set the environment variables in ```.env``` (checkpoint paths, wandb
entity/project, and `RL_STORAGE` — the single source of truth for where run
outputs land; it points at the repo-local `./outputs` by default).

Training in this repo logs to wandb. Ensure that you run `wandb login` and
set the correct entity and project in `.env`.

## Training Runs

### Training from scratch

A training run from scratch is started with ```main_train.py``` (```trainRL_Adel.py``` is deprecated). Construction, the loop, and wandb logging live in
```curious_george/training/{setup,loop,logging}.py```. Configs use
[Hydra](https://hydra.cc/docs/intro/): `Configs/main.yaml` composes swappable
groups, and any key can be overridden from the CLI. Run `uv run main_train.py --cfg job`
to print the fully composed config — that, not this file, is where the defaults live.

```bash
# default run: the groups listed under `defaults:` in Configs/main.yaml
uv run main_train.py

# override single keys. Run length is set in EPISODES; total environment steps
# (and both optimizer-step budgets) are derived - see the schedule printed at
# startup, and curious_george/training/schedule.py.
uv run main_train.py rl.episodes_total=40 logging.save_every_steps=0

# swap whole components (Configs/<group>/*.yaml)
uv run main_train.py algo=a2c                      # loss function (rl.loss)
uv run main_train.py rewards=curious                # curiosity reward alignment variant
uv run main_train.py world_model=thrnn5win_prevact # pRNN arch + matching action encoding
uv run main_train.py env=fourrooms model=plain_ac  # environment / AC architecture
uv run main_train.py exp.num_envs=4                # parallel rollout collection
```

Possible inputs the agent can receive (config group `model=`):

- FO: full observation (often used as a positive control)
- PO: partial observation (the same type of input as the pRNN)
- h: the hidden state of the pRNN
- h+PO: the hidden state of the pRNN and a partial observation

**HOWEVER, we almost always only use h.**

### Training for a task (from scratch or checkpoint)

To run additional analyses, we may change the task/environment that
the pRNN-agent interacts with and trains in. For example, to train
on the Object Memory Task, which involves the introduction of a 
novel object if a familiar environment (ie we train from a checkpoint):

```bash
uv run tasks/omt/main_task.py
```

`justfile` holds the canonical invocations (`just omt-start-rand` and friends).
`tasks/README.md` documents how a task loads its checkpoints, what each one
freezes, and where its outputs land.

## Hardware: Mila's cluster

This repo can be run locally (GPU or CPU) or on Mila's cluster. On the cluster,
you can use slurm scripts listed in `slurm/` to get started.


## Training (from scratch) flow

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
            ├─> every log_every_steps: wandb metrics
            ├─> every plot_every_steps: sample-trajectory + behaviour figures
            ├─> every analysis_every_steps: sRSA + SWdist (evaluation/spatial.py)
            │   and on-policy analysis (reuses the training rollout - free)
            └─> every save_every_steps: status.pt + predictiveNet_state.pt
                (0 = disabled; everything lands under RL_STORAGE)

All four cadences are counted in ENVIRONMENT STEPS, not updates, because an
update is rl.frames steps and therefore scales with exp.num_envs.
```

## Behavior guarantees

`uv run pytest` is the gate. `docs/refactor_notes.md` documents the
temporal-alignment contract, device policy, and batched-mode constraints.

⚠️ **The training golden fixture is not enforced.** `tests/golden/` holds
`golden_v0.pt` (pre-migration stack) and `golden_v1.pt` (current stack), written
by `capture_golden.py` and compared by the standalone `compare_io.py` — but
neither is a `test_*.py`, so pytest never collects them and no run of the suite
reads either fixture. The training path is currently gated only by the ordinary
unit tests. The OMT path *is* gated, by `tests/golden_omt/test_golden_omt.py`.
