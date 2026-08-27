# The OMT recipes moved to ../curious-george-questions (justfile there) with
# the task itself on 2026-08-25. This repo is the library; questions live in
# the questions repo.

# Training
fourroom *EXTRA:
    uv run main_train.py logging.wandb_project=curious-george-fourroom exp.exp_name=pRNN_fourroom exp.env_name=MiniGrid-FourRooms-Objects-v0 {{EXTRA}}

train-rand-fourroom *EXTRA:
    just fourroom exp.start_rand=False predNet.pRNNtype=thRNN_5win exp.curious_agent=False exp.random_action_agent=True {{EXTRA}}

train-fourroom *EXTRA:
    just fourroom exp.start_rand=False predNet.pRNNtype=thRNN_5win {{EXTRA}}

train EXP_NAME *EXTRA:
    uv run main_train.py exp.exp_name=pRNN{{EXP_NAME}} predNet.pRNNtype=thRNN_5win {{EXTRA}}

train-prev *EXTRA:
    uv run main_train.py exp.exp_name=pRNN-prev predNet.pRNNtype=thRNN_5win_prevAct predNet.action_encoding=SpeedNextHD {{EXTRA}}

train-prev-rand *EXTRA:
    uv run main_train.py exp.exp_name=pRNN-prev predNet.pRNNtype=thRNN_5win_prevAct predNet.action_encoding=SpeedNextHD exp.curious_agent=False exp.random_action_agent=True {{EXTRA}}

# Formatting and testing
lint:
    uv run ruff format .

test:
    uv run -m pytest -m "not slow"

test-slow:
    uv run -m pytest -m "slow"