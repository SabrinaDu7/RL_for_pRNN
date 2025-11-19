# Run experiments
omt *EXTRA:
    uv run tasks/ObjectMemoryTask/run_task.py {{EXTRA}}

omt-rand *EXTRA:
    uv run tasks/ObjectMemoryTask/run_task.py exp.random_action_agent=True exp.curious_agent=False {{EXTRA}}

# Training
train-prev *EXTRA:
    uv run trainRL_Adel.py exp.exp_name=pRNN-prev predNet.pRNNtype=thRNN_5win_prevAct predNet.action_encoding=SpeedNextHD {{EXTRA}}

train-prev-rand *EXTRA:
    uv run trainRL_Adel.py exp.exp_name=pRNN-prev predNet.pRNNtype=thRNN_5win_prevAct predNet.action_encoding=SpeedNextHD exp.curious_agent=False exp.random_action_agent=True {{EXTRA}}

# Formatting and testing
lint:
    uv run ruff format .

test:
    uv run -m pytest -m "not slow"

test-slow:
    uv run -m pytest -m "slow"