# Run experiments
omt-start-near *EXTRA:
    uv run tasks/omt/main_task.py tasks.testing.start_random=False {{EXTRA}}

omt-start-away *EXTRA:
    uv run tasks/omt/main_task.py tasks.testing.start_random=False tasks.testing.start_up_bound=[8,14] tasks.testing.start_low_bound=[1,7] {{EXTRA}}

omt-start-rand *EXTRA:
    uv run tasks/omt/main_task.py tasks.testing.start_random=True tasks.testing.start_up_bound=[] tasks.testing.start_low_bound=[] {{EXTRA}}

omt-rand-start-near *EXTRA:
    just omt-start-near exp.random_action_agent=True exp.curious_agent=False tasks.control=True {{EXTRA}}

omt-rand-start-away *EXTRA:
    just omt-start-away exp.random_action_agent=True exp.curious_agent=False tasks.control=True {{EXTRA}}

omt-rand-start-rand *EXTRA:
    just omt-start-rand exp.random_action_agent=True exp.curious_agent=False tasks.control=True {{EXTRA}}


# Run control omt
omt-start-rand-ctrl *EXTRA:
    just omt-start-rand tasks.control=True tasks.training.saving_interval=1000000 logging.wandb_project=curious-george-ctrl tasks.new_obj_loc=[7,11] {{EXTRA}}

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