# Run experiments
omt *EXTRA:
    uv run tasks/ObjectMemoryTask/run_task.py {{EXTRA}}

omt-rand *EXTRA:
    uv run tasks/ObjectMemoryTask/run_task.py exp.random_action_agent=True exp.curious_agent=False {{EXTRA}}

# Formatting and testing
lint:
    uv run ruff format .

test:
    uv run -m pytest -m "not slow"

test-slow:
    uv run -m pytest -m "slow"