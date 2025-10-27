# Run experiments
omt *EXTRA:
    uv run tasks/ObjectMemoryTask/run_task.py {{EXTRA}}

# Formatting and testing
lint:
    uv run ruff format .

test:
    uv run -m pytest -m "not slow"