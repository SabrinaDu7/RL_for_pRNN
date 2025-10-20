# Formatting and testing
lint:
    uv run ruff format .

test:
    uv run -m pytest -m "not slow"