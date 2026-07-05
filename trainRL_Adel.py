"""DEPRECATED shim: training moved to main_train.py.

Kept so existing shell commands / scripts keep working. Verified A/B: this
entry and main_train.py produce bitwise-identical checkpoints (see
docs/refactor_progress.md).
"""

from main_train import my_main

if __name__ == "__main__":
    my_main()
