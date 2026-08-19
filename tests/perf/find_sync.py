"""Locate every synchronizing CUDA op in one rollout, with its Python stack.

Uses torch.cuda.set_sync_debug_mode("warn"), which makes the CUDA caching
allocator emit a UserWarning on each synchronizing call. We capture the
warning together with the repo-side stack frame that triggered it.
"""
import collections, sys, traceback, warnings
from pathlib import Path

# The repo is the CWD - this script lives outside the tree, so climbing from
# __file__ walks to / and never terminates.
REPO = Path.cwd()
assert (REPO / "curious_george").is_dir(), f"run from the repo root, not {REPO}"
sys.path.insert(0, str(REPO))

import torch
from hydra import compose, initialize_config_dir

BASE = [
    "logging.wandb_log=false", "logging.save_every_steps=0",
    "logging.analysis_every_steps=0", "logging.plot_every_steps=0",
    "logging.log_every_steps=1000000000",
]
overrides = BASE + sys.argv[1:]
with initialize_config_dir(config_dir=str(REPO / "Configs"), version_base=None):
    cfg = compose(config_name="main", overrides=overrides)

from curious_george.training.setup import setup_training
from curious_george.utils.common import DEVICE

print(f"device={DEVICE}  num_envs={cfg.exp.num_envs}  frames={cfg.rl.frames}")
comps = setup_training(cfg)
algo = comps.algo

# warm up so lazy init / autotune syncs are not counted
exps, _ = algo.collect_experiences()
algo.update_parameters(exps=exps)
torch.cuda.synchronize()

hits: collections.Counter = collections.Counter()

def show(message, category, filename, lineno, file=None, line=None):
    stack = traceback.extract_stack()
    # innermost frame inside the repo (not .venv, not this script)
    site = None
    for fr in reversed(stack):
        if "/.venv/" in fr.filename or fr.filename == __file__:
            continue
        if str(REPO) in fr.filename:
            site = f"{Path(fr.filename).relative_to(REPO)}:{fr.lineno} in {fr.name}()"
            break
    # nearest torch-side call for context
    tsite = next(
        (f"{Path(fr.filename).name}:{fr.lineno}:{fr.name}"
         for fr in reversed(stack) if "/.venv/" in fr.filename), "?")
    hits[(site or "<outside repo>", tsite)] += 1

warnings.showwarning = show
torch.cuda.set_sync_debug_mode("warn")
exps, _ = algo.collect_experiences()
torch.cuda.set_sync_debug_mode("default")
warnings.showwarning = warnings._showwarning_orig  # type: ignore[attr-defined]

T = cfg.rl.frames // cfg.exp.num_envs
total = sum(hits.values())
print(f"\n=== syncs during ONE collect_experiences (T={T} sequential steps) ===")
print(f"total {total}   = {total / T:.2f} per sequential step\n")
for (site, tsite), n in hits.most_common():
    print(f"{n:6d}  {n/T:6.2f}/step   {site}\n                      via {tsite}")
