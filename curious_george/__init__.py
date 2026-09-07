# Curious George: curiosity-driven RL with pRNN spatial representations.
#
# Package layout:
#   configs        - the typed `Config`, its presets and the CLI
#   envs           - environment construction (factory), accessors, the device
#                    pool, observation banks, layouts, the palette, the action graph
#   models         - the actor-critic, the pRNN adapter (the world-model seam),
#                    the device context managers
#   rl             - the PPO algo, rollout collection, the updates
#   training       - construction, the loop, the schedule, wandb logging
#   evaluation     - the online metrics and the offline probes
#   log_and_store  - paths under RL_STORAGE, checkpoints, provenance, wandb reads
#   utils          - device handle, seeding, timing, checkpoint keys, enums
#
# The names below are what the questions repo (../experiment-curiousgeorge)
# and the tests import from the top level. Everything else is imported from
# its module; the sixty-name re-export surface this used to carry was pruned
# 2026-09-06 (audit 2026-09-05, §3).

from curious_george.envs.factory import make_env
from curious_george.log_and_store.storage import get_pN, get_model_dir
from curious_george.models.policy import ACModelSR
from curious_george.rl.algo import PredictivePPOAlgo
from curious_george.rl.collect.agent import ActorCriticAgent
from curious_george.rl.collect.collector import get_dist_travelled
from curious_george.rl.collect.format import get_obss_preprocessor
from curious_george.utils.common import DEVICE, get_device, grid_to_pixel_coords, seed
from curious_george.utils.dev_env import get_ckpt_env_vars, get_env_var
from curious_george.utils.enums import AgentInputType, AgentType

__all__ = [
    "make_env",
    "get_pN",
    "get_model_dir",
    "ACModelSR",
    "PredictivePPOAlgo",
    "ActorCriticAgent",
    "get_dist_travelled",
    "get_obss_preprocessor",
    "DEVICE",
    "get_device",
    "grid_to_pixel_coords",
    "seed",
    "get_ckpt_env_vars",
    "get_env_var",
    "AgentInputType",
    "AgentType",
]
