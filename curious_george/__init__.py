# Curious George: curiosity-driven RL with pRNN spatial representations.
#
# Package layout (see throwaway/ported/docs_legacy/refactor_baseline.md and the refactor plan):
#   utils        - get_device/on_cuda/DEVICE, seeding, small stat helpers, checkpoints, enums, dev_env
#   envs         - env construction (factory) and wrapper accessors (access)
#   models       - actor-critic models (ACModel, ACModelSR)
#   rl           - PPO algo, rollout agent, obs preprocessing
#   evaluation   - on-policy analysis and figures
#   world_model  - the pRNN seam (adapter; the only place importing prnn)
#   storage      - paths, checkpoints, factories

from curious_george.utils.common import (
    DEVICE,
    get_device,
    on_cuda,
    seed,
    synthesize,
    mean_by_action,
    grid_to_pixel_coords,
)

from curious_george.utils.enums import (
    AgentInputType,
    AgentType,
)

from curious_george.utils.checkpoints import (
    ACMODEL_STATUS,
    StatusCkptKeys,
    load_statedict_from_acmodel_status,
)

from curious_george.utils.dev_env import (
    get_env_var,
    get_ckpt_env_vars,
    resolve_prnn_ckpt,
    get_wandb_env_vars,
    get_logdir_env_var,
    get_root_dir_env_var,
)

from curious_george.rl.collect.agent import (
    ActorCriticAgent,
)

from curious_george.rl.algo import (
    compare_trajs,
    get_dist_travelled,
    PredictivePPOAlgo,
)

from curious_george.evaluation.on_policy import (
    mutual_info_policy,
    plot_heatmaps,
    EnvironmentFeaturesAnalysis,
    OnPolicyAnalysis,
    get_occupancy_fig,
    occupancy_counts,
)

from curious_george.envs.factory import (
    episode_video_trigger,
    make_env,
    ResetWrapper,
    HDObsWrapper,
)

from curious_george.envs.access import (
    get_subroom_id,
)

from curious_george.rl.collect.format import (
    get_obss_preprocessor,
    preprocess_images,
    preprocess_int,
    preprocess_texts,
    Vocabulary,
)

from curious_george.models import (
    init_params,
    ACModel,
    ACModelSR,
)

from curious_george.storage import (
    create_folders_if_necessary,
    get_storage_dir,
    get_model_dir,
    get_video_dir,
    get_tmp_dir,
    get_tmp_model_dir,
    get_status_path,
    get_status,
    get_pN,
    get_SR_acmodel,
    save_status,
    save_analysis_of_agent_behav,
    get_goal_loc,
    get_agent,
    save_pN_and_acmodel,
)


__all__ = [
    # Agent
    "ActorCriticAgent",
    # Algo
    "compare_trajs",
    "PredictivePPOAlgo",
    # Analysis
    "mutual_info_policy",
    "plot_heatmaps",
    "EnvironmentFeaturesAnalysis",
    "OnPolicyAnalysis",
    "get_occupancy_fig",
    "occupancy_counts",
    "get_dist_travelled",
    # Env
    "episode_video_trigger",
    "make_env",
    "ResetWrapper",
    "HDObsWrapper",
    "get_subroom_id",
    # Format
    "get_obss_preprocessor",
    "preprocess_images",
    "preprocess_int",
    "preprocess_texts",
    "Vocabulary",
    # Model
    "init_params",
    "ACModel",
    "ACModelSR",
    # Other
    "DEVICE",
    "get_device",
    "on_cuda",
    "seed",
    "synthesize",
    "mean_by_action",
    "grid_to_pixel_coords",
    # Storage
    "create_folders_if_necessary",
    "get_storage_dir",
    "get_model_dir",
    "get_video_dir",
    "get_tmp_dir",
    "get_tmp_model_dir",
    "get_status_path",
    "get_status",
    "get_pN",
    "get_SR_acmodel",
    "save_status",
    "save_analysis_of_agent_behav",
    "get_goal_loc",
    "get_agent",
    # Enums
    "AgentInputType",
    "AgentType",
    # Checkpoints
    "ACMODEL_STATUS",
    "StatusCkptKeys",
    "load_statedict_from_acmodel_status",
    # Env vars
    "get_env_var",
    "get_ckpt_env_vars",
    "resolve_prnn_ckpt",
    "get_wandb_env_vars",
    "get_logdir_env_var",
    "get_root_dir_env_var",
    "save_pN_and_acmodel",
]
