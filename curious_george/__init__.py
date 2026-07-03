# Curious George: curiosity-driven RL with pRNN spatial representations.
#
# Package layout (see docs/refactor_baseline.md and the refactor plan):
#   common       - DEVICE, seeding, small stat helpers
#   envs         - env construction (factory) and wrapper accessors (access)
#   models       - actor-critic models (ACModel, ACModelSR)
#   rl           - PPO algo, rollout agent, obs preprocessing
#   evaluation   - on-policy analysis and figures
#   world_model  - the pRNN seam (adapter; the only place importing prnn)
#   storage      - paths, checkpoints, factories

from curious_george.common import (
    DEVICE,
    seed,
    synthesize,
    mean_by_action,
    grid_to_pixel_coords,
)

from curious_george.rl.agent import (
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

from curious_george.rl.format import (
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
    get_algo,
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
    "get_algo",
    "get_SR_acmodel",
    "save_status",
    "save_analysis_of_agent_behav",
    "get_goal_loc",
    "get_agent",
    "save_pN_and_acmodel",
]
