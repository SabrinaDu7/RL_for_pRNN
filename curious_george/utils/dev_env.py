import os
from dotenv import load_dotenv

from curious_george.utils.enums import AgentType
from prnn.utils import MinigridEnvNames

def get_env_var(var_name: str) -> str:
    """Return the value of an environment variable or raise if missing/empty.

    Loads .env lazily on first use (idempotent) so importing this module has
    no side effects.
    """
    load_dotenv()  # Load variables from .env
    value = os.getenv(var_name)
    if value is None or str(value).strip() == "":
        raise EnvironmentError(
            f"Required environment variable '{var_name}' is missing or empty. "
            "Set it in your shell or in a .env file at the project root."
        )
    return value


# A finished main_train.py run always writes these two filenames side by side
# (storage.save_pN_and_acmodel / prnn.utils.checkpoints.save_pN), so pointing
# at the run DIRECTORY is enough to locate both.
PRNN_CKPT_FILENAME = "predictiveNet_state.pt"
ACMODEL_CKPT_FILENAME = "status.pt"


def get_ckpt_dir_var(agent_type: AgentType = AgentType.AC, env_type: MinigridEnvNames = MinigridEnvNames.LRoom) -> str:
    """Name of the env var holding the training-run directory for this
    agent/env combination, e.g. CUR_CKPT_DIR or FOURROOM_RAND_CKPT_DIR."""
    env_prefix = "FOURROOM_" if env_type == MinigridEnvNames.FourRoomsObjs else ""
    agent_prefix = "CUR" if agent_type == AgentType.AC else "RAND"
    return f"{env_prefix}{agent_prefix}_CKPT_DIR"


def get_ckpt_env_vars(agent_type: AgentType = AgentType.AC, env_type: MinigridEnvNames = MinigridEnvNames.LRoom) -> tuple[str, str]:
    """Resolve (prnn_ckpt, acmodel_status_ckpt) paths for a finished training run.

    Preferred: set ONE directory variable (see get_ckpt_dir_var), e.g.
    `CUR_CKPT_DIR=/path/to/pRNN_curious_26-07-23-10-06-25`. This is the
    portable form - the same variable name works locally (.env) and on the
    cluster (exported in the sbatch script), and only the value changes.

    Fallback: the legacy per-file variables (PRNN_CUR_CKPT / ACMODEL_CUR_CKPT
    and their RAND / FOURROOM_ counterparts), kept so existing .env files and
    scripts keep working.
    """
    dir_var = get_ckpt_dir_var(agent_type=agent_type, env_type=env_type)
    try:
        ckpt_dir = get_env_var(dir_var)
    except EnvironmentError:
        ckpt_dir = None

    if ckpt_dir is not None:
        prnn_ckpt = os.path.join(ckpt_dir, PRNN_CKPT_FILENAME)
        acmodel_status_ckpt = os.path.join(ckpt_dir, ACMODEL_CKPT_FILENAME)
        for path in (prnn_ckpt, acmodel_status_ckpt):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{dir_var}={ckpt_dir} does not contain '{os.path.basename(path)}'. "
                    "Point it at a finished training-run directory."
                )
        return prnn_ckpt, acmodel_status_ckpt

    # Legacy names, preserved verbatim including a pre-existing inconsistency:
    # the AC vars are env-specific (PRNN_FOURROOM_CUR_CKPT) but the RANDOM ones
    # are NOT, so a random-agent FourRooms lookup silently returns the LRoom
    # checkpoint. Not fixed here to avoid changing behavior; use the *_CKPT_DIR
    # form above, which is env-specific for both agent types.
    if env_type == MinigridEnvNames.FourRoomsObjs:
        prnn_ckpt_name, acmodel_ckpt_name = "PRNN_FOURROOM", "ACMODEL_FOURROOM"
    else:
        prnn_ckpt_name, acmodel_ckpt_name = "PRNN", "ACMODEL"

    if agent_type == AgentType.AC:
        return get_env_var(f"{prnn_ckpt_name}_CUR_CKPT"), get_env_var(f"{acmodel_ckpt_name}_CUR_CKPT")
    return get_env_var("PRNN_RAND_CKPT"), get_env_var("ACMODEL_RAND_CKPT")


def get_wandb_env_vars(omt: bool) -> tuple[str, str]:
    """Get and validate Weights & Biases environment variables.
    """
    wandb_entity = get_env_var("WANDB_ENTITY")
    wandb_project = get_env_var("WANDB_PROJECT_OMT" if omt else "WANDB_PROJECT")
    return wandb_entity, wandb_project


def get_logdir_env_var() -> str:
    """Get and validate RL storage directory environment variable.
    """
    logdir = get_env_var("RL_STORAGE")
    return logdir


def get_root_dir_env_var() -> str:
    """Get and validate root directory environment variable.
    """
    root_dir = get_env_var("ROOT_DIR")
    return root_dir