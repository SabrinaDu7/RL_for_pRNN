import os
from dotenv import load_dotenv

from utils.enums import AgentType
from prnn.utils import MinigridEnvNames

load_dotenv()  # Load variables from .env

def get_env_var(var_name: str) -> str:
    """Return the value of an environment variable or raise if missing/empty.
    """
    value = os.getenv(var_name)
    if value is None or str(value).strip() == "":
        raise EnvironmentError(
            f"Required environment variable '{var_name}' is missing or empty. "
            "Set it in your shell or in a .env file at the project root."
        )
    return value


def get_ckpt_env_vars(agent_type: AgentType = AgentType.AC, env_type: MinigridEnvNames = MinigridEnvNames.LRoom) -> tuple[str, str]:
    """Get and validate checkpoint-related environment variables.
    """
    if env_type == MinigridEnvNames.FourRoomsObjs:
        prnn_ckpt_name = "PRNN_FOURROOM"
        acmodel_ckpt_name = "ACMODEL_FOURROOM"
    else:
        prnn_ckpt_name = "PRNN"
        acmodel_ckpt_name = "ACMODEL"

    prnn_ckpt = get_env_var(f"{prnn_ckpt_name}_CUR_CKPT") if agent_type == AgentType.AC  else get_env_var("PRNN_RAND_CKPT")
    acmodel_status_ckpt = get_env_var(f"{acmodel_ckpt_name}_CUR_CKPT") if agent_type == AgentType.AC else get_env_var("ACMODEL_RAND_CKPT")

    return prnn_ckpt, acmodel_status_ckpt


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