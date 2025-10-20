import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

def require_env(var_name: str) -> str:
    """Return the value of an environment variable or raise if missing/empty.
    """
    value = os.getenv(var_name)
    if value is None or str(value).strip() == "":
        raise EnvironmentError(
            f"Required environment variable '{var_name}' is missing or empty. "
            "Set it in your shell or in a .env file at the project root."
        )
    return value


def get_ckpt_env_vars() -> tuple[str, str]:
    """Get and validate checkpoint-related environment variables.
    """
    prnn_ckpt = require_env("PRNN_CKPT")
    acmodel_status_ckpt = require_env("ACMODEL_CKPT")
    return prnn_ckpt, acmodel_status_ckpt


def get_wandb_env_vars() -> tuple[str, str]:
    """Get and validate Weights & Biases environment variables.
    """
    wandb_entity = require_env("WANDB_ENTITY")
    wandb_project = require_env("WANDB_PROJECT")
    return wandb_entity, wandb_project
