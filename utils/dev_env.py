import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env

def get_ckpt_env_vars():
    prnn_ckpt = os.getenv("PRNN_CKPT")
    acmodel_status_ckpt = os.getenv("ACMODEL_CKPT")
    return prnn_ckpt, acmodel_status_ckpt

def get_wandb_env_vars():
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_project = os.getenv("WANDB_PROJECT")
    return wandb_entity, wandb_project
