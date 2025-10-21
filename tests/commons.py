from minigrid.envs import LEnv_18_goal, LEnv_16_goal
from utils import get_ckpt_env_vars

SIZE = 16
ENV_NAME = f"MiniGrid-LRoom_Goal-{SIZE}x{SIZE}-v0"
ENV_CLASS = LEnv_18_goal if SIZE == 18 else LEnv_16_goal
PRNN_CKPT, ACMODEL_STATUS_CKPT = get_ckpt_env_vars()
