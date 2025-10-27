from prnn.utils import (
    make_env,
    MinigridEnvNames,
    ActionEncodingsEnum
)

def get_minigrid_env(env_name: MinigridEnvNames, act_enc: ActionEncodingsEnum):
    env = make_env(env_key=env_name, act_enc=act_enc)
    env.reset()
    return env
