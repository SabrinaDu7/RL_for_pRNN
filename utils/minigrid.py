from prnn.utils import (
    MinigridEnvNames,
    ActionEncodingsEnum
)
import RLutils
from utils import AgentInputType

# TODO: To get rid of this function and make RLutils.make_env directly usable, we need to
def get_minigrid_env(env_name: MinigridEnvNames, act_enc: ActionEncodingsEnum, input_type: AgentInputType):
    env = RLutils.make_env(env_key=env_name, input_type=input_type, act_enc=act_enc)
    env.reset()
    return env
