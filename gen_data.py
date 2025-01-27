from prnn.utils.data import generate_trajectories
from prnn.utils.env import make_env
from prnn.utils.agent import create_agent


env = make_env('MiniGrid-DonutLava-Long-v2', 'farama-minigrid',
               'SpeedHD')

agent = create_agent('MiniGrid-DonutLava-Long-v2', env, 'RandomActionAgent')

generate_trajectories(env, agent, 10000, 1000,
                      '/network/scratch/a/aleksei.efremov/Data' + '/' + env.name + '-' + type(agent).__name__)