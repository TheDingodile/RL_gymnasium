import gymnasium as gym
from gymnasium.utils.play import play

def run(parameters):
    annotations = dict(parameters.__annotations__)
    params = {e: parameters.__getattribute__(parameters, e) for e in annotations}
    print("---------")
    print("parameters: ")
    print(params)
    print("---------")
    params["train_loop"](**params)

def get_env(env_name, render_mode, continuous, num_envs, **args):
    try:
        env = gym.vector.make(env_name, num_envs=num_envs, render_mode=render_mode, continuous=continuous)
    except:
        env = gym.vector.make(env_name, num_envs=num_envs, render_mode=render_mode)
    return env
    