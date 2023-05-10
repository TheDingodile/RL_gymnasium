import gymnasium as gym
from gymnasium.utils.play import play
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import json
from exploration import Explorations, Exploration
sns.set_theme()

def run(parameters):
    annotations = dict(parameters.__annotations__)
    params = {e: parameters.__getattribute__(parameters, e) for e in annotations}
    if params["train_loop"].__name__ != "eval":
        write_parameters(params)
    if params["train_loop"].__name__ == "eval":
        params = eval_mode(**params)
    elif params["train_loop"].__name__ == "reinforce_learn":
        params = reinforce_mode(**params)
    params["train_loop"](**params)

def get_env(env_name, render_mode, continuous, num_envs, **args):
    try:
        env = gym.vector.make(env_name, num_envs=num_envs, render_mode=render_mode, continuous=continuous)
    except:
        env = gym.vector.make(env_name, num_envs=num_envs, render_mode=render_mode)
    return env

def write_parameters(params):
    name = params['name']
    if not os.path.exists("trained_agents/" + name):
        os.makedirs("trained_agents/" + name)

    with open('trained_agents/' + name + '/parameters.txt', 'w') as f:
        f.write("---------\n")
        f.write("parameters: \n")
        for key in params:
            if key == "train_loop":
                f.write(key + ": " + params[key].__name__ + "\n")
            else:
                f.write(key + ": " + str(params[key]) + "\n")
        f.write("---------")

def get_action_space(env, continuous, **args):
    if continuous:
        return env.action_space.shape[0]
    else:
        return env.action_space[0].n

def eval_mode(**args):
    args["num_envs"] = 1
    args["render_mode"] = "human"
    args["exploration"] = Explorations.greedy
    # open txt file
    with open('trained_agents/' + args["name"] + '/parameters.txt', 'r') as f:
        # wait for line with train_loop
        while True:
            line = f.readline()
            if line.startswith("train_loop: "):
                args["train_loop"] = line.split("train_loop: ")[1].strip()
            if line.startswith("continuous: "):
                args["continuous"] = line.split("continuous: ")[1].strip() == "True"
            # if no more lines break
            if line == "":
                break
    return args

def reinforce_mode(**args):
    args["exploration"] = Explorations.softmax
    return args

def save_experiment(agent, data_collector, name, save_agent_every, **args):
    if data_collector.counter % save_agent_every < agent.num_envs:
        torch.save(agent.network.state_dict(), "trained_agents/" + name + "/model.pt")

        measure_every = data_collector.measure_performance_every
        dones = np.array(data_collector.all_dones)
        rewards = np.array(data_collector.all_rewards)
        x_axis = np.arange(len(rewards)) * measure_every

        plt.plot(x_axis, rewards/measure_every)
        plt.xlabel("frames")
        plt.ylabel("reward per frame")
        plt.title("reward per frame - " + name)
        plt.savefig("trained_agents/" + name + "/reward_per_frame.png")
        plt.clf()

        plt.plot(x_axis, rewards/dones)
        plt.xlabel("frames")
        plt.ylabel("reward per episode")
        plt.title("reward per episode - " + name)
        plt.savefig("trained_agents/" + name + "/reward_per_episode.png")
        plt.clf()

        plt.plot(x_axis, measure_every/dones)
        plt.xlabel("frames")
        plt.ylabel("episode length")
        plt.title("episode length - " + name)
        plt.savefig("trained_agents/" + name + "/episode_length.png")
        plt.clf()


