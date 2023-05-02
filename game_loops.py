from helpers import get_env, save_experiment, eval_mode
from agents import Agent, QAgent
from copy import deepcopy
from replay_buffer import replay_buffer
from collector import collector
import torch

def eval(**args):
    args = eval_mode(**args)
    env = get_env(**args)
    state, _ = env.reset()
    if args['train_loop'] == "deep_q_learn":
        agent = QAgent(env, **args)
    agent.network.load_state_dict(torch.load("trained_agents/" + args["name"] + "/model.pt"))
    while True:
        action = agent.take_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        # print("state: ", state, "new_state: ", new_state, "reward: ", reward, "done: ", done, "truncated: ", truncated, "info: ", info)
        state = new_state


def deep_q_learn(**args):
    env = get_env(**args)
    state, _ = env.reset()
    agent = QAgent(env, **args)
    buffer = replay_buffer(**args)
    data_collector = collector(**args)

    while True:
        action = agent.take_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        buffer.save_data((state, action, reward, new_state, done), truncated)
        data_collector.collect(reward, done)
        state = new_state
        buffer.get_batch()
        agent.train(buffer)
        save_experiment(agent, data_collector, **args)