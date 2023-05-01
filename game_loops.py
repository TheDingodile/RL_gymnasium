from helpers import get_env
from agents import Agent, QAgent
from copy import deepcopy
from replay_buffer import replay_buffer
from collector import collector

def deep_q_learn(**args):
    env = get_env(**args)
    state, _ = env.reset()
    agent = QAgent(env, **args)
    buffer = replay_buffer(**args)
    data_collector = collector(**args)

    while True:
        action = agent.take_action(state)
        new_state, reward, done, _, _ = env.step(action)
        buffer.save_data((state, action, reward, new_state, done))
        data_collector.collect(reward, done)
        state = new_state
        buffer.get_batch()
        agent.train(buffer)