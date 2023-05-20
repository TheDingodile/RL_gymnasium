from helpers import get_env, save_experiment, eval_mode
from agents import Agent, QAgent, Actor_Agent, BaselineAgent, REINFORCE_Agent, Actorcritic_actor, Actorcritic_critic    
from copy import deepcopy
from replay_buffer import replay_buffer, episodic_replay_buffer
from collector import collector
import torch

def eval(**args):
    args = eval_mode(**args)
    env = get_env(**args)
    state, _ = env.reset()
    if args['train_loop'] == "deep_q_learn":
        agent = QAgent(env, **args)
    elif args['train_loop'] == "reinforce_learn":
        agent = REINFORCE_Agent(env, **args)
    elif args['train_loop'] == "actor_critic_learn":
        agent = Actorcritic_actor(env, **args)
    agent.network.load_state_dict(torch.load("trained_agents/" + args["name"] + "/model.pt"))
    while True:
        action = agent.take_action(state)
        state, _, _, _, _ = env.step(action)


def deep_q_learn(**args):
    env = get_env(**args)
    state, _ = env.reset()
    agent = QAgent(env, **args)
    buffer = replay_buffer(**args)
    data_collector = collector(**args)

    while True:
        action = agent.take_action(state)
        new_state, reward, done, truncated, _ = env.step(action)
        buffer.save_data((state, action, reward, new_state, done), truncated)
        data_collector.collect(reward, done, truncated)
        state = new_state
        buffer.get_batch()
        agent.train(buffer)
        save_experiment(agent, data_collector, **args)


def reinforce_learn(**args):
    env = get_env(**args)
    state, _ = env.reset()
    agent = REINFORCE_Agent(env, **args)
    baseline_model = None
    if args['baseline_model']:
        baseline_model = BaselineAgent(env, **args)

    buffer = episodic_replay_buffer(**args)
    data_collector = collector(**args)

    while True:
        action = agent.take_action(state)
        new_state, reward, done, truncated, _ = env.step(action)
        buffer.save_data((state, action, reward, done), truncated)
        data_collector.collect(reward, done, truncated)
        state = new_state
        agent.train(buffer, baseline_model)
        save_experiment(agent, data_collector, **args)

def actor_critic_learn(**args):
    env = get_env(**args)
    state, _ = env.reset()
    agent_actor = Actorcritic_actor(env, **args)
    agent_critic = Actorcritic_critic(env, **args)
    data_collector = collector(**args)
    buffer = episodic_replay_buffer(**args)

    while True:
        action = agent_actor.take_action(state)
        new_state, reward, done, truncated, _ = env.step(action)
        buffer.save_data((state, action, reward, done), truncated)
        data_collector.collect(reward, done, truncated)
        state = new_state
        agent_actor.train(buffer, agent_critic)
        save_experiment(agent_actor, data_collector, **args)