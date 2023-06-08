from helpers import get_env, save_experiment, eval_mode
from agents import Agent, QAgent, Actor_Agent, BaselineAgent, REINFORCE_Agent, Actorcritic_actor, Actorcritic_critic, PPO_Agent, PPO_dual_network_Agent, Soft_actorcritic_Actor, Soft_actorcritic_critic
from copy import deepcopy
from replay_buffer import replay_buffer, episodic_replay_buffer
from collector import collector
import torch

def eval(**args):
    args = eval_mode(**args)
    env = get_env(**args)
    state, _ = env.reset()
    agent = get_agent(env, **args)  
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

def PPO_learn(**args):
    env = get_env(**args)
    state, _ = env.reset()
    agent_actor = PPO_Agent(env, **args)
    value_agent = BaselineAgent(env, **args)
    data_collector = collector(**args)
    buffer = episodic_replay_buffer(log_probs=True, **args)

    while True:
        action, log_probs = agent_actor.take_action(state, output_log_prob=True)
        new_state, reward, done, truncated, _ = env.step(action)
        buffer.save_data((state, action, reward, done, log_probs), truncated)
        data_collector.collect(reward, done, truncated)
        state = new_state
        agent_actor.train(buffer, value_agent)
        save_experiment(agent_actor, data_collector, **args)

def PPO_learn_batches(**args):
    env = get_env(**args)
    state, _ = env.reset()
    agent = PPO_dual_network_Agent(env, **args)
    data_collector = collector(**args)
    buffer = replay_buffer(log_probs=True, **args)
    while True:
        for _ in range(40000 // args["num_envs"]):
            action, log_prob = agent.take_action(state, output_log_prob=True)
            new_state, reward, done, truncated, info = env.step(action)
            buffer.save_data((state, action, reward, new_state, done, log_prob), truncated)
            data_collector.collect(reward, done, truncated)
            state = new_state
            save_experiment(agent, data_collector, **args)

        for _ in range(400):
            states, actions, rewards, next_states, dones, log_probs = buffer.get_batch()
            agent.train(states, actions, next_states, rewards, dones, log_probs)
        buffer.buffer = [None for _ in range(buffer.buffer_size)]
        buffer.counter = 0

def soft_actor_critic_learn(**args):
    env = get_env(**args)
    state, _ = env.reset()
    agent_actor = Soft_actorcritic_Actor(env, **args)
    agent_critic1 = Soft_actorcritic_critic(env, **args)
    agent_critic2 = Soft_actorcritic_critic(env, **args)
    data_collector = collector(**args)
    buffer = replay_buffer(**args)

    while True:
        action = agent_actor.take_action(state)
        new_state, reward, done, truncated, _ = env.step(action)
        buffer.save_data((state, action, reward, new_state, done), truncated)
        data_collector.collect(reward, done, truncated)
        state = new_state
        agent_actor.train(buffer, agent_critic1, agent_critic2)
        save_experiment(agent_actor, data_collector, **args)



def get_agent(env, **args):
    if args['train_loop'] == "deep_q_learn":
        agent = QAgent(env, **args)
    elif args['train_loop'] == "reinforce_learn":
        agent = REINFORCE_Agent(env, **args)
    elif args['train_loop'] == "actor_critic_learn":
        agent = Actorcritic_actor(env, **args)
    elif args['train_loop'] == "PPO_learn":
        agent = PPO_Agent(env, **args)
    elif args['train_loop'] == "PPO_learn_batches":
        agent = PPO_dual_network_Agent(env, **args)
    elif args['train_loop'] == "soft_actor_critic_learn":
        agent = Soft_actorcritic_Actor(env, **args)
    return agent