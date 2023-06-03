import torch.nn as nn
import torch.optim as optim
import torch
from replay_buffer import replay_buffer, episodic_replay_buffer
from copy import deepcopy
from exploration import Exploration, Explorations
from networks import Network, Networks
from helpers import get_action_space
from torch.distributions.multivariate_normal import MultivariateNormal

class Agent():
    def __init__(self, env, network, num_envs, exploration, learning_rate, weight_decay, train_after_frames, trains_every_frames, batch_size, **args):
            
        self.action_space = get_action_space(env, **args)
        self.train_after_frames = train_after_frames
        self.trains_every_frames = trains_every_frames
        self.num_envs = num_envs
        self.batch_size = batch_size

        self.network = Network(network, env, **args).network
        self.exploration = Exploration(exploration, self.action_space, **args)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def take_action(self, state):
        policy = self.forward(torch.tensor(state))
        action = self.exploration.explore(policy)
        return action
    
    def forward(self, state):
        return self.network(state)
    
    def train(self, buffer):
        pass

class QAgent(Agent):
    def __init__(self, env, update_target_every_frames, gamma, **args):
        super().__init__(env, Networks.QNetwork, **args)
        self.target_network = deepcopy(self.network)
        self.update_target_every_frames = update_target_every_frames
        self.gamma = gamma

    def train(self, buffer: replay_buffer):
        if buffer.counter < self.train_after_frames:
            return
        if (buffer.counter // self.num_envs) % self.update_target_every_frames == 0:
            self.target_network = deepcopy(self.network)
        for _ in range(self.trains_every_frames):
            states, actions, rewards, next_states, dones = buffer.get_batch()
            q_values = torch.gather(self.forward(states), 1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = torch.gather(self.target_network(next_states), 1, torch.argmax(self.forward(next_states), 1).unsqueeze(1)).squeeze(1)
            target = (rewards + self.gamma * next_q_values * (1 - dones.int())).detach().float()
            loss = self.criterion(q_values, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

class Actor_Agent(Agent):
    def __init__(self, env, entropy_regulization, **args):
        continuous = args.get("continuous", False)
        super().__init__(env, Networks.Normal_distribution if continuous else Networks.Actor_Network , **args)
        self.entropy_regulization = entropy_regulization
        self.continuous = continuous

    def train(self, buffer: episodic_replay_buffer, base_line_model):
        pass

    def take_action(self, state, output_log_prob=False):
        policy = self.forward(torch.tensor(state))
        action = self.exploration.explore(policy)
        if output_log_prob:
            return action, self.log_policy(policy, torch.tensor(action))[0].detach()
        else:
            return action

    def log_policy(self, policy, actions):
        if self.continuous:
            return MultivariateNormal(policy, 0.2 * torch.eye(self.action_space)).log_prob(actions), - ((actions[:, 0] - 0.5) ** 2 + (torch.abs(actions[:, 1]) - 0.5) ** 2)
        else:
            log_policy = torch.log(policy)
            entropy_of_policy = -torch.sum(policy * log_policy, dim=1)
            return torch.gather(log_policy, 1, actions.unsqueeze(1)).squeeze(1), entropy_of_policy

class REINFORCE_Agent(Actor_Agent):
    def __init__(self, env, **args):
        super().__init__(env, **args)

    def train(self, buffer: episodic_replay_buffer, base_line_model):
        if buffer.is_ready_to_train() == False:
            return
        states, actions, labels = buffer.get_data_monte_carlo()
        for _ in range(self.trains_every_frames):
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i+self.batch_size]
                batch_actions = actions[i:i+self.batch_size]
                batch_labels = labels[i:i+self.batch_size]
                policy = self.forward(batch_states)
                log_policy, entropy_of_policy = self.log_policy(policy, batch_actions)
                if base_line_model != None:
                    base_line_model.train(batch_states, batch_labels)
                    baseline = base_line_model.forward(batch_states)
                    batch_labels = batch_labels - baseline
                loss = torch.mean(-log_policy * batch_labels - self.entropy_regulization * entropy_of_policy)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

class Actorcritic_critic(Agent):
    def __init__(self, env, gamma, lambda_, update_target_every_frames, **args):
        super().__init__(env, Networks.VNetwork, **args)
        self.target_network = deepcopy(self.network)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.update_target_every_frames = update_target_every_frames

    def train(self, batch_states, batch_new_states, batch_rewards, batch_dones):
        values = self.forward(batch_states)
        values_next = self.target_network(batch_new_states).squeeze(1)
        target = ((batch_rewards + self.gamma * values_next * (1 - batch_dones.int())).detach()).unsqueeze(1)
        loss = self.criterion(values, target)
        loss.backward()
        self.optimizer.step()
        for f in self.network.parameters():
            f.grad *= self.lambda_ * self.gamma
        return (target - values).detach()

class Actorcritic_actor(Actor_Agent):
    def __init__(self, env, lambda_, gamma, **args):
        super().__init__(env, **args)
        self.lambda_ = lambda_
        self.gamma = gamma

    def train(self, buffer: episodic_replay_buffer, critic: Actorcritic_critic):
        if (buffer.counter // self.num_envs) % critic.update_target_every_frames == 0:
            critic.target_network = deepcopy(critic.network)
        if buffer.is_ready_to_train() == False:
            return
        states, actions, rewards, dones = buffer.get_data_eligibility_traces()
        for _ in range(self.trains_every_frames):
            for i in range(0, len(states), self.batch_size):
                for j in range(states.shape[1]):
                    batch_dones = dones[i:i+self.batch_size, j]
                    batch_states = states[i:i+self.batch_size, j]
                    batch_actions = actions[i:i+self.batch_size, j]
                    batch_rewards = rewards[i:i+self.batch_size, j]
                    batch_new_states = states[i:i+self.batch_size, (j+1) % states.shape[1]]

                    # This part deals with the fact that our episode slice only sometimes ends with a done.
                    # If it does not, we can't use the next state for training, so we just skip it.
                    last_time_step = (j == states.shape[1] - 1)
                    if last_time_step:
                        batch_states = batch_states[batch_dones == True]
                        batch_actions = batch_actions[batch_dones == True]
                        batch_rewards = batch_rewards[batch_dones == True]
                        batch_new_states = batch_new_states[batch_dones == True]
                        batch_dones = batch_dones[batch_dones == True]
                        if len(batch_states) == 0:
                            continue

                    policy = self.forward(batch_states)
                    log_policy, entropy_of_policy = self.log_policy(policy, batch_actions)
                    error = critic.train(batch_states, batch_new_states, batch_rewards, batch_dones)
                    loss = torch.mean(-log_policy * error - self.entropy_regulization * entropy_of_policy)
                    loss.backward()
                    self.optimizer.step()
                    for f in self.network.parameters():
                        f.grad *= self.lambda_ * self.gamma
                self.optimizer.zero_grad()
                critic.optimizer.zero_grad()

class BaselineAgent(Agent):
    def __init__(self, env, **args):
        super().__init__(env, Networks.VNetwork, **args)

    def train(self, batch_states, batch_labels):
        values = self.forward(batch_states)
        loss = self.criterion(values, batch_labels.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return values.detach().squeeze(1)

class PPO_Agent(Actor_Agent):
    def __init__(self, env, epsilon_clip, gamma, **args):
        self.epsilon_clip = epsilon_clip
        self.gamma = gamma
        super().__init__(env, **args)

    def train(self, buffer: episodic_replay_buffer, base_line_model: BaselineAgent):
        if buffer.is_ready_to_train() == False:
            return
        states, actions, rewards, dones, log_probs = buffer.get_data_eligibility_traces()
        for _ in range(self.trains_every_frames):
            for i in range(0, len(states), self.batch_size):
                for j in range(states.shape[1]):
                    batch_states = states[i:i+self.batch_size, j]
                    batch_actions = actions[i:i+self.batch_size, j]
                    batch_rewards = rewards[i:i+self.batch_size, j]
                    batch_dones = dones[i:i+self.batch_size, j]
                    batch_log_probs = log_probs[i:i+self.batch_size, j]
                    batch_new_states = states[i:i+self.batch_size, (j+1) % states.shape[1]]

                    # This part deals with the fact that our episode slice only sometimes ends with a done.
                    # If it does not, we can't use the next state for training, so we just skip it.
                    last_time_step = (j == states.shape[1] - 1)
                    if last_time_step:
                        batch_states = batch_states[batch_dones == True]
                        batch_actions = batch_actions[batch_dones == True]
                        batch_rewards = batch_rewards[batch_dones == True]
                        batch_new_states = batch_new_states[batch_dones == True]
                        batch_log_probs = batch_log_probs[batch_dones == True]
                        batch_dones = batch_dones[batch_dones == True]
                        if len(batch_states) == 0:
                            continue

                    policy = self.forward(batch_states)
                    log_policy, entropy_of_policy = self.log_policy(policy, batch_actions)
                    ratio = torch.exp(log_policy - batch_log_probs)
                    values_next = base_line_model.forward(batch_new_states).detach().squeeze(1)
                    target = ((batch_rewards + self.gamma * values_next * (1 - batch_dones.int())).detach())
                    values = base_line_model.train(batch_states, target)
                    advantage = target - values
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage  
                    loss = torch.mean(-torch.min(surr1, surr2) - self.entropy_regulization * entropy_of_policy)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

class PPO_dual_network_Agent(Agent):
    def __init__(self, env, entropy_regulization, epsilon_clip, gamma, **args):
        continuous = args.get("continuous", False)
        super().__init__(env, Networks.Policy_advantage_network_continuous if continuous else Networks.Policy_advantage_network , **args)
        self.entropy_regulization = entropy_regulization
        self.continuous = continuous
        self.epsilon_clip = epsilon_clip
        self.gamma = gamma

    def train(self, states, actions, new_states, rewards, dones, log_probs):
        for _ in range(self.trains_every_frames):
            output = self.forward(states)
            policies, values = output[:, :-1], output[:, -1]
            log_policies, entropy_of_policies = self.log_policy(policies, actions)
            ratio = torch.exp(log_policies - log_probs)
            values_next = self.network.forward(new_states).detach()[:, -1]
            target = ((rewards.type(torch.float32) + self.gamma * values_next * (1 - dones.int())).detach())
            advantage = target - values
            surr1 = ratio * advantage.detach()
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage.detach()
            loss = torch.mean(-torch.min(surr1, surr2) - self.entropy_regulization * entropy_of_policies + 0.5 * advantage ** 2)
            print(ratio[0], advantage[0], entropy_of_policies[0])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def take_action(self, state, output_log_prob=False):
        output = self.forward(torch.tensor(state))
        policy, _ = output[:, :-1], output[:, -1]
        action = self.exploration.explore(policy)
        if output_log_prob:
            return action, self.log_policy(policy, torch.tensor(action))[0].detach()
        else:
            return action
        
    def log_policy(self, policy, actions):
        if self.continuous:
            return MultivariateNormal(policy, 0.2 * torch.eye(self.action_space)).log_prob(actions), - ((actions[:, 0] - 0.5) ** 2 + (torch.abs(actions[:, 1]) - 0.5) ** 2)
        else:
            log_policy = torch.log(policy)
            entropy_of_policy = -torch.sum(policy * log_policy, dim=1)
            return torch.gather(log_policy, 1, actions.unsqueeze(1)).squeeze(1), entropy_of_policy

