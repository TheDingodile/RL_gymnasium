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
        return self.exploration.explore(self.forward(torch.tensor(state)))
    
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
        if buffer.is_ready_to_train() == False:
            return
        states, actions, labels, _ = buffer.get_data()
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

    def log_policy(self, policy, actions):
        if self.continuous:
            return MultivariateNormal(policy, 0.2 * torch.eye(self.action_space)).log_prob(actions), - ((actions[:, 0] - 0.5) ** 2 + (torch.abs(actions[:, 1]) - 0.5) ** 2)
        else:
            log_policy = torch.log(policy)
            entropy_of_policy = -torch.sum(policy * log_policy, dim=1)
            return torch.gather(log_policy, 1, actions.unsqueeze(1)).squeeze(1), entropy_of_policy

        

class BaselineAgent(Agent):
    def __init__(self, env, **args):
        super().__init__(env, Networks.VNetwork, **args)

    def train(self, batch_states, batch_labels):
        values = self.forward(batch_states)
        loss = self.criterion(values, batch_labels.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()