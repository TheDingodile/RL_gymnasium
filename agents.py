import torch.nn as nn
import torch.optim as optim
import torch
from replay_buffer import replay_buffer
from copy import deepcopy
from Qexploration import Exploration, Explorations


class Agent():
    def __init__(self, env, num_envs, exploration, learning_rate, weight_decay, train_after_frames, train_every_frames, continuous, **args):
        self.train_after_frames = train_after_frames
        self.train_every_frames = train_every_frames
        self.num_envs = num_envs
        self.state_space_size = env.observation_space.shape[1]
        self.action_space_size = env.action_space[0].n
        self.exploration = Exploration(exploration, self.action_space_size, **args)
        self.network = nn.Sequential(nn.Linear(self.state_space_size, 64), 
                                     nn.LeakyReLU(), nn.Linear(64, 64), 
                                     nn.LeakyReLU(), nn.Linear(64, 32), 
                                     nn.LeakyReLU(), nn.Linear(32, self.action_space_size))
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.trains = 0

    def take_action(self, state):
        return self.exploration.explore(self.forward(torch.tensor(state)))

    def try_train(self, buffer):
        self.trains += 1
        if buffer.counter < self.train_after_frames or self.trains % self.train_every_frames != 0:
            return
        self.train(buffer)
    
    def forward(self, state):
        return self.network(state)

class QAgent(Agent):
    def __init__(self, env, update_target_after_training, gamma, **args):
        super().__init__(env, **args)
        self.target_network = deepcopy(self.network)
        self.update_target_after_training = update_target_after_training
        self.gamma = gamma

    def train(self, buffer: replay_buffer):
        if (self.trains//self.train_every_frames) % self.update_target_after_training == 0:
            self.target_network = deepcopy(self.network)
        states, actions, rewards, next_states, dones = buffer.get_batch()
        q_values = torch.gather(self.forward(states), 1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = torch.gather(self.target_network(next_states), 1, torch.argmax(self.forward(next_states), 1).unsqueeze(1)).squeeze(1)
        target = (rewards + self.gamma * next_q_values * (1 - dones.int())).detach().float()
        loss = self.criterion(q_values, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

class Actor_Agent(Agent):
    def __init__(self, env, **args):
        super().__init__(env, **args)

    def train(self, target_agent):
        self.trains += 1
        states, actions, returns, next_states, dones = buffer.get_batch()
        policy = self.forward(states)
        baseline = target_agent.forward(states).flatten()
        log_policy = torch.log(policy)
        entropy_of_policy = -torch.sum(policy * log_policy, dim=1)
        advantage_function = returns - baseline
        loss = torch.mean(-torch.gather(log_policy, 1, actions.unsqueeze(1)).squeeze(1) * advantage_function - 0.5 * torch.sum(log_policy, dim=1) - entropy_of_policy)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()