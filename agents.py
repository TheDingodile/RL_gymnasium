import torch.nn as nn
import torch.optim as optim
import torch
from replay_buffer import replay_buffer
from copy import deepcopy
from Qexploration import Exploration, Explorations


class Agent():
    def __init__(self, env, learning_rate, weight_decay, **args):
        self.state_space_size = env.observation_space.shape[1]
        self.action_space_size = env.action_space[0].n
        self.network = nn.Sequential(nn.Linear(self.state_space_size, 64), 
                                     nn.LeakyReLU(), nn.Linear(64, 64), 
                                     nn.LeakyReLU(), nn.Linear(64, 32), 
                                     nn.LeakyReLU(), nn.Linear(32, self.action_space_size))
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def take_action(self, state, **args):
        raise NotImplementedError

    def train(self, **args):
        raise NotImplementedError


class QAgent(Agent):
    def __init__(self, env, exploration, num_envs, update_target_after_training, train_after_frames, train_every_frames, gamma, learning_rate, weight_decay, **args):
        super().__init__(env, learning_rate, weight_decay, **args)
        self.exploration = Exploration(exploration, self.action_space_size, **args)
        self.target_network = deepcopy(self.network)
        self.num_envs = num_envs
        self.update_target_after_training = update_target_after_training
        self.train_after_frames = train_after_frames
        self.gamma = gamma
        self.trains = 0
        self.train_every_frames = train_every_frames

    def take_action(self, state):
        return self.exploration.explore(self.forward(torch.tensor(state)))

    def forward(self, state):
        return self.network(state)

    def train(self, buffer: replay_buffer):
        if buffer.counter < self.train_after_frames:
            return
        if buffer.counter//self.num_envs % self.update_target_after_training == 0:
            self.target_network = deepcopy(self.network)

        self.trains += 1
        if self.trains % self.train_every_frames != 0:
            return
        
        states, actions, rewards, next_states, dones = buffer.get_batch()
        q_values = torch.gather(self.forward(states), 1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = torch.gather(self.target_network(next_states), 1, torch.argmax(self.forward(next_states), 1).unsqueeze(1)).squeeze(1)
        target = (rewards + self.gamma * next_q_values * (1 - dones.int())).detach().float()
        loss = self.criterion(q_values, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
