import torch
from enum import Enum

class Explorations(Enum):
    greedy = 0
    eps_greedy = 1
    boltzmann = 2
    eps_boltzmann = 3
    linearly_decaying_eps_greedy = 4

class Exploration:
    def __init__(self, exploration: Explorations, action_space, epsilon_start, epsilon_end, decay_period_of_epsilon, **args):
        self.action_space = action_space
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_period_of_epsilon = decay_period_of_epsilon
        if exploration == Explorations.greedy:
            self.explore = self.greedy
        elif exploration == Explorations.eps_greedy:
            self.explore = self.eps_greedy
        elif exploration == Explorations.linearly_decaying_eps_greedy:
            self.explore = self.linearly_decaying_eps_greedy
        self.counter = 0
        

    @property
    def epsilon(self):
        return max(self.epsilon_end, self.epsilon_start - self.counter / self.decay_period_of_epsilon * (self.epsilon_start - self.epsilon_end))

    def explore(self, state):
        pass
        
    def greedy(self, state):
        return torch.argmax(state, dim=1).tolist()

    def eps_greedy(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.action_space, (state.shape[0],)).tolist()
        return self.greedy(state)
    
    def linearly_decaying_eps_greedy(self, state):
        self.counter += state.shape[0]
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.action_space, (state.shape[0],)).tolist()
        return self.greedy(state)