import torch
from enum import Enum
from torch.distributions.multivariate_normal import MultivariateNormal

class Explorations(Enum):
    greedy = 0
    eps_greedy = 1
    boltzmann = 2
    eps_boltzmann = 3
    linearly_decaying_eps_greedy = 4
    softmax = 5
    normal_distribution = 6

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
        elif exploration == Explorations.softmax:
            self.explore = self.softmax
        elif exploration == Explorations.normal_distribution:
            self.explore = self.normal_distribution
        self.counter = 0
        

    @property
    def epsilon(self):
        return max(self.epsilon_end, self.epsilon_start - self.counter / self.decay_period_of_epsilon * (self.epsilon_start - self.epsilon_end))

    def explore(self, values):
        pass
        
    def greedy(self, values):
        return torch.argmax(values, dim=1).tolist()

    def eps_greedy(self, values):
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.action_space, (values.shape[0],)).tolist()
        return self.greedy(values)
    
    def linearly_decaying_eps_greedy(self, values):
        self.counter += values.shape[0]
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.action_space, (values.shape[0],)).tolist()
        return self.greedy(values)
    
    def softmax(self, values):
        return torch.multinomial(values, 1).squeeze(1).tolist()
    
    def normal_distribution(self, values):
        return MultivariateNormal(values, 0.2 * torch.eye(self.action_space)).sample().tolist()