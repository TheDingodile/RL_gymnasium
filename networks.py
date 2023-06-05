from enum import Enum
import torch
import torch.nn as nn
from helpers import get_action_space
import numpy as np

class Networks(Enum):
    QNetwork = 0
    VNetwork = 1
    Actor_Network = 2
    Normal_distribution = 3
    Policy_advantage_network = 4
    Policy_advantage_network_continuous = 5
    SAC_q_network = 6

class Multiply(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha =  alpha
    
    def forward(self, x):
        return x * self.alpha
    
class ParallelModule(nn.Sequential):
    def __init__(self, *args):
        super(ParallelModule, self).__init__( *args )

    def forward(self, input):
        output = []
        for module in self:
            output.append(module(input))
        return torch.cat(output, dim=1)

class Network:
    def __init__(self, network: Networks, env, continuous, **args):
        self.action_space = get_action_space(env, continuous)
        self.state_space_size = env.observation_space.shape[1]
        last_hidden_layer_size = 32
        mult = (np.max(env.action_space.high) - np.min(env.action_space.low[0]))/2
        self.base_network = nn.Sequential(nn.Linear(self.state_space_size, 64), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(64, last_hidden_layer_size),
                                        nn.LeakyReLU())
        if network == Networks.QNetwork:
            self.network = nn.Sequential(self.base_network, nn.Linear(last_hidden_layer_size, self.action_space))
        elif network == Networks.VNetwork:
            self.network = nn.Sequential(self.base_network, nn.Linear(last_hidden_layer_size, 1))
        elif network == Networks.Actor_Network:
            self.network = nn.Sequential(self.base_network, nn.Linear(last_hidden_layer_size, self.action_space), nn.Softmax(dim=1))
        elif network == Networks.Normal_distribution:
            self.network = nn.Sequential(self.base_network, nn.Linear(last_hidden_layer_size, self.action_space), nn.Tanh(), Multiply(mult))
        elif network == Networks.Policy_advantage_network:
            self.network = nn.Sequential(self.base_network, ParallelModule(nn.Sequential(nn.Linear(last_hidden_layer_size, self.action_space), nn.Softmax(dim=1)), nn.Linear(last_hidden_layer_size, 1)))
        elif network == Networks.Policy_advantage_network_continuous:   
            self.network = nn.Sequential(self.base_network, ParallelModule(nn.Sequential(nn.Linear(last_hidden_layer_size, self.action_space), nn.Tanh(), Multiply(mult)), nn.Linear(last_hidden_layer_size, 1)))
        elif network == Networks.SAC_q_network:
            self.network = nn.Sequential(nn.Linear(self.state_space_size + self.action_space, 64), nn.LeakyReLU(), nn.Linear(64, last_hidden_layer_size), nn.LeakyReLU(), nn.Linear(last_hidden_layer_size, 1))

