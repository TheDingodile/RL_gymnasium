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

class Multiply(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha =  alpha
    
    def forward(self, x):
        return x * self.alpha

class Network:
    def __init__(self, network: Networks, env, continuous, **args):
        self.action_space = get_action_space(env, continuous)
        self.state_space_size = env.observation_space.shape[1]
        last_hidden_layer_size = 32
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
            mult = (np.max(env.action_space.high) - np.min(env.action_space.low[0]))/2
            self.network = nn.Sequential(self.base_network, nn.Linear(last_hidden_layer_size, self.action_space), nn.Tanh(), Multiply(mult))    
