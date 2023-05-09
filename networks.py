from enum import Enum
import torch
import torch.nn as nn

class Networks(Enum):
    QNetwork = 0
    VNetwork = 1
    Actor_Network = 2

class Network:
    def __init__(self, network: Networks, env, **args):
        self.action_space = env.action_space[0].n
        self.state_space_size = env.observation_space.shape[1]
        self.base_network = nn.Sequential(nn.Linear(self.state_space_size, 64), 
                                        nn.LeakyReLU(), nn.Linear(64, 32), 
                                        nn.LeakyReLU())
        if network == Networks.QNetwork:
            self.network = nn.Sequential(self.base_network, nn.Linear(32, self.action_space))
        elif network == Networks.VNetwork:
            self.network = nn.Sequential(self.base_network, nn.Linear(32, 1))
        elif network == Networks.Actor_Network:
            self.network = nn.Sequential(self.base_network, nn.Linear(32, self.action_space), nn.Softmax(dim=1))
