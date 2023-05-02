import torch
import numpy as np

class replay_buffer():
    def __init__(self, buffer_size, batch_size, **args):
        self.counter = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = [None for _ in range(self.buffer_size)]
        self.states = []
        self.actions = []
        self.rewards = []

    def save_data(self, data, truncated):
        for i in range(len(data[0])):
            if not truncated[i]:
                self.buffer[self.counter % self.buffer_size] = (torch.tensor(data[0][i]), torch.tensor(data[1][i]), torch.tensor(data[2][i]), torch.tensor(data[3][i]), torch.tensor(data[4][i]))
                self.counter += 1

    def get_batch(self):
        idx = torch.randint(0, min(self.counter, self.buffer_size), (self.batch_size,))
        states = torch.stack([self.buffer[i][0] for i in idx])
        actions = torch.stack([self.buffer[i][1] for i in idx])
        rewards = torch.stack([self.buffer[i][2] for i in idx])
        next_states = torch.stack([self.buffer[i][3] for i in idx])
        dones = torch.stack([self.buffer[i][4] for i in idx])
        return states, actions, rewards, next_states, dones
