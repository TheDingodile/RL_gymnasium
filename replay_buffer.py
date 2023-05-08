import torch
import numpy as np

class replay_buffer():
    def __init__(self, buffer_size, batch_size, **args):
        self.counter = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = [None for _ in range(self.buffer_size)]

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
    
class episodic_replay_buffer():
    def __init__(self, num_envs, episodes_before_train, gamma, **args):
        self.gamma = gamma
        self.episodes_before_train = max(num_envs, episodes_before_train)
        self.free = list(range(self.episodes_before_train * 2))
        self.currently_in_idx = {env_number: None for env_number in range(self.episodes_before_train)}
        self.buffer = [[] for _ in range(self.episodes_before_train * 2)]

    def save_data(self, data, truncated):
        for i in range(len(data[0])):
            currently_in = self.currently_in_idx[i]
            if currently_in == None:
                currently_in = self.free.pop()

            if not truncated[i]:
                self.buffer[currently_in].append((torch.tensor(data[0][i]), torch.tensor(data[1][i]), torch.tensor(data[2][i]), torch.tensor(data[3][i]), torch.tensor(data[4][i])))
            if data[4][i]:
                self.currently_in_idx[i] = None

    def is_ready_to_train(self):
        return len(self.free) <= self.episodes_before_train
    
    def get_batch(self):
        for i in range(self.episodes_before_train):
            if i not in self.free:
                self.free.append(i)
                for j in range(len(self.buffer[i])):
                    self.buffer[i][j] = None
                self.buffer[i] = []

                idx = torch.randint(0, len(self.buffer[i]), (1,))
                states = torch.stack([self.buffer[i][idx][0] for i in range(self.episodes_before_train)])
                actions = torch.stack([self.buffer[i][idx][1] for i in range(self.episodes_before_train)])
                rewards = torch.stack([self.buffer[i][idx][2] for i in range(self.episodes_before_train)])
                next_states = torch.stack([self.buffer[i][idx][3] for i in range(self.episodes_before_train)])
                dones = torch.stack([self.buffer[i][idx][4] for i in range(self.episodes_before_train)])
                return states, actions, rewards, next_states, dones
            
    def calculate_values(self, episode):
        # calculate discounted returns from an episode
        rewards = [episode[i][2] for i in range(len(episode))]
        returns = [0 for _ in range(len(episode))]
        returns[-1] = rewards[-1]
        for i in range(len(episode) - 2, -1, -1):
            returns[i] = rewards[i] + self.gamma * returns[i + 1]
        return returns
    
    def get_batch(self):