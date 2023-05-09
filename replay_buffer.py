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
    def __init__(self, num_envs, episodes_before_train, gamma, sample_lengths, **args):
        self.gamma = gamma
        self.episodes_before_train = max(num_envs, episodes_before_train)
        self.free = list(range(self.episodes_before_train * 2))
        self.dones = []
        self.currently_in_idx = {env_number: None for env_number in range(self.episodes_before_train)}
        self.buffer = [[] for _ in range(self.episodes_before_train * 2)]
        self.counter = 0
        self.sample_lengths = sample_lengths

    def save_data(self, data, truncated):
        for i in range(len(data[0])):
            if self.currently_in_idx[i] == None:
                self.currently_in_idx[i] = self.free.pop()
            if not truncated[i]:
                self.buffer[self.currently_in_idx[i]].append((torch.tensor(data[0][i]), torch.tensor(data[1][i]), torch.tensor(data[2][i]), torch.tensor(data[3][i]), torch.tensor(data[4][i])))
                self.counter += 1
            if data[4][i]:
                self.dones.append(self.currently_in_idx[i])
                self.currently_in_idx[i] = None

    def is_ready_to_train(self):
        return len(self.dones) >= self.episodes_before_train
    
    def get_data_monte_carlo(self):
        states = []
        actions = []
        new_states = []
        labels = []
        for i in self.dones:
            self.free.append(i)
            episode = self.buffer[i]
            [labels.append(label.type(torch.float32)) for label in self.calculate_values_monte_carlo(episode)]
            [states.append(episode[i][0]) for i in range(len(episode))]
            [actions.append(episode[i][1]) for i in range(len(episode))]
            [new_states.append(episode[i][3]) for i in range(len(episode))]
            self.buffer[i] = []
        self.dones = []
        return torch.stack(states), torch.stack(actions), torch.stack(labels), torch.stack(new_states)
    
    def get_data_eligibility_traces(self):
        states = []
        actions = []
        new_states = []
        labels = []
        for i in self.dones:
            self.free.append(i)
            episode = self.buffer[i]
            [labels.append(label.type(torch.float32)) for label in self.calculate_values_eligibility_traces(episode)]
            [states.append(episode[i][0]) for i in range(len(episode))]
            [actions.append(episode[i][1]) for i in range(len(episode))]
            [new_states.append(episode[i][3]) for i in range(len(episode))]
            self.buffer[i] = []
        self.dones = []
        return torch.stack(states), torch.stack(actions), torch.stack(labels), torch.stack(new_states)

    
    def calculate_values_monte_carlo(self, episode):
        rewards = [episode[i][2] for i in range(len(episode))]
        returns = [None for _ in range(len(episode))]
        returns[-1] = rewards[-1]
        for i in range(len(episode) - 2, -1, -1):
            returns[i] = rewards[i] + self.gamma * returns[i + 1]
        return returns