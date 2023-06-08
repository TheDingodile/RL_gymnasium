import torch

class replay_buffer():
    def __init__(self, buffer_size, batch_size, log_probs=False, **args):
        self.log_probs = log_probs
        self.counter = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = [None for _ in range(self.buffer_size)]

    def save_data(self, data, truncated):
        for i in range(len(data[0])):
            if not truncated[i]:
                if self.log_probs:
                    self.buffer[self.counter % self.buffer_size] = (torch.tensor(data[0][i]), torch.tensor(data[1][i]), torch.tensor(data[2][i]), torch.tensor(data[3][i]), torch.tensor(data[4][i]), data[5][i])
                else:
                    self.buffer[self.counter % self.buffer_size] = (torch.tensor(data[0][i]), torch.tensor(data[1][i]), torch.tensor(data[2][i]), torch.tensor(data[3][i]), torch.tensor(data[4][i]))
                self.counter += 1

    def get_batch(self):
        idx = torch.randint(0, min(self.counter, self.buffer_size), (self.batch_size,))
        states = torch.stack([self.buffer[i][0] for i in idx])
        actions = torch.stack([self.buffer[i][1] for i in idx])
        rewards = torch.stack([self.buffer[i][2] for i in idx])
        next_states = torch.stack([self.buffer[i][3] for i in idx])
        dones = torch.stack([self.buffer[i][4] for i in idx])
        if self.log_probs:
            log_probs = torch.stack([self.buffer[i][5] for i in idx])
            return states, actions, rewards, next_states, dones, log_probs
        return states, actions, rewards, next_states, dones
    
class episodic_replay_buffer():
    def __init__(self, num_envs, episodes_before_train, gamma, sample_lengths, log_probs = False, **args):
        self.episodes_before_train = episodes_before_train
        episode_slots = max(num_envs + episodes_before_train, episodes_before_train * 2)    
        self.gamma = gamma
        self.free = list(range(episode_slots))
        self.dones = []
        self.currently_in_idx = {env_number: None for env_number in range(episode_slots)}
        self.buffer = [[] for _ in range(episode_slots)]
        self.counter = 0
        self.sample_lengths = sample_lengths
        self.log_probs = log_probs

    def save_data(self, data, truncated):
        for i in range(len(data[0])):
            if self.currently_in_idx[i] == None:
                self.currently_in_idx[i] = self.free.pop()

            # This if statement simply checks if we also want to save the log_probs (for PPO)
            if self.log_probs:
                d = (torch.tensor(data[0][i]), torch.tensor(data[1][i]), torch.tensor(data[2][i]).type(torch.float32), torch.tensor(data[3][i]), data[4][i])
            else:
                d = (torch.tensor(data[0][i]), torch.tensor(data[1][i]), torch.tensor(data[2][i]).type(torch.float32), torch.tensor(data[3][i]))
            
            self.buffer[self.currently_in_idx[i]].append(d)
            self.counter += 1

            if data[3][i] or truncated[i]:
                self.dones.append(self.currently_in_idx[i])
                self.currently_in_idx[i] = None

    def is_ready_to_train(self):
        return len(self.dones) >= self.episodes_before_train
    
    def get_data_monte_carlo(self):
        states = []
        actions = []
        labels = []
        for i in self.dones:
            self.free.append(i)
            episode = self.buffer[i]
            [labels.append(label) for label in self.calculate_values_monte_carlo(episode)]
            [states.append(episode[i][0]) for i in range(len(episode))]
            [actions.append(episode[i][1]) for i in range(len(episode))]
            self.buffer[i] = []
        self.dones = []
        return torch.stack(states), torch.stack(actions), torch.stack(labels)
    
    def get_data_eligibility_traces(self):
        states = []
        actions = []
        rewards = []
        dones = []
        if self.log_probs:
            log_probs = []

        sample_length = self.sample_lengths
        for k in self.dones:
            episode_length = len(self.buffer[k])
            if episode_length < sample_length:
                sample_length = episode_length

        for i in self.dones:
            self.free.append(i)
            episode = self.buffer[i]
            for j in range(0, len(episode), sample_length):
                j = min(j, len(episode) - sample_length)
                sample = episode[j:j + sample_length]
                states.append(torch.stack([i[0] for i in sample]))
                actions.append(torch.stack([i[1] for i in sample]))
                rewards.append(torch.stack([i[2] for i in sample]))
                dones.append(torch.stack([i[3] for i in sample]))
                if self.log_probs:
                    log_probs.append(torch.stack([i[4] for i in sample]))
            self.buffer[i] = []
        self.dones = []
        if self.log_probs:
            return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(dones), torch.stack(log_probs)
        else:
            return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(dones)

    def get_data_PPO(self):
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        for i in self.dones:
            self.free.append(i)
            episode = self.buffer[i]
            [states.append(episode[i][0]) for i in range(len(episode))]
            [actions.append(episode[i][1]) for i in range(len(episode))]
            [rewards.append(episode[i][2]) for i in range(len(episode))]
            [dones.append(episode[i][3]) for i in range(len(episode))]
            [log_probs.append(episode[i][4]) for i in range(len(episode))]
            self.buffer[i] = []
        self.dones = []
        return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(dones), torch.stack(log_probs)
    
    def calculate_values_monte_carlo(self, episode):
        rewards = [episode[i][2] for i in range(len(episode))]
        returns = [None for _ in range(len(episode))]
        returns[-1] = rewards[-1]
        for i in range(len(episode) - 2, -1, -1):
            returns[i] = rewards[i] + self.gamma * returns[i + 1]
        return returns