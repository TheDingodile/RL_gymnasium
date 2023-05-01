import matplotlib.pyplot as plt


class collector():
    def __init__(self, measure_performance_every, num_envs, **args) -> None:
        self.measure_performance_every = measure_performance_every
        self.num_envs = num_envs
        self.all_rewards = []
        self.all_dones = []
        self.dones = 0
        self.summed_reward = 0
        self.counter = 0

    def add_data(self):
        print("Played " + str(self.counter) + " frames, average reward per frame: " + str(self.summed_reward / self.measure_performance_every) + ", average reward per episode: " + str(self.summed_reward / max(self.dones, 1)))
        self.all_rewards.append(self.summed_reward)
        self.all_dones.append(max(self.dones, 1))
        self.summed_reward = 0
        self.dones = 0

    def collect(self, rewards, dones):
        self.summed_reward += sum(rewards)
        self.dones += sum(dones)

        for _ in range(self.num_envs):
            self.counter += 1

            if self.counter % self.measure_performance_every == 0:
                self.add_data()