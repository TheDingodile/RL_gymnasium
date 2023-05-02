from typing import Callable
from helpers import run
from game_loops import deep_q_learn
from Qexploration import Explorations, Exploration

class parameters:
    # general parameters
    name: str = "deep_q_learn_cartpole" # (name of the trained agent)
    train_loop: Callable = deep_q_learn # (choose between deep q learning, policy gradient, actor critic, etc.)

    # training parameters
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # only used with a replay buffer
    train_every_frames: int = 2
    train_after_frames: int = 20000
    buffer_size: int = 200000
    update_target_after_training: int = 1000

    # agent parameters
    gamma: float = 0.995 # (only used with n-step learning)

    # for QAgent
    exploration: Explorations = Explorations.linearly_decaying_eps_greedy # (choose between epsilon greedy, greedy, etc.)
    # only used with epsilon greedy
    epsilon_start: float = 0.9
    epsilon_end: float = 0.1
    decay_period_of_epsilon: int = 100000

    # environment parameters
    env_name: str = "CartPole-v1" # (choose between LunarLander-v2, CartPole-v1, etc.)
    render_mode: str = None # (human or None, only use human if very few num_envs)
    continuous: bool = False
    num_envs: int = 16

    # extra parameters
    save_agent_every: int = 50000 # (how many frames before screebshot of agent and it's performance is saved)
    measure_performance_every: int = 10000 # (how many frames to accumulate into 1 point in plots)

if __name__ == "__main__":
    run(parameters)
