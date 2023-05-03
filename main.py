from typing import Callable
from helpers import run
from game_loops import deep_q_learn, eval
from Qexploration import Explorations, Exploration

class parameters:
    # general parameters
    name: str = "test2" # (name of the trained agent)
    train_loop: Callable = deep_q_learn # (choose between deep q learning, policy gradient, actor critic, eval)
    # (eval is used to evaluate a trained agent with the name specified above)

    # training parameters
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5

    # only used with a replay buffer
    trains_every_frames: int = 1
    train_after_frames: int = 50000
    buffer_size: int = 200000
    update_target_every_frames: int = 4000

    # agent parameters
    gamma: float = 0.995 # (only used with n-step learning)

    # for QAgent
    exploration: Explorations = Explorations.linearly_decaying_eps_greedy # (choose between epsilon greedy, greedy, etc.)
    # only used with epsilon greedy
    epsilon_start: float = 0.9
    epsilon_end: float = 0.1
    decay_period_of_epsilon: int = 200000

    # environment parameters
    env_name: str = "LunarLander-v2" # (choose between LunarLander-v2, CartPole-v1, etc.)
    render_mode: str = None # (human or None, only use human if very few num_envs (overwritten by eval)))
    continuous: bool = False
    num_envs: int = 16

    # extra parameters
    save_agent_every: int = 100000 # (how many frames before screenshot of agent and it's performance is saved)
    measure_performance_every: int = 40000 # (how many frames to accumulate into 1 point in plots)

if __name__ == "__main__":
    run(parameters)
