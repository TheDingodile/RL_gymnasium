from typing import Callable
from helpers import run
from game_loops import deep_q_learn, eval, reinforce_learn
from exploration import Explorations, Exploration

class parameters:
    # general parameters
    name: str = "test" # (name of the trained agent)
    train_loop: Callable = deep_q_learn # (choose between deep q learning, policy gradient, actor critic, eval)
    exploration: Explorations = Explorations.linearly_decaying_eps_greedy # (How to choose action from output of agent)
    # (choose between epsilon greedy, greedy, multinomial (eg. If discrete REINFORCE), normal distribution (cont. REINFORCE), etc.)

    # training parameters
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    trains_every_frames: int = 1

    # only used with a replay buffer
    train_after_frames: int = 50000
    buffer_size: int = 200000
    update_target_every_frames: int = 2000

    # only used with learning from episodes (eg. REINFORCE or actor critic)
    episodes_before_train: int = 16 # (How many episodes is played before we train on them. If set to lower than num_envs, it will be set to num_envs)
    baseline_model: bool = True # (whether to use a baseline or not (for advantage function estimation))

    # agent parameters
    gamma: float = 0.995 # (only used with td learning)
    lambda_: float = 0.9 # (only used with eligibility traces)
    sample_lengths: int = 30 # (only used with eligibility traces)
    entropy_regulization: float = 5 # (only used with reinforce)

    # only used with epsilon greedy
    epsilon_start: float = 0.9
    epsilon_end: float = 0.1
    decay_period_of_epsilon: int = 200000

    # environment parameters
    env_name: str = "LunarLander-v2" # (choose between LunarLander-v2, CartPole-v1, etc.)
    render_mode: str = None # (human or None, only use human if very few num_envs (is always human if eval mode)))))
    continuous: bool = False
    num_envs: int = 16

    # extra parameters
    save_agent_every: int = 40000 # (how many frames before screenshot of agent and it's performance is saved)
    measure_performance_every: int = 40000 # (how many frames to accumulate into 1 point in plots)

if __name__ == "__main__":
    run(parameters)
