from typing import Callable
from helpers import run
from game_loops import deep_q_learn, eval, reinforce_learn, actor_critic_learn
from exploration import Explorations, Exploration

class parameters:
    # general parameters
    name: str = "deep_q_learn_cartpole" # (name of the trained agent)
    train_loop: Callable = deep_q_learn # (choose between deep q learning, policy gradient, actor critic, eval)
    exploration: Explorations = Explorations.linearly_decaying_eps_greedy # (How to choose action from output of agent)
    # (choose between epsilon greedy, greedy, multinomial (eg. If discrete REINFORCE), normal distribution (cont. REINFORCE), etc.)

    # training parameters
    batch_size: int = 512
    learning_rate: float = 1e-3 # (if multiple agents are used this learning rate is used for all of them)
    weight_decay: float = 1e-5
    trains_every_frames: int = 2

    # only used with a replay buffer
    train_after_frames: int = 20000
    buffer_size: int = 200000
    update_target_every_frames: int = 1000

    # only used with learning from episodes (eg. REINFORCE or actor critic)
    episodes_before_train: int = 1 # (How many episodes is played before we train on them. Higher number is faster but less efficient)
    baseline_model: bool = True # (whether to use a baseline or not (for advantage function estimation))

    # agent parameters
    gamma: float = 0.995 # (only used with td learning)
    lambda_: float = 0.9 # (only used with eligibility traces)
    sample_lengths: int = 100 # (only used with eligibility traces and only has effect when lambda_ > 0)
    entropy_regulization: float = 0.2 # (only used with policy agents)

    # only used with epsilon greedy
    epsilon_start: float = 0.9
    epsilon_end: float = 0.1
    decay_period_of_epsilon: int = 100000

    # environment parameters
    env_name: str = "CartPole-v1" # (choose between LunarLander-v2, CartPole-v1, etc.)
    render_mode: str = None # (human or None, only use human if very few num_envs and you want to see it play while training (is always human if eval mode))
    continuous: bool = False # (whether the environment is continuous or not)
    num_envs: int = 16 # (how many environments to run in parallel)

    # extra parameters
    save_agent_every: int = 40000 # (how many frames before screenshot of agent and it's performance is saved)
    measure_performance_every: int = 40000 # (how many frames to accumulate into 1 point in plots)

if __name__ == "__main__":
    run(parameters)
