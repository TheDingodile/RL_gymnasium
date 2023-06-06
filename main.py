from typing import Callable
from helpers import run
from game_loops import deep_q_learn, eval, reinforce_learn, actor_critic_learn, PPO_learn, PPO_learn_batches, soft_actor_critic_learn
from exploration import Explorations, Exploration

class parameters:
    # general parameters
    name: str = "Soft_ActorCritic_lunarlander" # (name of the trained agent)
    train_loop: Callable = soft_actor_critic_learn # (choose between deep q learning, policy gradient, actor critic, eval)
    exploration: Explorations = Explorations.normal_distribution # (How to choose action from output of agent)
    # (choose between epsilon greedy, greedy, multinomial (eg. If discrete REINFORCE), normal distribution (cont. REINFORCE), etc.)

    # training parameters
    batch_size: int = 256 # (if multiple agents are used this batch size is used for all of them)
    learning_rate: float = 1e-4 # (if multiple agents are used this learning rate is used for all of them)
    weight_decay: float = 1e-5
    trains_every_frames: int = 1

    # only used with a replay buffer
    train_after_frames: int = 40000
    buffer_size: int = 500000
    update_target_every_frames: int = 500

    # only used with learning from episodes (eg. REINFORCE or actor critic)
    episodes_before_train: int = 32 # (How many episodes is played before we train on them. Higher number is faster but less efficient)
    baseline_model: bool = True # (whether to use a baseline or not (for advantage function estimation))

    # agent parameters
    gamma: float = 0.995 # (only used with td learning)
    lambda_: float = 0.8 # (only used with eligibility traces)
    sample_lengths: int = 30 # (only used with eligibility traces and only has effect when lambda_ > 0)
    entropy_regulization: float = 3 # (only used with policy agents)
    epsilon_clip: float = 0.1 # (only used with PPO)

    # only used with epsilon greedy
    epsilon_start: float = 0.9
    epsilon_end: float = 0.1
    decay_period_of_epsilon: int = 200000

    # environment parameters
    env_name: str = "LunarLander-v2" # (choose between LunarLander-v2, CartPole-v1, etc.)
    render_mode: str = None # (human or None, only use human if very few num_envs and you want to see it play while training (is always human if eval mode))
    continuous: bool = True # (whether the environment is continuous or not)
    num_envs: int = 16 # (how many environments to run in parallel)

    # extra parameters
    save_agent_every: int = 40000 # (how many frames before screenshot of agent and it's performance is saved)
    measure_performance_every: int = 40000 # (how many frames to accumulate into 1 point in plots)

if __name__ == "__main__":
    run(parameters)
