This project allows you to use all the most well known Reinforcement Learning algorithms on the gymnasium environment.

## Installation

To install the project, you need to clone the repository and install the requirements.

```bash
git clone
cd RL_gymnasium
pip install -r requirements.txt
```
For windows, often the gymnasium can only install if you have installed [Swig](http://www.swig.org/download.html) and added it to your path. Another common problem is the error "Microsoft Visual C++ 14.0 or greater is required". Here you can install microsoft build tools and install the necessary dependencies.

## Usage

To use the project, you run the main.py file. Here you can choose between the different methods (deep_q, REINFORCE, etc..) and the different environments (CartPole, LunarLander, etc...). All the hyper-parameters that you can tune is also in the main.py file. In general, it should be fairly straight forward to choose the parameters you want, as most of them are classes that displays their options or are just a number. You can go to the train_agents folder to have a look at the experiments that has already been run.

Here is a list of all the hyper-parameters and what they do:

- **name**: 
It is the name you give the experiment. The experiment will be saved in the trained_agents folder. If you choose a name which already exists it will be overwritten. A folder will be created with the model you are training, some performance graphs and a txt file of the used hyper-parameters. However, if you use eval mode (see below) the model and performance graphs will be loaded from the trained_agents folder instead.

- **train_loop**:
This parameter decides what kind of RL algorithm you want to use. It can also be set to eval which loads the folder with the name given and you can see it play.

- **exploration**:
This parameter decides how you want to choose your action. The reason for the name exploration is because in deep-q learning it often is associated with what exploration method you use (eg. epsilon greedy, greedy). However, in other train_loops it should be seen as more "how do i sample from the output of my model". So for a discrete REINFORCE train_loop you could choose Multinomial, and for a continious REINFORCE train_loop you could choose a normal distribution.

- **batch_size**:
When training, how big of a batch size do you want to use (if multiple models same batch_size is used).

- **learning_rate**:
The learning rate of the model(s).

- **weight_decay**:
The weight decay used in the ADAM optimizer(s).

- **trains_every_frame**:
How many times to you want to train for every step taken in the environment. It is recommended to run multiple environments in parallel (see below), and so it sometimes makes sense to train more than once per step.

- **train_after_frames**:
How many frames do you want to wait before you start training. This is useful if you want to collect some data before you start training. So you have some data from an approximately random policy before you start training, and so the data you train on when using a replay buffer is not too correlated.

- **buffer_size**:
Some methods uses a "normal" replay buffer (deep_q_learning, soft_actorcritic), and this parameter decides how big the buffer should be.

- **update_target_every_frames**:
Some methods uses a target network (eg. deep_q_learning), and this parameter decides how many frames you need to see before you update the target network.

- **episodes_before_train**:
Methods like REINFORCE finishes episodes before it can train. This parameter decided how many episodes you want to finish before you train on them. Usually this would be set to the number of environments you run in parallel, but it can be more.

-  **baseline_model**:
Some methods uses a baseline (eg. REINFORCE), and this parameter decides that. 

- **gamma**:
The discount factor used in the RL algorithms.

- **lambda_**:
The lambda used in the algorithms with eligibility traces.

- **sample_length**:
When training with eligibility traces samples are drawn of this length.

- **entropy_regulization**:
Some methods uses entropy regulization (REINFORCE, PPO, actor_critic, soft_actor_critic_learn), and this parameter decides how much of that entropy regulization you want to use.

- **epsilon_clip**:
For PPO you can clip the ratio between the new and old policy. This parameter decides the ratio of when you want to clip it.

- **epsilon_start**:
For epsilon greedy exploration, this parameter decides the starting epsilon.

- **epsilon_end**:
For epsilon greedy exploration, this parameter decides the ending epsilon.

- **decay_period_of_epsilon**:
For epsilon greedy exploration, this parameter decides how many frames you want to decay the epsilon over. (The decay is linear)

- **env_name**:
This parameter decides what environment you want to use. It is the name of the gymnasium environment.

- **render_mode**:
This parameter decides if you want to render the environment or not. It can be set to "human" or "None". Using "None" will not render the environment, but it will speed up the training significantly.

- **num_envs**:
This parameter decides how many environments you want to run in parallel.

- **save_every_frames**:
This parameter decides how often you want to save the model (and save performance for plots).

- **measure_performance_every**:
This parameter decides how often you want to measure the performance of the model. So you can get a smoother plot and not save too much data.


In general, not all combinations of hyper-parameters is able to run. For example, not all environments has a continious version, so choosing continuous on a non-continious version might crash the program. Likewise, choosing you want to sample actions from a normal-distribution given you have a discrete action space will also crash the program. However, the program should be able to handle most of the combinations.

## Details
The PPO implementation follows the original paper https://arxiv.org/pdf/1707.06347.pdf with regulization entropy. There are two versions implemented. The first uses seperate networks for the actor and advantage estimates, the second version uses a dual net.

The soft actor critic implementation follows this https://spinningup.openai.com/en/latest/algorithms/sac.html. Which uses two Q-functions rather than a V-function as in the original paper which is found here https://spinningup.openai.com/en/latest/algorithms/sac.html.

The project is build up in the following way:

- **main.py**:
Is the front-end of the project. Here you can choose the hyper-parameters and what you want to train. It simply runs the "run" function with your class of arguments.

- **helpers.py**:
This file has the run function, which just takes care of the arguments and runs the correct function. It also saves the hyper-parameters in a txt file.
helpers also have other helpful function to make the code more readable.

- **game_loops.py**:
Here all the gameloops are. Most of them look similar (have a take_action, step, train, collect), but they vary to some extent.

- **agent.py**:
This file has all the different agents and their training methods. There is a superclass, Agent, that holds much of the attributes all the agents has in common. Then there is a subclass of this class called Actor_agent, which carries the common charateristics of the policy gradient methods. I tried to fit all the other agents under these two umbrella classes as much as possible. An agent has a network and an exploration tool which enables it to sample actions. Not all agents use their exploration tool (eg. the Baseline agent is used only to evaluate). 

short description of the agents:
1. **QAgent**:
This agent is used in Q learning and outputs Q values for each actions. It is trained with double Q learning.
2. **REINFORCE_Agent**:
Trained via the standard REINFORCE algorithm using monte carlo returns. It can use a baseline network for advantage estimation to reduce variance.
3. **Actorcritic_critic**:
This is the critic for the actorcritic algorithm. It is similar to the QAgent, but it can use eligibility traces. 


- **exploration.py**:
This file holds a class which holds all the exploration methods. This makes it easy to choose what exploration method you want to use. There are the epsilon greedy variants for the Q-learning, and  for policy gradient methods the multinomial for discrete and normal distribution with and without standard deviation for continuous.


Right now for continious environments a normal distribution is used. It has a hardcoded diagonal covariance matrix with a standard deviation of 0.1. 

PPO_batch is hardcoded to run 40000 frames before training 400 times with batch size.

Some hyper-parameter configurations could also cause the program to crash. Eg. A too high learning rate could make some models diverge. This is especially a potential issue in the monte carlo methods, as the variance is quite high here.



