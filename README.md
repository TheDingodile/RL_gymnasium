This project allows you to use all the most well known Reinforcement Learning algorithms on the gymnasium environment.

## Installation

To install the project, you need to clone the repository and install the requirements.

```bash
git clone
cd RL_gymnasium
pip install -r requirements.txt
```
For windows, often the gymnasium can only install if you have installed [Swig](http://www.swig.org/download.html) and added it to your path.

## Usage

To use the project, you run the main.py file. Here you can choose between the different methods (deep_q, REINFORCE, etc..) and the different environments (CartPole, MountainCar, etc...). All the hyper-parameters that you can tune is also in the main.py file. In general, it should be fairly straight forward to choose the parameters you want, as most of them are classes that displays their options or are just a number.

Here is a list of all the hyper-parameters and what they do:

- **name**: 
It is the name you give the experiment. The experiment will be saved in the trained_agents folder. If you choose a name which already exists it will be overwrittin. A folder will be created with the model you are training, some performance graphs and a txt file of the used hyper-parameters. However, if you use eval mode (see below) the model and performance graphs will be loaded from the trained_agents folder.

- **train_loop**:
This parameter decides what kind of RL algorithm you want to use. It can also be set to eval which loads the folder with the name given and you can see it play.

- **exploration**:
This parameter decides how you want to choose your action. The reason for the name exploration is because in deep-q learning it often is associated with what exploration method you use (eg. epsilon greedy, greedy). However, in other train_loops it should be seen as more "how do i sample from the output of my model". So for a discrete REINFORCE train_loop you could choose Multinomial, and for a continious REINFORCE train_loop you could choose a normal distribution.

- **batch_size**:
When training, how big of a batch size do you want to use.


In general, not all combinations of hyper-parameters is able to run. For example, not all environments has a continious version, so choosing continious on a non-continious version might crash the program. Likewise, choosing you want to sample actions from a normal-distribution given you have a discrete action space will also crash the program. However, the program should be able to handle most of the combinations.