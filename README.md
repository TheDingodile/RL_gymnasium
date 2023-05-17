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
    
```python

The name parameter is the name you give the experiment. A folder will be created with the model you are training, some performance graphs and a txt file of the used hyper-parameters.

the train-loop parameter decides what kind of RL algorithm you want to use. It should be 