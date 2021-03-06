# Navigation-Udacity-Deep-RL-project
Install
--------------------------------------------------------------------------------
We use:
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
  to setup the environment,
- and python 3.7

Setup our environment:
```bash
conda --version

# Clone the repo
git clone https://github.com/KhalilGorsan/Navigation-Udacity-Deep-RL-project.git
cd Navigation-Udacity-Deep-RL-project

# Create a conda env
conda env create -f environment.yml

source activate deeprl_udacity

# Install pre-commit hooks
pre-commit install
```
Don't forget to add The Banana.app unity environment in the root of the project.

To install an already built environment for you, you can download it from one
of the links below. You need only to select the environment that matches your operating
system and unzip it:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Environment
--------------------------------------------------------------------------------
In this project, we will train an agent to navigate (and collect bananas!) in a large,
square world.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is
provided for collecting a blue banana. Thus, the goal of your agent is to collect as
many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with
ray-based perception of objects around the agent's forward direction. Given this
information, the agent has to learn how to best select actions. Four discrete actions
are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an
average score of +13 over 100 consecutive episodes.

Training
--------------------------------------------------------------------------------
The code was designed to automate the experiment pipeline.
In order to run an experiment please follow these steps:

1. Create a `config.yaml` file under `experiments` directory. The file structure needs
to be like the existing default configs that we used to run our proper experiments.

2. Run your experiment with the following code:
```bash
python train.py --config=experiments/config.yaml
```

Experiments
--------------------------------------------------------------------------------
We have conducted five experiments with different configs:
- Vanilla DQN
- Dueling DQN
- Double DQN
- Dueling Double DQN
- LR annealing with vanilla DQN

The results (plots and checkpoints) are under `results` directory.

Evaluation
--------------------------------------------------------------------------------
Below is a Double DQN agent interacting with the environment. Actions are selected using
the trained policy.

![](banana.gif)

You can reproduce it by running the following code:
```bash
python evaluate.py --eval_config=experiments/double_dqn.yaml
```