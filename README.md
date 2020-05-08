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
cd gridworldrl

# Create a conda env
conda env create -f environment.yml
# If you don't have a GPU use the environment_cpu.yml
# conda env create -f environment_cpu.yml
source activate deeprl_udacity

# Install pre-commit hooks
pre-commit install
```
Environment
--------------------------------------------------------------------------------
For this project, you will train an agent to navigate (and collect bananas!) in a large,
square world.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is
provided for collecting a blue banana. Thus, the goal of your agent is to collect as
many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with
ray-based perception of objects around the agent's forward direction. Given this
information, the agent has to learn how to best select actions. Four discrete actions
are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an
average score of +13 over 100 consecutive episodes.

Training
--------------------------------------------------------------------------------
The code was designed to automate the experiment pipeline.
In order to run an experiment please follow these steps:

1- Create a `config.yaml` file under `experiments` directory. The file structure needs
to be like the existing default configs that we used to run our proper experiments.
2- Run your experiment with the following code:
```bash
python train.py --config=experiments/config.yaml
```

Experiments
--------------------------------------------------------------------------------
We have conducted four experiments with different configs:
- Vanilla DQN
- Dueling DQN
- Double DQN
- Dueling Double DQN
- LR annealing with vanilla DQN

The results (plots and checkpoints) are under `results` directory.

Evaluation
--------------------------------------------------------------------------------