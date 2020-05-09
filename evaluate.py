# load the weights from file
import time

import torch
from absl import app, flags

from core import BananaWrapper
from dqn_agent import Agent
from utils import extract_configs

flags.DEFINE_multi_string(
    "eval_config", None, "Filename containing the config for dqn agent."
)
FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv

    configs = extract_configs(*FLAGS.eval_config)
    # instantiate the Banana env
    env = BananaWrapper(file_name="./Banana")
    state_size = env.observation_size
    action_size = env.action_size
    # instantiate agent object
    agent = Agent(state_size=state_size, action_size=action_size, configs=configs)

    # load trained model
    agent.qnetwork_local.load_state_dict(
        torch.load("results/checkpoints/DoubleDQN.pth")
    )

    horizon = 1000
    episodes = 5
    for _ in range(episodes):
        state = env.reset()
        for _ in range(horizon):
            # Perform action given the trained policy
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            time.sleep(0.05)
            if done:
                break
            state = next_state

    # close env
    env.close()


if __name__ == "__main__":
    flags.mark_flag_as_required("eval_config")
    app.run(main)
