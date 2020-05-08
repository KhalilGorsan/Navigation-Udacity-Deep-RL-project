from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from absl import app, flags

from core import BananaWrapper
from dqn_agent import Agent
from utils import extract_configs

flags.DEFINE_multi_string(
    "config", None, "Filename containing the config for dqn agent."
)
FLAGS = flags.FLAGS


def dqn(
    env,
    agent,
    label,
    n_episodes=2000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.001,
    eps_decay=0.995,
):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    solved = False
    previous_score_window = 0
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for _ in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_window)
            ),
            end="",
        )
        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )
        if np.mean(scores_window) >= 13.0 and not solved:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(scores_window)
                )
            )
            solved = True
        # Keep the checkpoint with the best score
        if np.mean(scores_window) > previous_score_window:
            torch.save(
                agent.qnetwork_local.state_dict(),
                "results/checkpoints/" + str(label) + ".pth",
            )
            previous_score_window = np.mean(scores_window)

    return scores


def main(unused_argv):
    del unused_argv

    configs = extract_configs(*FLAGS.config)
    exp_id = configs["exp_id"]
    training = configs["training"]
    label = configs["agent"]["name"]

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    env = BananaWrapper(file_name="./Banana")

    state_size = env.observation_size
    action_size = env.action_size
    agent = Agent(state_size=state_size, action_size=action_size, configs=configs)

    scores = dqn(env=env, agent=agent, label=label, **training)
    ax.plot(np.arange(len(scores)), scores, label=label)

    plt.ylabel("Score")
    plt.xlabel("Episode #")
    ax.legend(loc="upper center", shadow=True, fontsize="small")
    plt.savefig("results/plots/" + str(label))
    plt.show()


if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(main)
