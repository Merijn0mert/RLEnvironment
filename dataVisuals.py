import datetime
import matplotlib
import yaml

matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
from matplotlib import pyplot as plt
import os
from pathlib import Path
import torch
import numpy as np

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)  # Create directory if it doesn't exist

class DataVisuals():
    def __init__(self, hyperparameter_set):
        super().__init__()
        self.hyperparameter_set = hyperparameter_set
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def saveToLog(self, best_reward, episode, mean_reward, policy_dqn):
        log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: Episode {episode}: New best reward {mean_reward:0.1f} ({(mean_reward - best_reward) * 100:.1f}%)"
        print(log_message)
        with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')
        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
        # best_reward = mean_reward
        return best_reward

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        fig.set_size_inches(16, 10)
        plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.07, right=0.97, top=0.93, bottom=0.07)
        mean_reward = np.zeros(len(rewards_per_episode))
        # Calculate moving average of rewards
        for x in range(len(mean_reward)):
            mean_reward[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        # Graph 1 Left Top
        plt.subplot(221)
        plt.ylabel("Mean Reward")
        plt.plot(mean_reward)
        # Graph 2 Right Top
        plt.subplot(222)
        plt.ylabel("Episode Decay")
        plt.plot(epsilon_history)
        # Graph 3 full bottom (212) or left bottom (223)
        plt.subplot(212)
        plt.ylabel("Reward per episode")
        plt.plot(rewards_per_episode)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)  # Save the figure
        plt.close()