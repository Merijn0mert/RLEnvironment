import datetime
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import matplotlib


matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
from matplotlib import pyplot as plt
import os
from pathlib import Path
import torch
import numpy as np

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class checkpointer:
    def __init__(self, hyperparameter_set):
        super(checkpointer, self).__init__()
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]

        # Assign hyperparameters to class variables
        self.hyperparameter_set = hyperparameter_set
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.enable_double_dqn = hyperparameters['enable_double_dqn']

    def loadCheckpoint(self, continue_training, num_actions, num_states, device):
        checkpoint_path = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        replay_memory_path = checkpoint_path + ".memory"
        memory = ReplayMemory(self.replay_memory_size)
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if not continue_training:
            best_reward, epsilon, epsilon_history, rewards_per_episode, start_episode, step_counter = self.dqnVariables(
                checkpoint, continue_training)
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        else:
            if os.path.exists(replay_memory_path):
                memory.load_all(replay_memory_path)
                print(f"[INFO] Loaded replay memory from {replay_memory_path}")
            else:
                print("[WARNING] Replay memory file not found, starting with empty replay memory.")
            best_reward, epsilon, epsilon_history, rewards_per_episode, start_episode, step_counter = self.dqnVariables(
                checkpoint, continue_training)
            policy_dqn.load_state_dict(checkpoint["policy_net"])
            target_dqn.load_state_dict(checkpoint["target_net"])
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        return (best_reward, checkpoint_path, epsilon, epsilon_history, memory,
                policy_dqn, replay_memory_path, rewards_per_episode,
                start_episode, step_counter, target_dqn, self.optimizer)

    def dqnVariables(self, checkpoint, continue_training):
        if continue_training:
            epsilon = checkpoint["epsilon"]
            start_episode = checkpoint["episode"]
            rewards_per_episode = checkpoint["rewards_per_episode"]
            epsilon_history = checkpoint["epsilon_history"]
            best_reward = checkpoint["best_reward"]
            step_counter = checkpoint["step_counter"]
            return best_reward, epsilon, epsilon_history, rewards_per_episode, start_episode, step_counter
        else:
            start_episode = 0
            epsilon = self.epsilon_init
            epsilon_history = []
            rewards_per_episode = []
            step_counter = 0
            best_reward = -9999999
            return best_reward, epsilon, epsilon_history, rewards_per_episode, start_episode, step_counter

    def saveCheckpoint(self, best_reward, checkpoint_path, episode, epsilon, epsilon_history, memory, policy_dqn,
                       replay_memory_path, rewards_per_episode, step_counter, target_dqn):
        print("\n[INFO] Training interrupted. Saving checkpoint before exit...")
        # Save replay memory efficiently with joblib
        print(f"[INFO] Saving replay memory at {datetime.datetime.now()}...")
        memory.save_all(replay_memory_path)
        # Save checkpoint normally with torch for model & other info
        torch.save({
            "policy_net": policy_dqn.state_dict(),
            "target_net": target_dqn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_memory_data": [],  # Keep empty to reduce checkpoint size
            "epsilon": epsilon,
            "episode": episode,
            "rewards_per_episode": rewards_per_episode,
            "epsilon_history": epsilon_history,
            "best_reward": best_reward,
            "step_counter": step_counter,
        }, checkpoint_path)
        print(f"Checkpoint saved at time :{datetime.datetime.now()} and episode: {episode}. Exiting gracefully.")