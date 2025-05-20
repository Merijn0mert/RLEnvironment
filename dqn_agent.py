import datetime
import gymnasium
import numpy as np
from collections import deque
import random
import torch
from tqdm import tqdm
import joblib
from torch import nn
import yaml
import flappy_bird_gymnasium
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
from matplotlib import pyplot as plt
from networkx.generators.random_graphs import newman_watts_strogatz_graph
from rich.markup import render
from sympy import false
from dqn import DQN  # Import custom DQN model
from scoreoverlaywrapper import ScoreOverlayWrapper
from experience_replay import ReplayMemory  # Import experience replay buffer
import itertools
import argparse
from gymnasium.vector import SyncVectorEnv
import os
from gymnasium.wrappers import RecordVideo  # Import video recording wrapper
import time
from pathlib import Path

# Define date format for logging
DATE_FORMAT = "%m-%d %H:%M:%S"
# Directory to save training runs
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Set device to GPU if available, else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.get_device_name())
print(device)


class DQN_agent():
    def __init__(self, hyperparameter_set, device):
        # Load hyperparameters from YAML file
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

        self.device = device

        # Define loss function and optimizer placeholder
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Define file paths for logging, model saving, and graph plotting
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        self.checkpoint_path = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_checkpoint.pth")

    def run(self, is_training=True, render=False):
        NUM_ENVS = 16

        def make_env():
            def _init():
                return gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)

            return _init

        envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
        obs, _ = envs.reset()

        num_states = obs.shape[1]
        num_actions = envs.single_action_space.n

        rewards_per_episode = []
        last_graph_update_time = datetime.datetime.now()

        checkpoint_path = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_checkpoint.pth")
        #memory_path = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_replay_memory.joblib")
        # Path to separate replay memory file
        replay_memory_path = checkpoint_path + ".memory"
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            epsilon = self.epsilon_init

            memory = ReplayMemory(self.replay_memory_size)
            # Load replay memory if file exists

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            epsilon_history = []
            episode_counter = 0
            step_counter = 0
            best_reward = -9999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        try:
            for episode in itertools.count():
                obs, _ = envs.reset()
                states = torch.tensor(obs, dtype=torch.float32, device=device)
                dones = [False] * NUM_ENVS
                episode_rewards = np.zeros(NUM_ENVS)

                print("episode =", episode)
                episode_counter = episode

                if is_training:
                    print("epsilon =", epsilon)

                while not all(dones):
                    if is_training and len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)
                        if step_counter > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_counter = 0

                    if is_training and random.random() < epsilon:
                        actions = torch.tensor([envs.single_action_space.sample() for _ in range(NUM_ENVS)],
                                               dtype=torch.int64, device=device)
                    else:
                        with torch.no_grad():
                            actions = policy_dqn(states).argmax(dim=1)

                    next_obs, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
                    next_states = torch.tensor(next_obs, dtype=torch.float32, device=device)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                    done_flags = np.logical_or(terminations, truncations)

                    between_pipe_mask = (next_states[:, 4] < next_states[:, 9]) & (
                                next_states[:, 9] < next_states[:, 5])
                    rewards_tensor += between_pipe_mask.float() * 0.1

                    if is_training:
                        for i in range(NUM_ENVS):
                            memory.append((states[i], actions[i], next_states[i], rewards_tensor[i], done_flags[i]))

                        step_counter += NUM_ENVS

                    states = next_states
                    episode_rewards += rewards
                    dones = np.logical_or(dones, done_flags)

                mean_reward = np.mean(episode_rewards)
                rewards_per_episode.append(mean_reward)

                if is_training and mean_reward > best_reward:
                    self.saveToLog(best_reward, episode, mean_reward, policy_dqn)

                current_time = datetime.datetime.now()
                if is_training and (current_time - last_graph_update_time > datetime.timedelta(seconds=10)):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                    print('saving graph to', self.GRAPH_FILE)

                if is_training:
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

        except KeyboardInterrupt:

            self.saveCheckpoint(best_reward, checkpoint_path, episode, epsilon, epsilon_history, memory, policy_dqn,
                                replay_memory_path, rewards_per_episode, step_counter, target_dqn)

    def saveToLog(self, best_reward, episode, mean_reward, policy_dqn):
        log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: Episode {episode}: New best reward {mean_reward:0.1f} ({(mean_reward - best_reward) * 100:.1f}%)"
        print(log_message)
        with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')
        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
        best_reward = mean_reward
        return best_reward

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        mean_reward = np.zeros(len(rewards_per_episode))
        # Calculate moving average of rewards
        for x in range(len(mean_reward)):
            mean_reward[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        plt.subplot(121)
        plt.ylabel("Mean Reward")
        plt.plot(mean_reward)
        plt.subplot(122)
        plt.ylabel("Episode Decay")
        plt.plot(epsilon_history)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.GRAPH_FILE)  # Save the figure
        plt.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Unpack mini-batch
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        new_states = torch.stack(new_states).to(device)
        rewards = torch.stack(rewards).to(device)
        terminations = torch.tensor(terminations).float().to(device)

        # Compute target Q-values
        with torch.no_grad():
            if self.enable_double_dqn:
                best_action_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
        # Compute current Q-values
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, episodes=11):
        # Create a single environment with rendering
        env = gymnasium.make("FlappyBird-v0", render_mode="human", **self.env_make_params)
        self.evaluateProcess(env, episodes)
        env.close()

    def evaluateVideo(self, episodes=3):
        video_dir = os.path.join("videos", self.hyperparameter_set)
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        # Create a single environment with rendering
        base_env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", **self.env_make_params)
        env_with_score = ScoreOverlayWrapper(base_env)
        env = RecordVideo(env_with_score, video_folder=video_dir, episode_trigger=lambda episode_id: True, name_prefix=f"{self.hyperparameter_set}_eval")
        self.evaluateProcess(env, episodes)
        env.close()
        print(f"\nVideos saved in: {video_dir}")

    def evaluateProcess(self, env, episodes):
        obs, _ = env.reset()
        num_states = len(obs)
        num_actions = env.action_space.n
        # Load trained model
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(self.device)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        policy_dqn.eval()
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            print(f"\n--- Starting Evaluation Episode {episode + 1} ---")

            while not done:
                state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = policy_dqn(state).argmax(dim=1).item()

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if "score" in info:
                    score = info["score"]
                done = terminated or truncated

                time.sleep(1 / 200)  # Limit to ~200 FPS

            print(f"Episode {episode + 1} finished with reward: {total_reward} Score: {score}")

    def continue_training(self, is_training=True, render=False):
        NUM_ENVS = 16  # Number of parallel environments

        def make_env():
            def _init():
                return gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)

            return _init

        envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
        obs, _ = envs.reset()

        num_states = obs.shape[1]
        num_actions = envs.single_action_space.n
        last_graph_update_time = datetime.datetime.now()

        checkpoint_path = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        replay_memory_path = checkpoint_path + ".memory"
        memory = ReplayMemory(self.replay_memory_size)

        if os.path.exists(replay_memory_path):
            memory.load_all(replay_memory_path)
            print(f"[INFO] Loaded replay memory from {replay_memory_path}")
        else:
            print("[WARNING] Replay memory file not found, starting with empty replay memory.")

        epsilon = checkpoint["epsilon"]
        start_episode = checkpoint["episode"]
        rewards_per_episode = checkpoint["rewards_per_episode"]
        epsilon_history = checkpoint["epsilon_history"]
        best_reward = checkpoint["best_reward"]
        step_counter = checkpoint["step_counter"]

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(self.device)
        target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(self.device)
        policy_dqn.load_state_dict(checkpoint["policy_net"])
        target_dqn.load_state_dict(checkpoint["target_net"])

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        try:
            for episode in itertools.count(start=start_episode):
                obs, _ = envs.reset()
                states = torch.tensor(obs, dtype=torch.float32, device=self.device)
                dones = [False] * NUM_ENVS
                episode_rewards = np.zeros(NUM_ENVS)

                print("episode =", episode)
                if is_training:
                    print("epsilon =", epsilon)

                while not all(dones):
                    if is_training and len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)
                        if step_counter > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_counter = 0

                    if is_training and random.random() < epsilon:
                        actions = torch.tensor(
                            [envs.single_action_space.sample() for _ in range(NUM_ENVS)],
                            dtype=torch.int64, device=self.device
                        )
                    else:
                        with torch.no_grad():
                            actions = policy_dqn(states).argmax(dim=1)

                    next_obs, rewards, terminations, truncations, info = envs.step(actions.cpu().numpy())
                    next_states = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                    done_flags = np.logical_or(terminations, truncations)

                    between_pipe_mask = (next_states[:, 4] < next_states[:, 9]) & (
                            next_states[:, 9] < next_states[:, 5])
                    rewards_tensor += between_pipe_mask.float() * 0.1

                    if is_training:
                        for i in range(NUM_ENVS):
                            memory.append((states[i], actions[i], next_states[i], rewards_tensor[i], done_flags[i]))
                        step_counter += NUM_ENVS

                    states = next_states
                    episode_rewards += rewards
                    dones = np.logical_or(dones, done_flags)

                mean_reward = np.mean(episode_rewards)
                rewards_per_episode.append(mean_reward)

                # Log score from best-performing env
                best_env_index = int(np.argmax(episode_rewards))
                episode_score = None
                if isinstance(info, list) and best_env_index < len(info):
                    best_info = info[best_env_index]
                    if isinstance(best_info, dict) and "score" in best_info:
                        episode_score = best_info["score"]

                print('mean_reward =', mean_reward, 'best_reward =', best_reward)
                if is_training and mean_reward > best_reward:
                    log_message = (
                        f"{datetime.datetime.now().strftime(DATE_FORMAT)}: "
                        f"Episode {episode}: "
                        f"New best reward {mean_reward:0.1f} "
                        f"({(mean_reward - best_reward) * 100:.1f}%)"
                    )
                    if episode_score is not None:
                        log_message += f" | Best Env Score: {episode_score}"

                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = mean_reward

                current_time = datetime.datetime.now()
                if is_training and (current_time - last_graph_update_time > datetime.timedelta(seconds=10)):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                    print('saving graph to', self.GRAPH_FILE)

                if is_training:
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

        except KeyboardInterrupt:
            self.saveCheckpoint(best_reward, checkpoint_path, episode, epsilon, epsilon_history, memory, policy_dqn,
                                replay_memory_path, rewards_per_episode, step_counter, target_dqn
            )

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


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Train or test model')
        parser.add_argument('hyperparameters', help='Name of the hyperparameter set to use')
        parser.add_argument('--train', help='Enable training mode', action='store_true')
        parser.add_argument('--evaluate', help='Run evaluation mode with rendering', action='store_true')
        parser.add_argument('--save', help='Run evaluation recording mode', action='store_true')
        parser.add_argument('--continuetrain', help='Continue training mode', action='store_true')
        args = parser.parse_args()
        hyperparams = args.hyperparameters

        is_training = args.train
        is_save = args.save
        is_evaluation = args.evaluate
        training_continue = args.continuetrain
    else:
        # Default hyperparameters and training mode
        hyperparams = "flappybird1"
        is_training = True

    try:
        if is_training:
            dql = DQN_agent(hyperparameter_set=hyperparams, device=device)
            dql.run(is_training=is_training)
        if training_continue:
            dql = DQN_agent(hyperparameter_set=hyperparams, device=device)
            dql.continue_training(is_training=True)
        if is_save:
            dql = DQN_agent(hyperparameter_set=hyperparams, device=device)
            dql.evaluateVideo()
        if is_evaluation:
            dql = DQN_agent(hyperparameter_set=hyperparams, device=device)
            dql.evaluate()
    except KeyboardInterrupt:
        print('[INFO] Bye bye bye')


