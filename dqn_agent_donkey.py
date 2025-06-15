import datetime
#from operator import truedi
import gymnasium
import ale_py
import numpy as np
#from collections import deque
import random
import torch
#from tqdm import tqdm
import joblib
from torch import nn
import yaml
import flappy_bird_gymnasium
import matplotlib
import checkpointHandler
import dataVisuals
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
from matplotlib import pyplot as plt
from networkx.generators.random_graphs import newman_watts_strogatz_graph
from rich.markup import render
from sympy import false
from dqn_donkey import DQN_donkey  # Import custom DQN model
from scoreoverlaywrapper import ScoreOverlayWrapper
from experience_replay import ReplayMemory  # Import experience replay buffer
from dataVisuals import DataVisuals
import itertools
import argparse
from gymnasium.vector import SyncVectorEnv
from gymnasium.vector import AsyncVectorEnv
import os
from gymnasium.wrappers import RecordVideo  # Import video recording wrapper
import time
import torchvision.transforms as T
from pathlib import Path


# Define date format for logging
DATE_FORMAT = "%m-%d %H:%M:%S"
# Directory to save training runs
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Set device to GPU if available, else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
print(torch.cuda.get_device_name())
print(device)


class DQN_agent_donkey():
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
        #self.data_visuals = DataVisuals(self.hyperparameter_set)
        #dv = data_visuals.DataVisuals(self.hyperparameter_set)


        # Define file paths for logging, model saving, and graph plotting
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        self.checkpoint_path = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_checkpoint.pth")

    def run(self, is_training=True, render=False, continue_training=False):

        NUM_ENVS = 1  # Number of parallel environments
        dv = dataVisuals.DataVisuals(self.hyperparameter_set)
        cp = checkpointHandler.checkpointer(self.hyperparameter_set)

        def make_env():
            def _init():
                return gymnasium.make(self.env_id, render_mode='human')

                # **self.env_make_params
                #gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=False)

            return _init

        #envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
        envs = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

        obs, _ = envs.reset()

        num_states = np.prod(obs.shape[1:])

        num_actions = envs.single_action_space.n
        last_graph_update_time = datetime.datetime.now()
        (best_reward, checkpoint_path, epsilon, epsilon_history, memory,
         policy_dqn, replay_memory_path, rewards_per_episode,
         start_episode, step_counter, target_dqn, self.optimizer) = cp.loadCheckpoint(continue_training, num_actions, num_states, self.device)
        policy_dqn = DQN_donkey(num_states, num_actions).to(device)
        target_dqn = DQN_donkey(num_states, num_actions).to(device)

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

                    # between_pipe_mask = (next_states[:, 4] < next_states[:, 9]) & (
                    #        next_states[:, 9] < next_states[:, 5])
                    #rewards_tensor += between_pipe_mask.float() * 0.1

                    if is_training:
                        for i in range(NUM_ENVS):
                            memory.append((
                                states[i].detach().cpu(),
                                actions[i].detach().cpu(),
                                next_states[i].detach().cpu(),
                                rewards_tensor[i].detach().cpu(),
                                bool(done_flags[i])
                            ))

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
                    dv.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                    print('saving graph to', self.GRAPH_FILE)

                if is_training:
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

        except KeyboardInterrupt:
            cp.saveCheckpoint(best_reward, checkpoint_path, episode, epsilon, epsilon_history, memory, policy_dqn,
                                replay_memory_path, rewards_per_episode, step_counter, target_dqn)
            dv.save_graph(rewards_per_episode, epsilon_history)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Unpack mini-batch
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Move all tensors to the appropriate device
        device = next(policy_dqn.parameters()).device  # automatically get model's device

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        new_states = torch.stack(new_states).to(device)
        rewards = torch.stack(rewards).to(device)
        terminations = torch.tensor(terminations, dtype=torch.float32).to(device)

        # Compute target Q-values
        with torch.no_grad():
            if self.enable_double_dqn:
                best_action_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(dim=1,
                                                         index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).max(dim=1)[0]

        # Compute current Q-values
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, episodes=11):
        # Create a single environment with rendering
        env = gymnasium.make(self.env_id, render_mode="human")
        self.evaluateProcess(env, episodes)
        env.close()

    def evaluateVideo(self, episodes=3):
        video_dir = os.path.join("videos", self.hyperparameter_set)
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        # Create a single environment with rendering
        base_env = gymnasium.make(self.env_id, render_mode="rgb_array")
        env_with_score = ScoreOverlayWrapper(base_env)
        env = RecordVideo(env_with_score, video_folder=video_dir, episode_trigger=lambda episode_id: True, name_prefix=f"{self.hyperparameter_set}_eval")
        self.evaluateProcess(env, episodes)
        env.close()
        print(f"\nVideos saved in: {video_dir}")

    def evaluateProcess(self, env, episodes):
        import random

        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            time.sleep(1 / 30)  # Slow down for visibility
        #print(f"Random action episode finished with total reward: {total_reward}\n")

        # Now run your DQN evaluation
        obs, _ = env.reset()
        if isinstance(obs, np.ndarray):
            num_states = obs.size
        else:
            num_states = obs.numel()
        num_actions = env.action_space.n

        # Load trained model
        policy_dqn = DQN_donkey(num_states, num_actions).to(device)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        policy_dqn.eval()

        for episode in range(1, episodes):  # Already did 1 random episode
            obs, _ = env.reset()
            print(f"\n--- Starting Evaluation Episode {episode} ---")
            done = False
            total_reward = 0
            step_count = 0
            while not done:
                state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_values = policy_dqn(state)
                    action = q_values.argmax(dim=1).item()

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if "score" in info:
                    score = info["score"]
                done = terminated or truncated
                step_count += 1

                #print(
                   # f"Step {step_count}: Q-values: {q_values.cpu().numpy()}, chosen action: {action}, Reward: {reward}, Done: {done}")
                time.sleep(1 / 30)  # Slow down so we can see rendering

            score = info.get("score", None)
            print(f"Episode {episode} finished with total reward: {total_reward}, Score: {score}")


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
        hyperparams = "donkeykong1"
        is_training = True
        dql = DQN_agent_donkey(hyperparameter_set=hyperparams, device=device)
        dql.run(is_training=is_training, continue_training=True)

    try:
        if is_training:
            dql = DQN_agent_donkey(hyperparameter_set=hyperparams, device=device)
            dql.run(is_training=True, continue_training=False)
        if training_continue:
            dql = DQN_agent_donkey(hyperparameter_set=hyperparams, device=device)
            dql.run(is_training=True, continue_training=True)
        if is_save:
            dql = DQN_agent_donkey(hyperparameter_set=hyperparams, device=device)
            dql.evaluateVideo()
        if is_evaluation:
            dql = DQN_agent_donkey(hyperparameter_set=hyperparams, device=device)
            dql.evaluate()
    except KeyboardInterrupt:
        print('[INFO] Bye bye bye')


