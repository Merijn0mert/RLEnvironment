import matplotlib.pyplot as plt
import csv
import os
import math
import numpy as np
import torch
from dqn_agent_monster import DQN_agent_monster
from ple import PLE
from ple.games.monsterkong import MonsterKong
import sys
from datetime import datetime

reward_history = []
epsilon_history = []


# Define custom rewards for the game
rewards = {
    "positive": 0,    # coin
    "win": 500,         # save princess
    "negative": -1,    # fireball
    "tick": -0.1        # every frame
}

total_episodes = 0

ladder_reward = 0.1
climb_multiplier = 10.0
correct_direction = 0.5

log_file = "training_log.csv"


def decay_shaping_reward(base_value, decay_rate, episode, min_value=0.0):
    return max(min_value, base_value * (1 - decay_rate * episode))

def get_state_vector(state):
    """
    Convert the game data dictionary to a vector for the DQN agent.
    """
    return np.array([
        state["player_x"],
        state["player_y"],
        float(state["on_ladder"]),
        float(state["on_ground"]),
        state["closest_ladder_x"],
        state["closest_fireball_x"],
        state["closest_fireball_y"],
    ], dtype=np.float32)

def train_agent(env, agent, episodes):
    """
    Train the DQN agent over a specified number of episodes.
    """
    global total_episodes

    wonned = False
    current_ladder_reward = ladder_reward
    current_direction_reward = correct_direction

    # Setup logging
    write_header = not os.path.exists(log_file)
    if write_header:
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "TotalReward", "Epsilon"])

    for e in range(episodes):
        initial_game_state = env.getGameState()
        initial_vector_state = get_state_vector(initial_game_state)

        total_reward = 0

        best_y = initial_game_state["player_y"]
        timer = 0 # Prevent infinite loops while no lose state

        # Decay intermediate shaping rewards
        if wonned:
            current_ladder_reward = decay_shaping_reward(ladder_reward, 0.005, total_episodes)
            current_direction_reward = decay_shaping_reward(correct_direction, 0.005, total_episodes)

        while not env.game_over() and timer < 2501:
            # Agent selects an action
            action = agent.select_action(initial_vector_state)
            reward = env.act(action)

            # Process the updated game data
            post_action_game_state = env.getGameState()
            post_action_vector_state = get_state_vector(post_action_game_state)

            # Bonus reward for climbing higher on a ladder
            climb_reward = 0
            if post_action_game_state["player_y"] < best_y and post_action_game_state["on_ladder"]:
                climb_reward = (best_y - post_action_game_state["player_y"]) * climb_multiplier
                best_y = post_action_game_state["player_y"]
            reward += climb_reward

            # bonus reward for just being on a ladder
            if post_action_game_state["on_ladder"]:
                reward += ladder_reward

            # bonus reward for going towards the ladder
            prev_distance = abs(initial_game_state["player_x"] - initial_game_state["closest_ladder_x"])
            new_distance = abs(post_action_game_state["player_x"] - post_action_game_state["closest_ladder_x"])

            if new_distance < prev_distance:
                reward += correct_direction

            done = env.game_over()

            # print(f"initial vectors: {initial_vector_state} post action: {post_action_vector_state} action taken: {chr(action) if action is not None else "none"} reward: {round(reward, 3)} ")
            if contains_nan(initial_vector_state) or contains_nan(post_action_vector_state) or contains_nan(reward):
                print("⚠️ NaN detected!")
                print(f"initial vectors: {initial_vector_state} post action: {post_action_vector_state} action taken: {chr(action) if action is not None else 'none'} reward: {round(reward, 3) if not contains_nan(reward) else 'NaN'}")

            agent.remember(initial_vector_state, action, reward, post_action_vector_state, done)
            initial_vector_state = post_action_vector_state
            initial_game_state = post_action_game_state

            total_reward += reward
            timer += 1

            if timer % 500 == 0:
                agent.update_target_model()


            # Perform a training step
            agent.replay()

            # Break early if the win reward is achieved
            if reward >= rewards["win"]:
                # wonned = True
                break



        # Update the target model
        # if e % 5 == 0:
        #     agent.update_target_model()
        if total_episodes % 500 == 0:
            agent.epsilon  = max(agent.epsilon, 0.3)

        agent.decay_epsilon()
        env.reset_game()

        print(f"Episode {e + 1}/{episodes} - "
              f"Total Reward: {total_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Memory: {len(agent.memory)}/{agent.memory.maxlen} | ")
        total_episodes += 1
        print(f"Total episodes: {total_episodes}")

        # Log to CSV
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([total_episodes, total_reward, agent.epsilon])

def plot_csv():
    # --- Plotting from CSV ---
    plot_episodes = []
    plot_rewards = []
    plot_epsilons = []

    with open(log_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            plot_episodes.append(int(row["Episode"]))
            plot_rewards.append(float(row["TotalReward"]))
            plot_epsilons.append(float(row["Epsilon"]))

    plt.figure(figsize=(12, 5))

    # Total reward
    plt.subplot(1, 2, 1)
    plt.plot(plot_episodes, plot_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards Over Time')
    plt.legend()

    # Epsilon decay
    plt.subplot(1, 2, 2)
    plt.plot(plot_episodes, plot_epsilons, label='Epsilon', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Over Time')
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"training_plot_{timestamp}.png"

    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()

    # Optional: loss plot
    if len(agent.loss_history) > 0:
        if len(agent.loss_history) > 0:
            plt.figure()
            plt.plot(agent.loss_history, label='Training Loss', alpha=0.6)
            plt.xlabel('Training Step')
            plt.ylabel('MSE Loss')
            plt.title('Loss During Training')
            plt.legend()
            loss_filename = f"loss_plot_{timestamp}.png"
            plt.savefig(loss_filename)
            plt.show()

def contains_nan(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.isnan(x).any()
    if isinstance(x, (float, int)):
        return math.isnan(x)
    return False

if __name__ == "__main__":
    # Initialize the MonsterKong game environment
    game = MonsterKong()
    env = PLE(game, fps=30, display_screen=True, reward_values=rewards)
    env.init()

    # Get the set of possible actions
    action_set = env.getActionSet()

    # Determine the computing device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available, check torch installation")

    # Initialize the DQN agent
    agent = DQN_agent_monster(input_dim=len(env.getGameState()), action_set=action_set, device=device)

    # Attempt to load the agent's prior training checkpoint
    try:
        total_episodes = agent.load("checkpoint.pkl")
    except FileNotFoundError:
        print("No previous checkpoint found. Starting fresh.")
    except RuntimeError:
        print("Error decoding model. Starting fresh.")

    # Train the agent, handling interruptions gracefully
    try:
        train_agent(env, agent, episodes=15000)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Saving...")
        agent.save("checkpoint.pkl", total_episodes)
        plot_csv()
        sys.exit(0)