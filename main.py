import gymnasium
import torch
import flappy_bird_gymnasium
from rich.markup import render
from sympy import false
from dqn import DQN

#Set device to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("bro what the frick")

print('gpu werkt')

class Agent():
    def run(self, is_training=True, render=False):
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        #agent = DQN_agent(device, action_set=env.action_space)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_actions).to_device(device)
        obs, _ = env.reset()
        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing:
            obs, reward, terminated, _, info = env.step(action)

            # Checking if the player is still alive
            if terminated:
                break

        env.close()


