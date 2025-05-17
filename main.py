import sys

import gymnasium
import matplotlib
import torch
import flappy_bird_gymnasium
from rich.markup import render
from sympy import false

from dqn import DQN
from experience_replay import ReplayMemory
from dqn_agent import DQN_agent
import os


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Set device to gpu
    if not torch.cuda.is_available():
        print("No GPU available, using CPU instead")
    print('Using GPU')
    try:
        agent = DQN_agent(device=device, action_set=ord(' '), hyperparameter_set='cartpole1')
        agent.run()
    except KeyboardInterrupt:
        print('[INFO] Bye bye bye')
        sys.exit(0)





