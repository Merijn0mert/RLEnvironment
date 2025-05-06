import torch

class DQN_agent():
    #agent uses device made in main
    def __init__(self, device, action_set):
        self.device = device
        self.action_set = action_set