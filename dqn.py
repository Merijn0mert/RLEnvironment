import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, output_size, fc1_nodes):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, fc1_nodes),
            nn.ReLU(),
            nn.Linear(fc1_nodes, fc1_nodes),
            nn.ReLU(),
            nn.Linear(fc1_nodes, output_size)
        )

    def forward(self, x):
        return self.model(x)



if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)