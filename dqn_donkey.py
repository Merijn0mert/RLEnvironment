import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_donkey(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling=True):
        super(DQN_donkey, self).__init__()
        self.enable_dueling = enable_dueling

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling:
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input for linear layer
        x = F.relu(self.fc1(x))

        if self.enable_dueling:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + A - A.mean(dim=1, keepdim=True)
        else:
            Q = self.output(x)

        return Q


if __name__ == '__main__':
    state_dim = 12
    action_dim = 6
    net = DQN_donkey(state_dim, action_dim)

    dummy_state = torch.randn(1, state_dim)
    output = net(dummy_state)
    print(output.shape)  # torch.Size([1, 6])
