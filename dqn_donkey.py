import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_donkey(nn.Module):
    def __init__(self, action_dim):
        super(DQN_donkey, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Automatically compute flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 210, 160)  # Actual Atari frame size
            out = self.conv3(self.conv2(self.conv1(dummy_input)))
            self.flattened_size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):

        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # NHWC → NCHW


        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))


        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)

