from collections import deque
import random
import joblib
import torch


class ReplayMemory:
    def __init__(self, maxlength, seed=None):
        self.memory = deque([], maxlen=maxlength)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

    def save_all(self, path):
        """Save the entire memory to disk using joblib after moving tensors to CPU."""
        cpu_memory = []
        for transition in self.memory:
            state, action, reward, next_state, done = transition

            # Move tensors to CPU if they're tensors
            if hasattr(state, 'cpu'): state = state.detach().cpu()
            if hasattr(action, 'cpu'): action = action.detach().cpu()
            if hasattr(reward, 'cpu'): reward = reward.detach().cpu()
            if hasattr(next_state, 'cpu'): next_state = next_state.detach().cpu()
            if hasattr(done, 'cpu'): done = done.detach().cpu()

            cpu_memory.append((state, action, reward, next_state, done))

        joblib.dump(cpu_memory, path, compress=6)

    def load_all(self, path, device='cpu'):
        """Load memory from disk and move tensors to the target device."""
        raw_memory = joblib.load(path)
        self.memory = deque([], maxlen=len(raw_memory))
        for state, action, reward, next_state, done in raw_memory:
            if isinstance(state, torch.Tensor): state = state.to(device)
            if isinstance(action, torch.Tensor): action = action.to(device)
            if isinstance(reward, torch.Tensor): reward = reward.to(device)
            if isinstance(next_state, torch.Tensor): next_state = next_state.to(device)
            if isinstance(done, torch.Tensor): done = done.to(device)
            self.memory.append((state, action, reward, next_state, done))


