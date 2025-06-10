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

    def load_all(self, path, max_samples=None):
        """Load memory from disk and keep tensors on CPU."""
        raw_memory = joblib.load(path)

        if max_samples is not None:
            raw_memory = raw_memory[:max_samples]

        self.memory = deque([], maxlen=len(raw_memory))

        for state, action, reward, next_state, done in raw_memory:
            # Ensure all tensors remain on CPU
            if isinstance(state, torch.Tensor): state = state.cpu()
            if isinstance(action, torch.Tensor): action = action.cpu()
            if isinstance(reward, torch.Tensor): reward = reward.cpu()
            if isinstance(next_state, torch.Tensor): next_state = next_state.cpu()
            if isinstance(done, torch.Tensor): done = done.cpu()

            self.memory.append((state, action, reward, next_state, done))




