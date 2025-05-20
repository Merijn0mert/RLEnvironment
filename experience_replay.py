from collections import deque
import random
import joblib

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
        """Save the entire memory to disk using joblib."""
        joblib.dump(self.memory, path)

    def load_all(self, path):
        """Load memory from disk."""
        self.memory = joblib.load(path)

