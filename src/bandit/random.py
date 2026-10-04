import numpy as np


class RandomBandit:
    """Uniform-random baseline. Ignores predictions entirely."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.total_rounds = 0
        self.total_reward = 0.0

    def select_arm(self, num_arms):
        if num_arms <= 0:
            raise ValueError("num_arms must be positive")
        return int(self.rng.integers(0, num_arms))

    def update(self, reward):
        self.total_rounds += 1
        self.total_reward += float(reward)

    def stats(self):
        mean_reward = self.total_reward / self.total_rounds if self.total_rounds else 0.0
        return {
            "total_rounds": self.total_rounds,
            "total_reward": self.total_reward,
            "mean_reward": mean_reward,
        }
