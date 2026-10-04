import numpy as np


class EpsilonGreedy:
    """Pick the model's argmax with prob 1-eps, a uniform random arm otherwise."""

    def __init__(self, epsilon=0.1, seed=None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.total_rounds = 0
        self.total_reward = 0.0

    def select_arm(self, predicted_reward, uncertainty=None, candidate_items=None):
        # uncertainty / candidate_items are accepted so every bandit shares
        # the same call signature in the experiment runner; eps-greedy
        # simply doesn't use them.
        predicted_reward = np.asarray(predicted_reward, dtype=np.float32)
        if predicted_reward.size == 0:
            raise ValueError("predicted_reward cannot be empty")

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, predicted_reward.size))
        return int(np.argmax(predicted_reward))

    def update(self, item_id, reward):
        reward = float(np.clip(float(reward), 0.0, 1.0))
        self.total_rounds += 1
        self.total_reward += reward

    def stats(self):
        mean_reward = self.total_reward / self.total_rounds if self.total_rounds else 0.0
        return {
            "total_rounds": self.total_rounds,
            "total_reward": self.total_reward,
            "mean_reward": mean_reward,
        }
