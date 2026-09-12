import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon=0.1, seed=None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1"
            )

        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        self.total_rounds = 0
        self.total_reward = 0.0

    def select_arm(
        self,
        predicted_reward,
        uncertainty=None,
        candidate_items=None,
    ):
        predicted_reward = np.asarray(
            predicted_reward,
            dtype=np.float32
        )

        if predicted_reward.size == 0:
            raise ValueError(
                "predicted_reward cannot be empty"
            )

        if self.rng.random() < self.epsilon:
            return int(
                self.rng.integers(
                    0,
                    predicted_reward.size
                )
            )

        return int(
            np.argmax(predicted_reward)
        )

    def update(self, item_id, reward):
        reward = float(reward)

        reward = np.clip(
            reward,
            0.0,
            1.0
        )

        self.total_rounds += 1
        self.total_reward += reward

    def get_statistics(self):
        if self.total_rounds > 0:
            mean_reward = (
                self.total_reward
                / self.total_rounds
            )
        else:
            mean_reward = 0.0

        return {
            "total_rounds": self.total_rounds,
            "total_reward": self.total_reward,
            "mean_reward": mean_reward,
        }
