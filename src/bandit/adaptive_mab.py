"""AdaptiveMAB: uncertainty-adaptive exploration on top of a reward model.

Instead of a fixed exploration coefficient, beta scales with the mean
uncertainty of the current candidate set — the policy explores harder
exactly when the reward model is unsure, and settles into exploitation
once predictions are confident:

    beta  = beta_min + (beta_max - beta_min) * unc / (unc + tau)
    score = r_hat + beta * sigma + w * mean_history_reward

The saturating form keeps beta inside [beta_min, beta_max] regardless of
how uncertainty is scaled on a given dataset; tau sets how quickly it
saturates. The history term keeps a mild preference for items that have
actually paid off in past rounds, which hedges against unreliable model
predictions on cold items.
"""
import numpy as np


class AdaptiveMAB:
    def __init__(self, beta_min=0.1, beta_max=1.0, tau=0.01, history_weight=0.2):
        if beta_min < 0:
            raise ValueError("beta_min must be non-negative")
        if beta_max < beta_min:
            raise ValueError("beta_max must be >= beta_min")
        if tau <= 0:
            raise ValueError("tau must be positive")
        if history_weight < 0:
            raise ValueError("history_weight must be non-negative")

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.tau = tau
        self.history_weight = history_weight

        # running per-item statistics from observed environment feedback
        self.item_counts = {}
        self.item_reward_sums = {}
        self.total_rounds = 0
        self.total_reward = 0.0

    def _count(self, item_id):
        return self.item_counts.get(int(item_id), 0)

    def _mean_reward(self, item_id):
        item_id = int(item_id)
        n = self.item_counts.get(item_id, 0)
        return self.item_reward_sums[item_id] / n if n else 0.0

    def beta_for(self, uncertainty):
        """Map the candidate set's mean uncertainty to an exploration weight."""
        uncertainty = np.asarray(uncertainty, dtype=np.float32)
        if uncertainty.size == 0:
            raise ValueError("uncertainty cannot be empty")
        mean_unc = float(np.mean(uncertainty))
        beta = self.beta_min + (self.beta_max - self.beta_min) * (
            mean_unc / (mean_unc + self.tau)
        )
        return float(beta)

    def score(self, predicted_reward, uncertainty, candidate_items):
        predicted_reward = np.asarray(predicted_reward, dtype=np.float32)
        uncertainty = np.asarray(uncertainty, dtype=np.float32)
        candidate_items = np.asarray(candidate_items, dtype=np.int64)
        if predicted_reward.shape != uncertainty.shape:
            raise ValueError("predicted_reward and uncertainty must have the same shape")
        if predicted_reward.shape != candidate_items.shape:
            raise ValueError("candidate_items must have the same shape as predictions")

        beta = self.beta_for(uncertainty)
        scores = predicted_reward + beta * uncertainty

        if self.history_weight > 0:
            history = np.array(
                [self._mean_reward(item_id) for item_id in candidate_items],
                dtype=np.float32,
            )
            scores += self.history_weight * history

        return scores, beta

    def select_arm(self, predicted_reward, uncertainty, candidate_items):
        scores, beta = self.score(predicted_reward, uncertainty, candidate_items)
        return int(np.argmax(scores)), scores, beta

    def update(self, item_id, reward):
        item_id = int(item_id)
        reward = float(np.clip(float(reward), 0.0, 1.0))
        self.item_counts[item_id] = self.item_counts.get(item_id, 0) + 1
        self.item_reward_sums[item_id] = self.item_reward_sums.get(item_id, 0.0) + reward
        self.total_rounds += 1
        self.total_reward += reward

    def stats(self):
        mean_reward = self.total_reward / self.total_rounds if self.total_rounds else 0.0
        return {
            "total_rounds": self.total_rounds,
            "total_reward": self.total_reward,
            "mean_reward": mean_reward,
            "num_tracked_items": len(self.item_counts),
        }
