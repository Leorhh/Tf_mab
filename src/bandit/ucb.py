import numpy as np
class UCB:
    def __init__(self, c=1.0, prior_weight=0.5):
        if c < 0:
            raise ValueError(
                "c must be non-negative"
            )
        if prior_weight < 0:
            raise ValueError(
                "prior_weight must be non-negative"
            )
        self.c = c
        self.prior_weight = prior_weight
        self.item_counts = {}
        self.item_reward_sums = {}
        self.total_rounds = 0
        self.total_reward = 0.0

    def _get_count(self, item_id):
        return self.item_counts.get(
            int(item_id), 0
        )

    def _get_reward_sum(self, item_id):
        return self.item_reward_sums.get(
            int(item_id), 0.0
        )

    def select_arm(
        self,
        predicted_reward,
        uncertainty,
        candidate_items,
    ):
        pred_r = np.asarray(
            predicted_reward, dtype=np.float32
        )
        unc = np.asarray(
            uncertainty, dtype=np.float32
        )
        items = np.asarray(
            candidate_items, dtype=np.int64
        )
        if pred_r.shape != unc.shape:
            raise ValueError(
                "predicted_reward and uncertainty "
                "must have the same shape"
            )
        if pred_r.shape != items.shape:
            raise ValueError(
                "candidate_items must have the "
                "same shape as predictions"
            )
        if pred_r.size == 0:
            raise ValueError(
                "predicted_reward cannot be empty"
            )
        scores = np.zeros(
            pred_r.size, dtype=np.float32
        )
        log_term = np.log(
            max(self.total_rounds, 1) + 1
        )
        for i, item_id in enumerate(items):
            count = self._get_count(item_id)
            if count == 0:
                historical_mean = pred_r[i]
            else:
                historical_mean = (
                    self._get_reward_sum(item_id) / count
                )
            bonus = self.c * np.sqrt(
                log_term / (count + 1)
            )
            scores[i] = (
                pred_r[i] + self.prior_weight * historical_mean + bonus
            )
        return int(
            np.argmax(scores)
        )

    def update(self, item_id, reward):
        item_id = int(item_id)
        reward = float(reward)
        reward = np.clip(
            reward,
            0.0,
            1.0
        )
        self.item_counts[item_id] = (
            self.item_counts.get(
                item_id, 0
            ) + 1
        )
        self.item_reward_sums[item_id] = (
            self.item_reward_sums.get(
                item_id, 0.0
            ) + reward
        )
        self.total_rounds += 1
        self.total_reward += reward

    def get_statistics(self):
        if self.total_rounds > 0:
            mean_reward = (
                self.total_reward / self.total_rounds
            )
        else:
            mean_reward = 0.0
        return {
            "total_rounds": self.total_rounds,
            "total_reward": self.total_reward,
            "mean_reward": mean_reward,
            "num_tracked_items": len(
                self.item_counts
            ),
        }
