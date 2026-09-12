import numpy as np
class ThompsonSampling:
    def __init__(self, alpha_prior=1.0, beta_prior=1.0, seed=None):
        if alpha_prior <= 0:
            raise ValueError("alpha_prior must be positive")
        if beta_prior <= 0:
            raise ValueError("beta_prior must be positive")
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.alpha = {}
        self.beta = {}
        self.rng = np.random.default_rng(seed)
        self.total_rounds = 0
        self.total_reward = 0.0
    def _get_alpha(self, item_id):
        return self.alpha.get(
            int(item_id),
            self.alpha_prior
        )
    def _get_beta(self, item_id):
        return self.beta.get(
            int(item_id),
            self.beta_prior
        )
    def select_arm(
        self,
        predicted_reward,
        uncertainty,
        candidate_items,
    ):
        pred_r = np.asarray(
            predicted_reward,
            dtype=np.float32
        )
        unc = np.asarray(
            uncertainty,
            dtype=np.float32
        )
        items = np.asarray(
            candidate_items,
            dtype=np.int64
        )
        if pred_r.shape != unc.shape:
            raise ValueError(
                "predicted_reward and uncertainty must have the same shape"
            )
        if pred_r.shape != items.shape:
            raise ValueError(
                "candidate_items must have the same shape as predictions"
            )
        if pred_r.size == 0:
            raise ValueError(
                "predicted_reward cannot be empty"
            )
        samples = np.zeros(
            pred_r.size,
            dtype=np.float32
        )
        for i, item_id in enumerate(items):
            alpha = self._get_alpha(item_id)
            beta = self._get_beta(item_id)
            posterior_sample = self.rng.beta(
                alpha,
                beta
            )
            samples[i] = (
                0.5 * pred_r[i]
                + 0.5 * posterior_sample
                + unc[i]
            )
        return int(np.argmax(samples))
    def update(self, item_id, reward):
        item_id = int(item_id)
        reward = float(reward)
        reward = np.clip(reward, 0.0, 1.0)
        self.alpha[item_id] = (
            self._get_alpha(item_id)
            + reward
        )
        self.beta[item_id] = (
            self._get_beta(item_id)
            + (1.0 - reward)
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
            "num_tracked_items": len(self.alpha),
        }
