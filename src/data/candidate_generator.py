import numpy as np

class CandidateGenerator:
    def __init__(self, num_items, num_candidates=100, seed=42):
        if num_candidates < 2:
            raise ValueError("num_candidates must be at least 2")
        if num_items < num_candidates:
            raise ValueError("num_items must be greater than num_candidates")
        self.num_items = num_items
        self.num_candidates = num_candidates
        self.seed = seed

    def generate(self, target_item, history_items=None):
        target_item = int(target_item)
        if not 1 <= target_item <= self.num_items:
            raise ValueError(f"Invalid target_item: {target_item}")

        history_set = set()
        if history_items is not None:
            history_set = {int(it) for it in history_items if int(it) > 0}
        excluded = history_set | {target_item}

        rng = np.random.default_rng(self.seed + target_item)
        candidates = []
        while len(candidates) < self.num_candidates - 1:
            remaining = self.num_candidates - 1 - len(candidates)
            sampled = rng.integers(1, self.num_items + 1, size=remaining * 2)
            for item in sampled:
                item = int(item)
                if item in excluded or item in candidates:
                    continue
                candidates.append(item)
                if len(candidates) == self.num_candidates - 1:
                    break
        candidates.append(target_item)
        rng.shuffle(candidates)
        target_idx = candidates.index(target_item)
        return np.asarray(candidates, dtype=np.int64), target_idx
