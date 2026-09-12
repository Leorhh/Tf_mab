import json
import torch
from torch.utils.data import DataLoader
from src.data.sequence_dataset import AmazonSequenceDataset
from src.models.transformer import TransformerRewardModel
from src.models.candidate_uncertainty import score_candidates_with_uncertainty
from src.bandit.random import RandomBandit
from src.bandit.epsilon_greedy import EpsilonGreedy
from src.bandit.ucb import UCB
from src.bandit.thompson import ThompsonSampling
from src.bandit.adaptive_mab import AdaptiveMAB

DATA_PATH = "data/processed/amazon/test_sequences.csv"
MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_transformer.pt"
BATCH_SIZE = 4
NUM_CANDIDATES = 100
MC_SAMPLES = 10
MAX_SEQ_LEN = 20

def generate_candidates(target_items, history_items, num_items, num_candidates=100, seed=42):
    all_candidates = []
    for i in range(len(target_items)):
        target = int(target_items[i].item())
        history_set = {int(x) for x in history_items[i].cpu().tolist() if int(x) > 0}
        rng = torch.Generator()
        rng.manual_seed(seed + i)
        candidates = []
        while len(candidates) < num_candidates - 1:
            sampled = torch.randint(1, num_items + 1, (num_candidates * 2,), generator=rng)
            for item in sampled.tolist():
                if item == target or item in history_set or item in candidates:
                    continue
                candidates.append(item)
                if len(candidates) == num_candidates - 1:
                    break
        candidates.append(target)
        permutation = torch.randperm(num_candidates, generator=rng)
        candidates = torch.tensor(candidates, dtype=torch.long)[permutation]
        all_candidates.append(candidates)
    return torch.stack(all_candidates)

def main():
    print("=" * 60)
    print("Transformer + Uncertainty + MAB Integration Test")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1/5] Loading dataset...")
    dataset = AmazonSequenceDataset(
        sequence_path=DATA_PATH,
        mapping_path=MAPPING_PATH,
        max_seq_len=MAX_SEQ_LEN,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    hist_items = batch["history_items"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    target_items = batch["target_item"].to(device)
    target_rewards = batch["target_reward"].to(device)
    print(f"[INFO] History shape: {hist_items.shape}")
    print(f"[INFO] Target shape:  {target_items.shape}")

    print("\n[2/5] Loading model...")
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = TransformerRewardModel(
        num_items=num_items,
        max_seq_len=MAX_SEQ_LEN,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        padding_idx=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"[INFO] Number of items: {num_items}")

    print("\n[3/5] Generating candidates...")
    candidates = generate_candidates(
        target_items=target_items,
        history_items=hist_items,
        num_items=num_items,
        num_candidates=NUM_CANDIDATES,
        seed=42,
    ).to(device)
    print(f"[INFO] Candidates shape: {candidates.shape}")

    print("\n[4/5] Computing reward + uncertainty...")
    mean_prediction, uncertainty, _ = score_candidates_with_uncertainty(
        model=model,
        hist_items=hist_items,
        attn_mask=attn_mask,
        candidates=candidates,
        num_samples=MC_SAMPLES,
    )
    print(f"[INFO] Reward prediction shape: {mean_prediction.shape}")
    print(f"[INFO] Uncertainty shape: {uncertainty.shape}")

    print("\n[5/5] Running MAB strategies...")
    strategies = {
        "Random": RandomBandit(seed=42),
        "Epsilon-Greedy": EpsilonGreedy(epsilon=0.1, seed=42),
        "UCB": UCB(c=1.0),
        "Thompson": ThompsonSampling(seed=42),
        "Adaptive MAB": AdaptiveMAB(beta_min=0.1, beta_max=1.0, tau=0.01),
    }

    print("\n" + "=" * 60)
    print("Selection Results")
    print("=" * 60)
    for name, bandit in strategies.items():
        selected_arms = []
        selected_items = []
        selected_rewards = []
        beta = 0.0
        for i in range(BATCH_SIZE):
            pred = mean_prediction[i].detach().cpu().numpy()
            unc = uncertainty[i].detach().cpu().numpy()
            if name == "Random":
                arm = bandit.select_arm(NUM_CANDIDATES)
            elif name == "Epsilon-Greedy":
                arm = bandit.select_arm(pred)
            elif name == "UCB":
                arm = bandit.select_arm(pred, unc)
            elif name == "Thompson":
                arm = bandit.select_arm(pred, unc)
            else:
                arm, scores, beta = bandit.select_arm(pred, unc)
            item = int(candidates[i, arm].item())
            is_target = item == int(target_items[i].item())
            selected_arms.append(arm)
            selected_items.append(item)
            reward = float(target_rewards[i].item()) if is_target else 0.0
            selected_rewards.append(reward)
            bandit.update(reward)
        print(f"\n{name}:")
        print(f"  Selected arms: {selected_arms}")
        print(f"  Selected items: {selected_items}")
        print(f"  Rewards: {selected_rewards}")
        print(f"  Mean reward: {sum(selected_rewards)/BATCH_SIZE:.6f}")
        if name == "Adaptive MAB":
            print(f"  Final beta: {beta:.6f}")

    print("\n" + "=" * 60)
    print("PASS: MAB integration completed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
