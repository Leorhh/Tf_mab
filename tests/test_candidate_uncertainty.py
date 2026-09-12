import json
import torch
from torch.utils.data import DataLoader
from src.data.sequence_dataset import AmazonSequenceDataset
from src.models.transformer import TransformerRewardModel
from src.models.candidate_uncertainty import score_candidates_with_uncertainty

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
    print("MC Dropout Candidate Uncertainty Test")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1/4] Loading dataset...")
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
    print(f"[INFO] History shape: {hist_items.shape}")
    print(f"[INFO] Target shape:  {target_items.shape}")

    print("\n[2/4] Loading model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
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

    print("\n[3/4] Generating candidates...")
    candidates = generate_candidates(
        target_items=target_items,
        history_items=hist_items,
        num_items=num_items,
        num_candidates=NUM_CANDIDATES,
        seed=42,
    )
    candidates = candidates.to(device)
    print(f"[INFO] Candidates shape: {candidates.shape}")

    print("\n[4/4] Running MC Dropout...")
    mean_prediction, uncertainty, predictions = score_candidates_with_uncertainty(
        model=model,
        hist_items=hist_items,
        attn_mask=attn_mask,
        candidates=candidates,
        num_samples=MC_SAMPLES,
    )
    print(f"[INFO] Prediction shape: {mean_prediction.shape}")
    print(f"[INFO] Uncertainty shape: {uncertainty.shape}")
    print(f"[INFO] MC prediction shape: {predictions.shape}")

    print("\n" + "=" * 60)
    print("Result")
    print("=" * 60)
    print(f"Expected prediction shape: ({BATCH_SIZE}, {NUM_CANDIDATES})")
    print(f"Actual prediction shape:   {tuple(mean_prediction.shape)}")
    print(f"Expected uncertainty shape: ({BATCH_SIZE}, {NUM_CANDIDATES})")
    print(f"Actual uncertainty shape:   {tuple(uncertainty.shape)}")

    print("\nPrediction statistics:")
    print(f"Mean: {mean_prediction.mean().item():.6f}")
    print(f"Min:  {mean_prediction.min().item():.6f}")
    print(f"Max:  {mean_prediction.max().item():.6f}")

    print("\nUncertainty statistics:")
    print(f"Mean: {uncertainty.mean().item():.6f}")
    print(f"Min:  {uncertainty.min().item():.6f}")
    print(f"Max:  {uncertainty.max().item():.6f}")

    print("\nFirst user:")
    target = target_items[0].item()
    target_position = (candidates[0] == target).nonzero(as_tuple=True)[0].item()
    print(f"Target item:       {target}")
    print(f"Target position:   {target_position}")
    print(f"Target prediction: {mean_prediction[0, target_position].item():.6f}")
    print(f"Target uncertainty: {uncertainty[0, target_position].item():.6f}")

    print("\nFirst 10 candidates:")
    for i in range(10):
        item = candidates[0, i].item()
        pred = mean_prediction[0, i].item()
        unc = uncertainty[0, i].item()
        print(f"{i:2d}. item={item:6d} reward={pred:.6f} uncertainty={unc:.6f}")

    print("\nMC variation check:")
    first_candidate_predictions = predictions[:, 0, 0].detach().cpu().numpy()
    print(first_candidate_predictions)

    print("\n" + "=" * 60)
    valid = (mean_prediction.shape == (BATCH_SIZE, NUM_CANDIDATES)
             and uncertainty.shape == (BATCH_SIZE, NUM_CANDIDATES)
             and torch.all(uncertainty >= 0)
             and predictions.shape == (MC_SAMPLES, BATCH_SIZE, NUM_CANDIDATES))
    if valid:
        print("PASS: MC Dropout candidate uncertainty works.")
    else:
        print("FAIL: Unexpected result.")

if __name__ == "__main__":
    main()
