import torch
from torch.utils.data import DataLoader
from src.data.sequence_dataset import AmazonSequenceDataset
from src.models.transformer import TransformerRewardModel
from src.models.candidate_scorer import score_candidates

DATA_PATH = "data/processed/amazon/test_sequences.csv"
MAPPING_PATH = "data/processed/amazon/item_mapping.json"
CHECKPOINT_PATH = "checkpoints/best_transformer.pt"
BATCH_SIZE = 4
NUM_CANDIDATES = 100
MAX_SEQ_LEN = 20

def main():
    print("=" * 60)
    print("Candidate Scoring Test")
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
    target_item = batch["target_item"].to(device)
    print(f"[INFO] History shape: {hist_items.shape}")
    print(f"[INFO] Target shape:  {target_item.shape}")

    print("\n[2/4] Loading model...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    import json

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
    model.eval()
    print(f"[INFO] Number of items: {num_items}")

    print("\n[3/4] Creating candidate sets...")
    candidates = []
    for i in range(BATCH_SIZE):
        target = int(target_item[i].item())
        history = hist_items[i].cpu().numpy()
        history_set = {int(x) for x in history if int(x) > 0}
        generator_rng = torch.Generator()
        generator_rng.manual_seed(42 + i)
        candidate_list = []
        while len(candidate_list) < NUM_CANDIDATES - 1:
            sampled = torch.randint(1, num_items + 1, (NUM_CANDIDATES * 2,), generator=generator_rng)
            for item in sampled.tolist():
                if item == target or item in history_set or item in candidate_list:
                    continue
                candidate_list.append(item)
                if len(candidate_list) == NUM_CANDIDATES - 1:
                    break
        candidate_list.append(target)
        shuffle_indices = torch.randperm(NUM_CANDIDATES, generator=generator_rng)
        candidate_tensor = torch.tensor(candidate_list, dtype=torch.long)[shuffle_indices]
        candidates.append(candidate_tensor)
    candidates = torch.stack(candidates).to(device)
    print(f"[INFO] Candidates shape: {candidates.shape}")

    print("\n[4/4] Scoring candidates...")
    predictions = score_candidates(model=model, hist_items=hist_items, attn_mask=attn_mask, candidates=candidates)
    print(f"[INFO] Prediction shape: {predictions.shape}")

    print("\n" + "=" * 60)
    print("Result")
    print("=" * 60)
    print(f"Expected shape: ({BATCH_SIZE}, {NUM_CANDIDATES})")
    print(f"Actual shape:   {tuple(predictions.shape)}")

    print("\nFirst user:")
    print(f"Target item: {target_item[0].item()}")
    target_position = (candidates[0] == target_item[0]).nonzero(as_tuple=True)[0].item()
    print(f"Target position: {target_position}")

    print("\nFirst 10 candidates:")
    print(candidates[0, :10].detach().cpu().tolist())
    print("\nFirst 10 predictions:")
    print(predictions[0, :10].detach().cpu().numpy())

    print("\nPrediction range:")
    print(f"Min: {predictions.min().item():.6f}")
    print(f"Max: {predictions.max().item():.6f}")
    print("\nTarget prediction:")
    print(f"{predictions[0, target_position].item():.6f}")

    print("\n" + "=" * 60)
    if predictions.shape == (BATCH_SIZE, NUM_CANDIDATES):
        print("PASS: Candidate scoring works.")
    else:
        print("FAIL: Unexpected prediction shape.")


if __name__ == "__main__":
    main()
