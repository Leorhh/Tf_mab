import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.sequence_dataset import SequenceDataset
from src.models.transformer import TransformerRewardModel
from src.models.uncertainty import mc_dropout_predict

OUTPUT_DIR = "outputs/uncertainty"
BATCH_SIZE = 256
MC_SAMPLES = 10
NUM_WORKERS = 0

DATASETS = {
    "Amazon": {
        "test_path": "data/processed/amazon/test_sequences.csv",
        "mapping_path": "data/processed/amazon/item_mapping.json",
        "checkpoint_path": "checkpoints/best_transformer.pt",
        "max_seq_len": 20,
        "output_prefix": "amazon",
        "display_name": "Amazon Electronics",
    },
    "KuaiRand": {
        "test_path": "data/processed/Kuairand/test_sequences.csv",
        "mapping_path": "data/processed/Kuairand/item_mapping.json",
        "checkpoint_path": "checkpoints/best_kuairand_transformer.pt",
        "max_seq_len": 50,
        "output_prefix": "kuairand",
        "display_name": "KuaiRand",
    },
}


def load_model(config, device):
    mapping_path = config["mapping_path"]
    checkpoint_path = config["checkpoint_path"]
    max_seq_len = config["max_seq_len"]
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
    model = TransformerRewardModel(
        num_items=num_items,
        max_seq_len=max_seq_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        padding_idx=0,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    return model, num_items


def calculate_rank_correlation(x, y):
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    return np.corrcoef(x_rank, y_rank)[0, 1]


def build_uncertainty_groups(uncertainties, prediction_errors):
    quantiles = np.quantile(
        uncertainties,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    groups = []
    for i in range(5):
        if i == 4:
            mask = (uncertainties >= quantiles[i]) & (uncertainties <= quantiles[i + 1])
        else:
            mask = (uncertainties >= quantiles[i]) & (uncertainties < quantiles[i + 1])
        g_unc = uncertainties[mask]
        g_err = prediction_errors[mask]
        if mask.sum() == 0:
            raise RuntimeError(f"Uncertainty group {i + 1} contains no samples.")
        groups.append({
            "group": i + 1,
            "samples": int(mask.sum()),
            "uncertainty_mean": float(g_unc.mean()),
            "prediction_error_mean": float(g_err.mean()),
        })
    return groups


def run_dataset(dataset_name, config, device):
    print()
    print("=" * 80)
    print(f"UNCERTAINTY ANALYSIS | {dataset_name}")
    print("=" * 80)

    for key in ["test_path", "mapping_path", "checkpoint_path"]:
        path = config[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{dataset_name}: missing required file:\n{path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/5] Loading item mapping...")
    with open(config["mapping_path"], "r", encoding="utf-8") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
    print(f"[INFO] Number of items: {num_items:,}")

    print("\n[2/5] Loading test dataset...")
    dataset = SequenceDataset(
        sequence_path=config["test_path"],
        mapping_path=config["mapping_path"],
        max_seq_len=config["max_seq_len"],
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[INFO] Test samples: {len(dataset):,}")
    print(f"[INFO] Test batches: {len(loader):,}")

    print("\n[3/5] Loading trained model...")
    model, loaded_num_items = load_model(config=config, device=device)
    if loaded_num_items != num_items:
        raise ValueError(f"{dataset_name}: item mapping mismatch.")
    print("[INFO] Model loaded successfully.")

    print("\n[4/5] Running MC Dropout...")
    print(f"[INFO] MC samples: {MC_SAMPLES}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")

    mean_pred_list = []
    unc_list = []
    target_list = []
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        hist_items = batch["history_items"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        tgt_item = batch["target_item"].to(device)
        tgt_reward = batch["target_reward"].to(device)

        mean_pred, uncertainty, _ = mc_dropout_predict(
            model=model,
            hist_items=hist_items,
            attn_mask=attn_mask,
            tgt_item=tgt_item,
            num_samples=MC_SAMPLES,
        )

        mean_pred_list.append(mean_pred.detach().cpu().numpy())
        unc_list.append(uncertainty.detach().cpu().numpy())
        target_list.append(tgt_reward.detach().cpu().numpy())

        if (batch_idx + 1) % 100 == 0 or batch_idx == 0 or batch_idx + 1 == total_batches:
            progress = (batch_idx + 1) / total_batches * 100
            print(f"  Batch [{batch_idx + 1}/{total_batches}] ({progress:.1f}%)")

    mean_predictions = np.concatenate(mean_pred_list)
    uncertainties = np.concatenate(unc_list)
    targets = np.concatenate(target_list)
    prediction_errors = np.abs(mean_predictions - targets)

    print()
    print("[INFO] Prediction completed.")
    print(f"[INFO] Samples collected: {len(mean_predictions):,}")

    print("\n[5/5] Analyzing uncertainty...")
    pearson_corr = np.corrcoef(uncertainties, prediction_errors)[0, 1]
    spearman_corr = calculate_rank_correlation(uncertainties, prediction_errors)
    groups = build_uncertainty_groups(uncertainties, prediction_errors)

    summary = {
        "dataset": config["display_name"],
        "samples": int(len(targets)),
        "mc_samples": MC_SAMPLES,
        "prediction": {
            "mean": float(mean_predictions.mean()),
            "min": float(mean_predictions.min()),
            "max": float(mean_predictions.max()),
        },
        "target": {
            "mean": float(targets.mean()),
            "min": float(targets.min()),
            "max": float(targets.max()),
        },
        "uncertainty": {
            "mean": float(uncertainties.mean()),
            "std": float(uncertainties.std()),
            "min": float(uncertainties.min()),
            "max": float(uncertainties.max()),
        },
        "prediction_error": {
            "mae": float(prediction_errors.mean()),
            "max": float(prediction_errors.max()),
        },
        "correlation": {
            "pearson": float(pearson_corr),
            "spearman": float(spearman_corr),
        },
        "uncertainty_groups": groups,
    }

    prefix = config["output_prefix"]
    output_npz = os.path.join(OUTPUT_DIR, f"{prefix}_test_uncertainty.npz")
    output_json = os.path.join(OUTPUT_DIR, f"{prefix}_uncertainty_summary.json")
    output_groups_csv = os.path.join(OUTPUT_DIR, f"{prefix}_uncertainty_groups.csv")

    np.savez_compressed(
        output_npz,
        mean_prediction=mean_predictions,
        uncertainty=uncertainties,
        target_reward=targets,
        prediction_error=prediction_errors,
    )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    groups_df = pd.DataFrame(groups)
    groups_df.insert(0, "dataset", dataset_name)
    groups_df.to_csv(output_groups_csv, index=False)

    print()
    print("=" * 80)
    print(f"{dataset_name} Uncertainty Analysis Results")
    print("=" * 80)
    print(f"Samples:             {len(targets):,}")
    print(f"MC Samples:          {MC_SAMPLES}")
    print("\nUncertainty:")
    print(f"  Mean:              {uncertainties.mean():.6f}")
    print(f"  Std:               {uncertainties.std():.6f}")
    print(f"  Min:               {uncertainties.min():.6f}")
    print(f"  Max:               {uncertainties.max():.6f}")
    print("\nPrediction Error:")
    print(f"  MAE:               {prediction_errors.mean():.6f}")
    print(f"  Max:               {prediction_errors.max():.6f}")
    print("\nCorrelation:")
    print(f"  Pearson:           {pearson_corr:.6f}")
    print(f"  Spearman:          {spearman_corr:.6f}")
    print("\nUncertainty Groups:")
    print(f"{'Group':<8}{'Samples':<12}{'Mean Uncertainty':<20}{'Mean Error':<15}")
    print("-" * 55)
    for g in groups:
        print(f"{g['group']:<8}{g['samples']:<12,}{g['uncertainty_mean']:<20.6f}{g['prediction_error_mean']:<15.6f}")
    print("\nSaved files:")
    print(f"  {output_npz}")
    print(f"  {output_json}")
    print(f"  {output_groups_csv}")
    return groups_df


def main():

    global MC_SAMPLES

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "Amazon",
            "KuaiRand",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=MC_SAMPLES,
    )
    args = parser.parse_args()

    if args.mc_samples <= 1:
        raise ValueError("--mc-samples must be > 1")
    MC_SAMPLES = args.mc_samples

    print("=" * 80)
    print("FULL TEST SET UNCERTAINTY ANALYSIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    if args.dataset == "all":
        selected_datasets = ["Amazon", "KuaiRand"]
    else:
        selected_datasets = [args.dataset]

    all_group_frames = []
    for dataset_name in selected_datasets:
        groups_df = run_dataset(
            dataset_name=dataset_name,
            config=DATASETS[dataset_name],
            device=device,
        )
        all_group_frames.append(groups_df)

    if all_group_frames:
        combined_groups = pd.concat(all_group_frames, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "uncertainty_groups_all.csv")
        combined_groups.to_csv(combined_path, index=False)
        print()
        print(f"[INFO] Combined group summary saved:")
        print(f"  {combined_path}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
