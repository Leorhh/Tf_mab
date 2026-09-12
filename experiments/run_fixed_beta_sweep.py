import json
import os
import random
import numpy as np
import pandas as pd
import torch
from src.data.sequence_dataset import SequenceDataset
from src.bandit.adaptive_mab import AdaptiveMAB
from src.data.candidate_generator import CandidateGenerator
from src.models.transformer import TransformerRewardModel
from src.models.candidate_uncertainty import score_candidates_with_uncertainty

SEEDS = [42, 123, 2024, 3407, 7777]
FIXED_BETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
MAX_SAMPLES = 100000
NUM_CANDIDATES = 100
MC_SAMPLES = 10
BETA_TAU = 0.01
HISTORY_WEIGHT = 0.1
OUTPUT_ROOT = "outputs/experiments/fixed_beta_sweep_history_matched"

DATASETS = {
    "Amazon": {
        "sequence_path": "data/processed/amazon/test_sequences.csv",
        "mapping_path": "data/processed/amazon/item_mapping.json",
        "checkpoint_path": "checkpoints/best_transformer.pt",
        "max_seq_len": 20,
    },
    "KuaiRand": {
        "sequence_path": "data/processed/Kuairand/test_sequences.csv",
        "mapping_path": "data/processed/Kuairand/item_mapping.json",
        "checkpoint_path": "checkpoints/best_kuairand_transformer.pt",
        "max_seq_len": 50,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_paths(dataset_name, config):
    for key in ["sequence_path", "mapping_path", "checkpoint_path"]:
        path = config[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{dataset_name}: missing required file:\n{path}")


def load_dataset(config):
    return SequenceDataset(
        sequence_path=config["sequence_path"],
        mapping_path=config["mapping_path"],
        max_seq_len=config["max_seq_len"],
    )


def load_model(config, device):
    with open(config["mapping_path"], "r", encoding="utf-8") as f:
        mapping = json.load(f)
    num_items = mapping["num_items"]
    model = TransformerRewardModel(
        num_items=num_items,
        max_seq_len=config["max_seq_len"],
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        padding_idx=0,
    )
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def method_name(beta):
    return f"Fixed-Beta-{beta:g}"


def create_fixed_bandits():
    return {
        beta: AdaptiveMAB(
            beta_min=beta,
            beta_max=beta,
            tau=BETA_TAU,
            history_weight=HISTORY_WEIGHT,
        )
        for beta in FIXED_BETAS
    }


def run_one_seed(dataset_name, seed, dataset, model, device):
    set_seed(seed)
    candidate_generator = CandidateGenerator(
        num_items=dataset.num_items,
        num_candidates=NUM_CANDIDATES,
        seed=seed,
    )
    fixed_bandits = create_fixed_bandits()
    total_samples = min(len(dataset), MAX_SAMPLES)
    results = []
    print(f"[{dataset_name}] Seed {seed}: {total_samples} samples")

    for sample_index in range(total_samples):
        sample = dataset[sample_index]
        hist_items = sample["history_items"].unsqueeze(0).to(device)
        attn_mask = sample["attention_mask"].unsqueeze(0).to(device)
        target_item = int(sample["target_item"].item())
        target_reward = float(sample["target_reward"].item())
        history = sample["history_items"].cpu().numpy()
        history = history[history != 0].tolist()

        candidates, target_index = candidate_generator.generate(
            target_item=target_item,
            history_items=history,
        )
        candidates_np = np.asarray(candidates, dtype=np.int64)
        candidates_tensor = torch.tensor(candidates_np, dtype=torch.long).unsqueeze(0).to(device)

        mean_pred, uncertainty, _ = score_candidates_with_uncertainty(
            model=model,
            hist_items=hist_items,
            attn_mask=attn_mask,
            candidates=candidates_tensor,
            num_samples=MC_SAMPLES,
        )
        predicted_reward = mean_pred[0].detach().cpu().numpy()
        candidate_uncertainty = uncertainty[0].detach().cpu().numpy()
        greedy_arm = int(np.argmax(predicted_reward))

        for beta in FIXED_BETAS:
            bandit = fixed_bandits[beta]
            selected_arm, scores, actual_beta = bandit.select_arm(
                predicted_reward,
                candidate_uncertainty,
                candidates_np,
            )
            selected_arm = int(selected_arm)
            if not np.isclose(actual_beta, beta, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"{dataset_name} seed={seed}, sample={sample_index}: "
                    f"configured fixed beta {beta}, but AdaptiveMAB returned {actual_beta}."
                )
            selected_item = int(candidates_np[selected_arm])
            selected_target = int(selected_arm == target_index)
            observed_reward = target_reward if selected_target else np.nan
            non_greedy = int(selected_arm != greedy_arm)
            if selected_target:
                bandit.update(selected_item, target_reward)

            results.append({
                "dataset": dataset_name,
                "seed": seed,
                "sample_index": sample_index,
                "method": method_name(beta),
                "configured_beta": beta,
                "actual_beta": float(actual_beta),
                "selected_item": selected_item,
                "target_item": target_item,
                "target_reward": target_reward,
                "selected_target": selected_target,
                "observed_reward": observed_reward,
                "non_greedy": non_greedy,
                "selected_prediction": float(predicted_reward[selected_arm]),
                "selected_uncertainty": float(candidate_uncertainty[selected_arm]),
            })

        if sample_index == 0 or (sample_index + 1) % 5000 == 0 or sample_index + 1 == total_samples:
            print(f"  [{dataset_name}] Seed {seed}: {sample_index + 1}/{total_samples}")

    return pd.DataFrame(results)


def calculate_seed_summary(results_df):
    rows = []
    for beta in FIXED_BETAS:
        method = method_name(beta)
        df = results_df[results_df["method"] == method]
        if df.empty:
            raise ValueError(f"No results found for {method}.")
        unique_beta = df["actual_beta"].dropna().unique()
        if len(unique_beta) != 1 or not np.isclose(unique_beta[0], beta, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{method}: actual beta is not constant at {beta}. Observed values: {unique_beta}"
            )
        valid_reward = df["observed_reward"].dropna()
        rows.append({
            "dataset": df["dataset"].iloc[0],
            "seed": int(df["seed"].iloc[0]),
            "method": method,
            "beta": beta,
            "num_samples": int(len(df)),
            "num_logged_rewards": int(df["observed_reward"].notna().sum()),
            "mean_reward": float(valid_reward.mean()) if len(valid_reward) > 0 else np.nan,
            "target_selection_rate": float(df["selected_target"].mean()),
            "non_greedy_rate": float(df["non_greedy"].mean()),
            "mean_selected_prediction": float(df["selected_prediction"].mean()),
            "mean_selected_uncertainty": float(df["selected_uncertainty"].mean()),
        })
    return pd.DataFrame(rows)


def calculate_final_summary(seed_summary_df):
    rows = []
    for dataset_name in ["Amazon", "KuaiRand"]:
        dataset_df = seed_summary_df[seed_summary_df["dataset"] == dataset_name]
        for beta in FIXED_BETAS:
            df = dataset_df[np.isclose(dataset_df["beta"], beta)]
            if len(df) != len(SEEDS):
                raise ValueError(
                    f"{dataset_name}, beta={beta}: expected {len(SEEDS)} seeds, found {len(df)}."
                )
            rows.append({
                "dataset": dataset_name,
                "method": method_name(beta),
                "beta": beta,
                "num_seeds": len(df),
                "mean_reward_mean": float(df["mean_reward"].mean()),
                "mean_reward_std": float(df["mean_reward"].std(ddof=1)),
                "target_selection_rate_mean": float(df["target_selection_rate"].mean()),
                "target_selection_rate_std": float(df["target_selection_rate"].std(ddof=1)),
                "non_greedy_rate_mean": float(df["non_greedy_rate"].mean()),
                "non_greedy_rate_std": float(df["non_greedy_rate"].std(ddof=1)),
                "mean_selected_prediction_mean": float(df["mean_selected_prediction"].mean()),
                "mean_selected_uncertainty_mean": float(df["mean_selected_uncertainty"].mean()),
            })
    return pd.DataFrame(rows)


def validate_seed_summary(seed_summary_df):
    print()
    print("=" * 70)
    print("FIXED‑BETA SUMMARY SANITY CHECK")
    print("=" * 70)
    expected_rows = len(DATASETS) * len(SEEDS) * len(FIXED_BETAS)
    if len(seed_summary_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} seed‑summary rows, found {len(seed_summary_df)}."
        )
    for dataset_name in ["Amazon", "KuaiRand"]:
        for seed in SEEDS:
            subset = seed_summary_df[
                (seed_summary_df["dataset"] == dataset_name) & (seed_summary_df["seed"] == seed)
            ]
            if len(subset) != len(FIXED_BETAS):
                raise ValueError(
                    f"{dataset_name} seed={seed}: expected {len(FIXED_BETAS)} beta settings, found {len(subset)}."
                )
        print(f"[OK] {dataset_name}: all {len(SEEDS)} seeds contain all {len(FIXED_BETAS)} fixed‑beta settings.")


def save_metadata():
    metadata = {
        "experiment": "History‑Matched Fixed‑Beta Sweep",
        "datasets": ["Amazon", "KuaiRand"],
        "split": "test",
        "seeds": SEEDS,
        "fixed_betas": FIXED_BETAS,
        "max_samples_per_seed": MAX_SAMPLES,
        "num_candidates": NUM_CANDIDATES,
        "mc_samples": MC_SAMPLES,
        "tau": BETA_TAU,
        "history_weight": HISTORY_WEIGHT,
        "fixed_beta_implementation": "AdaptiveMAB with beta_min == beta_max == configured beta",
        "scoring_rule": "predicted_reward + beta * uncertainty + history_weight * historical_mean_reward",
        "feedback_protocol": "logged‑target‑only",
        "fairness_control": (
            "Fixed‑beta and Adaptive‑MAB use the same AdaptiveMAB class, "
            "history term, update rule, candidate sets, and MC‑dropout predictions. "
            "The intended difference is whether beta is fixed or adaptive."
        ),
        "adaptive_reference": {
            "beta_min": 0.1,
            "beta_max": 0.5,
            "tau": 0.01,
            "history_weight": 0.1,
            "source": "Existing 5‑seed mechanism‑ablation results.",
        },
    }
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    path = os.path.join(OUTPUT_ROOT, "experiment_config.json")
    with open(path, "w", encoding="utf‑8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved: {path}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("HISTORY‑MATCHED FIXED‑BETA SWEEP")
    print("=" * 70)
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Seeds: {SEEDS}")
    print(f"[INFO] Fixed betas: {FIXED_BETAS}")
    print(f"[INFO] Max samples/seed: {MAX_SAMPLES}")
    print(f"[INFO] Candidates: {NUM_CANDIDATES}")
    print(f"[INFO] MC samples: {MC_SAMPLES}")
    print(f"[INFO] History weight: {HISTORY_WEIGHT}")

    save_metadata()
    all_seed_summaries = []

    for dataset_name, config in DATASETS.items():
        print()
        print("=" * 70)
        print(dataset_name.upper())
        print("=" * 70)
        validate_paths(dataset_name, config)
        output_dir = os.path.join(OUTPUT_ROOT, dataset_name.lower())
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Loading {dataset_name} dataset...")
        dataset = load_dataset(config)
        total_samples = min(len(dataset), MAX_SAMPLES)
        print(f"[INFO] Dataset samples: {len(dataset)}")
        print(f"[INFO] Samples/seed: {total_samples}")
        print(f"[INFO] Loading {dataset_name} model...")
        model = load_model(config, device)
        print("[INFO] Model loaded.")

        for seed in SEEDS:
            print()
            print(f"{'=' * 20} {dataset_name} SEED {seed} {'=' * 20}")
            seed_results = run_one_seed(
                dataset_name=dataset_name,
                seed=seed,
                dataset=dataset,
                model=model,
                device=device,
            )
            raw_path = os.path.join(output_dir, f"seed_{seed}.csv")
            seed_results.to_csv(raw_path, index=False)
            print(f"[INFO] Saved: {raw_path}")

            seed_summary = calculate_seed_summary(seed_results)
            summary_path = os.path.join(output_dir, f"seed_{seed}_summary.csv")
            seed_summary.to_csv(summary_path, index=False)
            print(f"[INFO] Saved: {summary_path}")
            all_seed_summaries.append(seed_summary)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_summary_df = pd.concat(all_seed_summaries, ignore_index=True)
    validate_seed_summary(seed_summary_df)

    seed_summary_path = os.path.join(OUTPUT_ROOT, "seed_summary.csv")
    seed_summary_df.to_csv(seed_summary_path, index=False)

    final_summary_df = calculate_final_summary(seed_summary_df)
    final_summary_path = os.path.join(OUTPUT_ROOT, "final_summary.csv")
    final_summary_df.to_csv(final_summary_path, index=False)

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(final_summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("[INFO] Saved:")
    print(f"  {seed_summary_path}")
    print(f"  {final_summary_path}")
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
