import os
import json
import random
import numpy as np
import pandas as pd
import torch
from src.data.sequence_dataset import SequenceDataset
from src.models.transformer import TransformerRewardModel
from src.models.candidate_uncertainty import score_candidates_with_uncertainty
from src.data.candidate_generator import CandidateGenerator
from src.bandit.adaptive_mab import AdaptiveMAB
# Configuration
DATASETS = {
    "Amazon": {
        "test_path": "data/processed/amazon/test_sequences.csv",
        "mapping_path": "data/processed/amazon/item_mapping.json",
        "checkpoint": "checkpoints/best_transformer.pt",
        "max_seq_len": 20,
        "max_samples": 100000,
    },
    "KuaiRand": {
        "test_path": "data/processed/Kuairand/test_sequences.csv",
        "mapping_path": "data/processed/Kuairand/item_mapping.json",
        "checkpoint": "checkpoints/best_kuairand_transformer.pt",
        "max_seq_len": 50,
        "max_samples": 27285,
    },
}
MC_SAMPLES_LIST = [5, 10, 20]
CANDIDATE_SIZE = 100
SEEDS = [123, 2024, 3407, 42, 7777]
INFERENCE_BATCH_SIZE = 4
BETA_MIN = 0.1
BETA_MAX = 0.5
TAU = 0.01
HISTORY_WEIGHT = 0.1
OUTPUT_ROOT = "outputs/experiments/mc_dropout_sensitivity"
# Utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
def load_model(checkpoint_path, num_items, max_seq_len, device):
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
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
def get_output_path(
    dataset_name,
    mc_samples,
    seed,
):
    dataset_dir = os.path.join(
        OUTPUT_ROOT,
        dataset_name.lower(),
    )
    os.makedirs(
        dataset_dir,
        exist_ok=True,
    )
    return os.path.join(
        dataset_dir,
        f"mc_{mc_samples}_seed_{seed}.csv",
    )
def save_result(output_path, result):
    pd.DataFrame([result]).to_csv(
        output_path,
        index=False,
    )
# Evaluation
def evaluate_one_setting(
    model,
    dataset,
    mc_samples,
    seed,
    num_items,
    max_samples,
    device,
):
    candidate_generator = CandidateGenerator(
        num_items=num_items,
        num_candidates=CANDIDATE_SIZE,
        seed=seed,
    )
    mab = AdaptiveMAB(
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        tau=TAU,
        history_weight=HISTORY_WEIGHT,
    )
    total_samples = min(
        len(dataset),
        max_samples,
    )
    target_selected = 0
    non_greedy = 0
    selected_rewards = []
    selected_predictions = []
    selected_uncertainties = []
    beta_values = []
    for batch_start in range(
        0,
        total_samples,
        INFERENCE_BATCH_SIZE,
    ):
        batch_end = min(
            batch_start + INFERENCE_BATCH_SIZE,
            total_samples,
        )
        histories = []
        masks = []
        candidates_list = []
        target_items = []
        target_rewards = []
        for idx in range(
            batch_start,
            batch_end,
        ):
            sample = dataset[idx]
            hist = sample["history_items"]
            mask = sample["attention_mask"]
            target_item = int(
                sample["target_item"].item()
            )
            target_reward = float(
                sample["target_reward"].item()
            )
            candidates, _ = candidate_generator.generate(
                target_item=target_item,
                history_items=hist.tolist(),
            )
            histories.append(hist)
            masks.append(mask)
            candidates_list.append(candidates)
            target_items.append(target_item)
            target_rewards.append(target_reward)
        hist_batch = torch.stack(
            histories
        ).to(
            device,
            non_blocking=True,
        )
        mask_batch = torch.stack(
            masks
        ).to(
            device,
            non_blocking=True,
        )
        candidate_batch = torch.tensor(
            np.stack(candidates_list),
            dtype=torch.long,
            device=device,
        )
        mean_predictions, uncertainties, _ = (
            score_candidates_with_uncertainty(
                model=model,
                hist_items=hist_batch,
                attn_mask=mask_batch,
                candidates=candidate_batch,
                num_samples=mc_samples,
            )
        )
        mean_predictions = (
            mean_predictions
            .cpu()
            .numpy()
        )
        uncertainties = (
            uncertainties
            .cpu()
            .numpy()
        )
        for local_idx in range(
            len(target_items)
        ):
            candidates = candidates_list[
                local_idx
            ]
            predictions = mean_predictions[
                local_idx
            ]
            uncertainty = uncertainties[
                local_idx
            ]
            selected_idx, scores, beta = (
                mab.select_arm(
                    predicted_reward=predictions,
                    uncertainty=uncertainty,
                    candidate_items=candidates,
                )
            )
            selected_item = int(
                candidates[selected_idx]
            )
            greedy_idx = int(
                np.argmax(predictions)
            )
            greedy_item = int(
                candidates[greedy_idx]
            )
            if selected_item != greedy_item:
                non_greedy += 1
            if selected_item == target_items[
                local_idx
            ]:
                target_selected += 1
                reward = target_rewards[
                    local_idx
                ]
                selected_rewards.append(
                    reward
                )
                mab.update(
                    item_id=selected_item,
                    reward=reward,
                )
            selected_predictions.append(
                float(
                    predictions[selected_idx]
                )
            )
            selected_uncertainties.append(
                float(
                    uncertainty[selected_idx]
                )
            )
            beta_values.append(
                float(beta)
            )
        if (
            (batch_end % 5000 == 0)
            or batch_end == total_samples
        ):
            progress = (
                batch_end
                / total_samples
                * 100
            )
            print(
                f"\r    Progress: "
                f"{batch_end:,}/"
                f"{total_samples:,} "
                f"({progress:.1f}%)",
                end="",
                flush=True,
            )
    print()
    mean_reward = (
        float(np.mean(selected_rewards))
        if selected_rewards
        else 0.0
    )
    target_selection_rate = (
        target_selected
        / total_samples
    )
    non_greedy_rate = (
        non_greedy
        / total_samples
    )
    mean_selected_prediction = (
        float(
            np.mean(
                selected_predictions
            )
        )
        if selected_predictions
        else 0.0
    )
    mean_selected_uncertainty = (
        float(
            np.mean(
                selected_uncertainties
            )
        )
        if selected_uncertainties
        else 0.0
    )
    mean_beta = (
        float(
            np.mean(beta_values)
        )
        if beta_values
        else 0.0
    )
    return {
        "dataset": None,
        "mc_samples": mc_samples,
        "candidate_size": CANDIDATE_SIZE,
        "seed": seed,
        "samples": total_samples,
        "mean_reward": mean_reward,
        "target_selection_rate": target_selection_rate,
        "non_greedy_rate": non_greedy_rate,
        "mean_selected_prediction":
            mean_selected_prediction,
        "mean_selected_uncertainty":
            mean_selected_uncertainty,
        "mean_beta": mean_beta,
    }
# Main
def main():
    print("=" * 70)
    print("MC Dropout Sensitivity Analysis")
    print("=" * 70)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(
        f"[INFO] Device: {device}"
    )
    if device.type == "cuda":
        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
    print(
        f"[INFO] Candidate size: "
        f"{CANDIDATE_SIZE}"
    )
    print(
        f"[INFO] MC settings: "
        f"{MC_SAMPLES_LIST}"
    )
    print(
        f"[INFO] Inference batch size: "
        f"{INFERENCE_BATCH_SIZE}"
    )
    all_results = []
    for dataset_name, config in DATASETS.items():
        print()
        print("=" * 70)
        print(
            f"DATASET: {dataset_name}"
        )
        print("=" * 70)
        with open(
            config["mapping_path"],
            "r",
            encoding="utf-8",
        ) as f:
            mapping = json.load(f)
        num_items = mapping[
            "num_items"
        ]
        print(
            f"[INFO] Number of items: "
            f"{num_items:,}"
        )
        dataset = SequenceDataset(
            sequence_path=config[
                "test_path"
            ],
            mapping_path=config[
                "mapping_path"
            ],
            max_seq_len=config[
                "max_seq_len"
            ],
        )
        print(
            f"[INFO] Test samples: "
            f"{len(dataset):,}"
        )
        model = load_model(
            checkpoint_path=config[
                "checkpoint"
            ],
            num_items=num_items,
            max_seq_len=config[
                "max_seq_len"
            ],
            device=device,
        )
        for mc_samples in MC_SAMPLES_LIST:
            print()
            print("-" * 70)
            print(
                f"{dataset_name} | "
                f"MC Dropout={mc_samples}"
            )
            print("-" * 70)
            for seed in SEEDS:
                output_path = get_output_path(
                    dataset_name,
                    mc_samples,
                    seed,
                )
                if os.path.exists(
                    output_path
                ):
                    print(
                        f"[SKIP] "
                        f"MC={mc_samples}, "
                        f"seed={seed}"
                    )
                    try:
                        existing = pd.read_csv(
                            output_path
                        )
                        if len(existing) > 0:
                            all_results.append(
                                existing.iloc[
                                    0
                                ].to_dict()
                            )
                    except Exception as e:
                        print(
                            f"[WARN] "
                            f"Could not read "
                            f"{output_path}: {e}"
                        )
                    continue
                print()
                print(
                    f"[RUN] {dataset_name} | "
                    f"MC={mc_samples} | "
                    f"seed={seed}"
                )
                set_seed(seed)
                try:
                    result = (
                        evaluate_one_setting(
                            model=model,
                            dataset=dataset,
                            mc_samples=mc_samples,
                            seed=seed,
                            num_items=num_items,
                            max_samples=config[
                                "max_samples"
                            ],
                            device=device,
                        )
                    )
                    result[
                        "dataset"
                    ] = dataset_name
                    save_result(
                        output_path,
                        result,
                    )
                    all_results.append(
                        result
                    )
                    print(
                        f"[RESULT] "
                        f"reward="
                        f"{result['mean_reward']:.6f}, "
                        f"target="
                        f"{result['target_selection_rate']:.6f}, "
                        f"exploration="
                        f"{result['non_greedy_rate']:.6f}, "
                        f"uncertainty="
                        f"{result['mean_selected_uncertainty']:.6f}, "
                        f"beta="
                        f"{result['mean_beta']:.6f}"
                    )
                    print(
                        f"[SAVED] "
                        f"{output_path}"
                    )
                except RuntimeError as e:
                    if (
                        "out of memory"
                        in str(e).lower()
                    ):
                        print()
                        print(
                            "[ERROR] "
                            "CUDA OOM."
                        )
                        print(
                            "[ACTION] "
                            "Set "
                            "INFERENCE_BATCH_SIZE "
                            "= 2."
                        )
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    raise
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if not all_results:
        print(
            "[WARN] No results found."
        )
        return
    summary_df = pd.DataFrame(
        all_results
    )
    summary_df = (
        summary_df
        .drop_duplicates(
            subset=[
                "dataset",
                "mc_samples",
                "candidate_size",
                "seed",
            ],
            keep="last",
        )
    )
    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )
    all_results_path = os.path.join(
        OUTPUT_ROOT,
        "all_results.csv",
    )
    summary_df.to_csv(
        all_results_path,
        index=False,
    )
    final_summary = (
        summary_df
        .groupby(
            [
                "dataset",
                "mc_samples",
                "candidate_size",
            ]
        )
        .agg(
            num_seeds=(
                "seed",
                "count",
            ),
            mean_reward_mean=(
                "mean_reward",
                "mean",
            ),
            mean_reward_std=(
                "mean_reward",
                "std",
            ),
            target_selection_rate_mean=(
                "target_selection_rate",
                "mean",
            ),
            target_selection_rate_std=(
                "target_selection_rate",
                "std",
            ),
            non_greedy_rate_mean=(
                "non_greedy_rate",
                "mean",
            ),
            non_greedy_rate_std=(
                "non_greedy_rate",
                "std",
            ),
            mean_selected_prediction_mean=(
                "mean_selected_prediction",
                "mean",
            ),
            mean_selected_uncertainty_mean=(
                "mean_selected_uncertainty",
                "mean",
            ),
            mean_beta_mean=(
                "mean_beta",
                "mean",
            ),
        )
        .reset_index()
    )
    final_summary = final_summary.fillna(
        0.0
    )
    final_summary_path = os.path.join(
        OUTPUT_ROOT,
        "final_summary.csv",
    )
    final_summary.to_csv(
        final_summary_path,
        index=False,
    )
    print()
    print(
        final_summary.to_string(
            index=False
        )
    )
    print()
    print(
        f"[SAVED] "
        f"{all_results_path}"
    )
    print(
        f"[SAVED] "
        f"{final_summary_path}"
    )
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
if __name__ == "__main__":
    main()
