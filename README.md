# Tf_mab — Transformer Reward Model + Adaptive Multi-Armed Bandit

Combining Transformer-based sequential reward modeling with multi-armed
bandit exploration for recommendation. A Transformer encodes the
interaction history and predicts a reward for each candidate item;
MC Dropout turns those predictions into uncertainty estimates; and a
bandit layer — whose exploration coefficient adapts to the current
uncertainty — picks what to show. Evaluated on Amazon Electronics and
KuaiRand with multi-seed comparisons, ablations, and significance tests.

## Method

```
interaction history ──► Transformer encoder ──► user embedding ──┐
                                                                  ├─► reward head ──► r̂ ∈ [0,1]
candidate item embedding ────────────────────────────────────────┘
                        │
                        ▼  MC Dropout, T forward passes
            predictive mean r̂  +  uncertainty σ
                        │
                        ▼  AdaptiveMAB scoring
        score = r̂ + β(σ̄) · σ + w · mean_history_reward
```

The exploration weight is not fixed. It follows the mean uncertainty σ̄
of the current candidate set through a saturating map:

```
β(σ̄) = β_min + (β_max − β_min) · σ̄ / (σ̄ + τ)
```

so the policy explores harder exactly when the model is unsure, and
settles into exploitation as predictions become confident. The
saturating form keeps β inside `[β_min, β_max]` regardless of how
uncertainty is scaled on a given dataset; `τ` controls how fast it
saturates and is set relative to the typical σ observed after training.

Baselines implemented for comparison: Random, ε-Greedy, UCB
(prediction + `√(log t / n)` bonus), and a hybrid Thompson Sampling
(Beta posterior blended with model predictions).

## Results

All bandit numbers are mean ± std over 5 seeds, 100 candidates per step.

**Reward prediction (test set)**

| Dataset | MSE | MAE | RMSE |
|---|---|---|---|
| Amazon Electronics | 0.1014 | 0.2349 | 0.3184 |
| KuaiRand | 0.0260 | 0.1163 | 0.1612 |

**Main bandit comparison (mean observed reward)**

| Method | Amazon | KuaiRand |
|---|---|---|
| Random | 0.7910 ± 0.0074 | 0.0798 ± 0.0035 |
| ε-Greedy | 0.9063 ± 0.0018 | 0.1750 ± 0.0027 |
| Predictive UCB | 0.9020 ± 0.0076 | 0.1776 ± 0.0135 |
| Hybrid Thompson | 0.8706 ± 0.0071 | 0.1213 ± 0.0085 |
| Transformer-Greedy | 0.9104 ± 0.0023 | 0.1841 ± 0.0094 |
| **AdaptiveMAB** | **0.9090 ± 0.0020** | **0.1804 ± 0.0152** |

AdaptiveMAB trades a statistically insignificant amount of immediate
reward (paired t-test vs. Greedy: p = 0.105 on Amazon, p = 0.403 on
KuaiRand) for sustained exploration: it keeps a non-greedy selection
rate of 19.0% (Amazon) / 9.7% (KuaiRand) and, on Amazon, actually
selects the logged target item more often than Greedy (3.99% vs.
3.20%, p < 0.001). Against a fixed-β variant with the same scoring
rule, the adaptive β wins on KuaiRand (+0.13 pp reward, p = 0.066)
with no loss on Amazon.

**Ranking (100 candidates, target included)**

| Dataset | Hit@1 | Hit@5 | Hit@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| Amazon | 0.0320 | 0.1177 | 0.1994 | 0.1010 | 0.0955 |
| KuaiRand | 0.0094 | 0.0490 | 0.1014 | 0.0454 | 0.0513 |

Sensitivity analyses over candidate-set size {20, 50, 100, 200} and MC
Dropout samples {5, 10, 20} are in `experiments/`; the frozen result
tables and figures are under `outputs/`.

## Repository layout

```
train.py / train_kuairand.py     # reward model training (Amazon / KuaiRand)
evaluate_transformer.py          # reward prediction metrics
evaluate_candidate_ranking.py    # Hit@K / NDCG / MRR over 100 candidates
test_uncertainty.py              # MC Dropout sanity check on one batch
src/
  models/                        # RewardTransformer, candidate scoring, MC Dropout
  bandit/                        # AdaptiveMAB + Random/ε-Greedy/UCB/Thompson
  data/                          # cleaning, sequence building, packing, verification
experiments/                     # all benchmark / ablation / sensitivity scripts
tests/                           # unit and integration tests
outputs/                         # frozen result tables and figures
checkpoints/                     # trained weights (created by training)
```

## Running it

```bash
pip install -r requirements.txt
```

Data is not included; the expected layout is described in
`src/data/` scripts. The pipeline for each dataset is:
clean → build sequences → (optionally) pack:

```bash
python -m src.data.pack_sequences \
    data/processed/KuaiRand/train_sequences.csv \
    data/processed/KuaiRand/item_mapping.json \
    data/processed/KuaiRand/packed_train --max-seq-len 50
```

Packing converts the CSV into memory-mapped arrays so training memory
stays flat regardless of dataset size — this is what allows full
KuaiRand on a 16 GB machine. If no packed directory exists, the
training scripts fall back to reading the CSVs directly.

```bash
python train.py                 # Amazon
python train_kuairand.py        # KuaiRand
python evaluate_transformer.py
python evaluate_candidate_ranking.py
cd experiments && python run_mab_formal.py
```

CUDA, Apple Silicon (MPS), and CPU are all supported; the scripts pick
the best available device automatically.

## Limitations

- Evaluation is offline replay: a logged reward is available only for
  the item that was actually shown, so candidate sets always contain
  the logged target and bandit feedback comes from logged data. This
  biases absolute reward estimates toward the logging policy; relative
  comparisons between strategies remain informative, but the numbers
  should not be read as online performance.
- MC Dropout is the cheapest uncertainty approximation available and
  its estimates are not calibrated; the `T` sensitivity experiment is
  there precisely because of this.
- The adaptive-β rule is a heuristic — unlike UCB-style methods it
  carries no regret guarantee.
- Comparisons are against classic bandit baselines; learned neural
  bandits (NeuralUCB, NeuralTS) are not included.

## Citation

```bibtex
@misc{tfmab2026,
  author = {Leorhh},
  title  = {Tf_mab: Transformer-based Reward Modeling with Adaptive Exploration for Multi-Armed Bandits},
  year   = {2026},
  url    = {https://github.com/Leorhh/Tf_mab}
}
```
