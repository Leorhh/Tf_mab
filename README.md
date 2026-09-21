# Tf_mab — Transformer + Multi-Armed Bandit 序列推荐

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Task-Contextual%20Bandit-green" alt="Task">
</p>

将 **Transformer 序列建模** 与 **多臂老虎机（Multi-Armed Bandit, MAB）探索策略** 结合，用于序列推荐 / 上下文老虎机场景。
项目使用 Transformer 对交互历史进行编码并预测候选物品的奖励，再通过 **MC Dropout** 估计预测不确定性，
最终由核心算法 **AdaptiveMAB** 根据不确定性**自适应地调节探索-利用权衡**，在 Amazon 与 KuaiRand 两个真实数据集上完成系统性的对比、消融与统计验证实验。

---

## ✨ 核心亮点

- **Transformer 奖励模型**：基于物品 ID 序列 + 位置编码的 Encoder 结构，对交互历史做 mean-pooling 得到用户表示，与候选物品 Embedding 拼接后经 MLP 头输出 `[0,1]` 区间的奖励预测。
- **不确定性量化**：通过 **MC Dropout** 多次随机前向，用预测标准差作为不确定性，无需修改模型结构、即插即用。
- **AdaptiveMAB（核心方法）**：探索系数 `beta` 不再固定，而是随候选集**平均不确定性**自适应变化（`beta = beta_min + (beta_max - beta_min) * unc / (unc + tau)`），并融合历史经验奖励进行打分。
- **完整基线实现**：Random、Epsilon-Greedy、UCB、Thompson Sampling，与 AdaptiveMAB 公平对比。
- **双数据集**：Amazon 商品推荐序列、KuaiRand 快手短视频推荐（真实曝光/点击日志，适合 bandit 评估）。
- **可复现实验**：`experiments/` 内置多随机种子、机制消融、超参敏感性、统计显著性检验与论文图表生成脚本。

## 🧠 方法概览

```
交互历史序列 ──► Transformer Encoder ──► 用户表示 ──┐
                                                    ├─► Reward Head ──► 奖励预测 r̂ ∈ [0,1]
候选物品 Embedding ────────────────────────────────┘
        │
        ▼  MC Dropout × T 次采样
奖励预测均值 r̂  +  不确定性 σ
        │
        ▼  AdaptiveMAB 打分
score = r̂ + β(σ̄) · σ + w · history_reward   ──► 选择得分最高的臂
        │
        ▼ 环境反馈奖励
在线更新 bandit 统计（计数 / 奖励和 / 累计收益）
```

**AdaptiveMAB 打分公式**（`src/bandit/adaptive_mab.py`）：

```
β(σ̄)  = β_min + (β_max - β_min) · σ̄ / (σ̄ + τ)      # 不确定性越大 → 探索越积极
score = r̂ + β(σ̄) · σ + w · mean_history_reward        # 利用 + 探索 + 历史经验
```

- `σ̄`：当前候选集的平均不确定性；`τ`：调节曲线平滑度的温度系数
- `w`：历史经验奖励的权重（`history_weight`）
- 奖励统一 clip 到 `[0,1]`

## 📁 项目结构

```
Tf_mab/
├── train.py                      # Amazon 数据集训练入口
├── train_kuairand.py             # KuaiRand 数据集训练入口
├── evaluate_transformer.py       # 奖励预测评估（MSE / MAE / RMSE）
├── evaluate_candidate_ranking.py # 候选排序评估（Hit@K / NDCG@10 / MRR）
├── test_uncertainty.py           # MC Dropout 不确定性单元测试
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── transformer.py            # TransformerRewardModel 奖励模型
│   │   ├── uncertainty.py            # MC Dropout 不确定性估计
│   │   ├── candidate_uncertainty.py  # 候选集批量不确定性打分
│   │   └── candidate_scorer.py       # 候选打分器
│   ├── bandit/
│   │   ├── adaptive_mab.py       # ★ 自适应探索 MAB（核心方法）
│   │   ├── ucb.py                # UCB 基线
│   │   ├── thompson.py           # Thompson Sampling 基线
│   │   ├── epsilon_greedy.py     # ε-Greedy 基线
│   │   └── random.py             # Random 基线
│   ├── data/
│   │   ├── amazon_sequence.py            # Amazon 序列构建
│   │   ├── amazon_core.py                # Amazon 核心处理
│   │   ├── kuairand_sequence_preprocessor.py  # KuaiRand 序列预处理
│   │   ├── clean_kuairand.py / stats_kuairand.py / verify_kuairand.py
│   │   ├── sequence_dataset.py / sequence_builder.py
│   │   ├── candidate_generator.py / item_mapping.py
│   │   └── ...（数据检查与验证脚本）
│   ├── trainer/                  # （预留）训练器模块
│   └── utils/                    # （预留）工具模块
├── experiments/                  # 完整实验与论文图表生成
│   ├── run_mab_formal.py             # 正式 MAB 对比实验
│   ├── run_mab_multiseed.py          # 多随机种子实验
│   ├── run_mechanism_ablation.py     # 机制消融
│   ├── run_uncertainty_ablation.py   # 不确定性消融
│   ├── run_adaptive_ablation.py      # 自适应策略消融
│   ├── run_adaptive_beta_ablation.py # 自适应 β 消融
│   ├── run_fixed_beta_sweep.py       # 固定 β 扫描
│   ├── run_candidate_size_sensitivity.py  # 候选集大小敏感性
│   ├── run_mc_dropout_sensitivity.py # MC Dropout 采样次数敏感性
│   ├── run_kuairand_ranking.py       # KuaiRand 排序实验
│   ├── run_kuairand_multiseed.py     # KuaiRand 多种子实验
│   ├── statistical_analysis.py       # 统计显著性分析
│   ├── paired_difference_ci.py       # 配对差异置信区间
│   ├── verify_all_results.py         # 结果一致性校验
│   └── build_paper_figures.py / build_paper_tables.py / build_ieee_figures*.py
├── outputs/                      # 实验输出（结果 / 图表）
└── checkpoints/                  # 模型权重（训练后自动生成）
```

## 🛠️ 安装

```bash
git clone https://github.com/Leorhh/Tf_mab.git
cd Tf_mab

# 建议使用虚拟环境
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

依赖：`numpy` · `torch` · `torchvision` · `pandas` · `scikit-learn` · `scipy` · `pyyaml` · `tqdm` · `matplotlib` · `seaborn` · `tensorboard`

## 📦 数据准备

仓库不附带原始数据，需准备为如下目录结构（训练脚本按此路径读取）：

```
data/processed/
├── amazon/
│   ├── train_sequences.csv
│   ├── val_sequences.csv
│   ├── test_sequences.csv
│   └── item_mapping.json
└── KuaiRand/
    ├── train_sequences.csv
    ├── val_sequences.csv
    ├── test_sequences.csv
    └── item_mapping.json
```

- **Amazon**：使用 `src/data/amazon_sequence.py` / `amazon_core.py` 从原始评论数据构建交互序列；
- **KuaiRand**：使用 `src/data/clean_kuairand.py` → `kuairand_sequence_preprocessor.py` 生成序列，
  并可用 `stats_kuairand.py` / `verify_kuairand.py` 检查数据分布与一致性。

CSV 每行是一条样本，至少包含：`history_items`（历史物品 ID 序列）、`target_item`（目标物品）、`target_reward`（归一化到 `[0,1]` 的奖励）。

## 🚀 快速开始

### 1. 训练 Transformer 奖励模型

```bash
# Amazon（序列长度 20）
python train.py

# KuaiRand（序列长度 50）
python train_kuairand.py
```

训练使用 AdamW + MSELoss，最优验证集权重自动保存：

| 数据集 | Checkpoint |
|---|---|
| Amazon | `checkpoints/best_transformer.pt` |
| KuaiRand | `checkpoints/best_kuairand_transformer.pt` |

### 2. 评估奖励预测精度

```bash
python evaluate_transformer.py
# 输出 MSE / MAE / RMSE，以及预测与目标的均值、范围
```

### 3. 评估候选排序能力

```bash
python evaluate_candidate_ranking.py
# 每个样本构造 100 个候选（含目标物品），输出 Hit@1/5/10、NDCG@10、MRR 与 Rank 分布
```

### 4. 运行 MAB 实验

```bash
cd experiments
python run_mab_formal.py               # 正式对比：AdaptiveMAB vs 各基线
python run_mab_multiseed.py            # 多种子（均值±方差）
python run_mechanism_ablation.py       # 消融：各组件的贡献
python run_uncertainty_ablation.py     # 消融：不确定性项
python run_adaptive_beta_ablation.py   # 消融：自适应 β
python run_fixed_beta_sweep.py         # 敏感性：固定 β 扫描
python run_candidate_size_sensitivity.py  # 敏感性：候选集大小 {20, 50, 100, ...}
python run_mc_dropout_sensitivity.py   # 敏感性：MC Dropout 采样次数 T
python verify_all_results.py           # 校验全部结果一致性
```

### 5. 统计分析与论文图表

```bash
python statistical_analysis.py         # 显著性检验
python paired_difference_ci.py         # 配对差异置信区间
python build_paper_figures.py          # 生成论文图
python build_paper_tables.py           # 生成论文表
python build_ieee_figures_final.py     # IEEE 风格图表
```

## 🎰 Bandit 算法一览

| 算法 | 文件 | 探索机制 |
|---|---|---|
| Random | `src/bandit/random.py` | 纯随机探索 |
| Epsilon-Greedy | `src/bandit/epsilon_greedy.py` | 以 ε 概率随机，否则贪心 |
| UCB | `src/bandit/ucb.py` | 置信上界加成 |
| Thompson Sampling | `src/bandit/thompson.py` | 后验采样 |
| **AdaptiveMAB** ★ | `src/bandit/adaptive_mab.py` | **不确定性自适应 β + 历史经验融合** |

AdaptiveMAB 关键参数：

```python
from src.bandit.adaptive_mab import AdaptiveMAB

mab = AdaptiveMAB(
    beta_min=0.1,       # 最小探索系数
    beta_max=1.0,       # 最大探索系数
    tau=0.01,           # 温度系数，控制 β 随不确定性的变化速率
    history_weight=0.2, # 历史经验奖励权重
)
action, scores, beta = mab.select_arm(predicted_reward, uncertainty, candidate_items)
mab.update(item_id, reward)   # 收到环境反馈后更新统计
```

## 📏 默认超参数

| 参数 | Amazon | KuaiRand |
|---|---|---|
| `MAX_SEQ_LEN` | 20 | 50 |
| `BATCH_SIZE` | 256 | 256 |
| `D_MODEL` | 128 | 128 |
| `N_HEAD` / `NUM_LAYERS` | 4 / 2 | 4 / 2 |
| `DIM_FEEDFORWARD` | 256 | 256 |
| `DROPOUT` | 0.1 | 0.1 |
| `LR` / `WEIGHT_DECAY` | 1e-3 / 1e-4 | 1e-3 / 1e-4 |
| `EPOCHS` | 5 | 5 |

## 📊 评估指标

- **奖励预测**：MSE、MAE、RMSE
- **候选排序**：Hit@1 / Hit@5 / Hit@10、NDCG@10、MRR、Rank 分布（Mean / Median / P90 / P95）
- **Bandit 在线表现**：累计收益（Cumulative Reward）、平均奖励（Mean Reward）、
  多种子均值±标准差、配对差异置信区间与显著性检验

## ⚠️ 说明

- `main.py` 为 IDE 自动生成的占位文件，项目入口为 `train.py` / `train_kuairand.py`；
- `src/trainer/` 与 `src/utils/` 为预留模块（当前为空实现）；
- `outputs/` 与 `checkpoints/` 为运行时生成目录，不会提交到仓库。

## 📜 License

本项目暂未指定开源许可证，如需使用请与作者联系。

## 🙏 引用

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@misc{tfmab2026,
  author = {Leorhh},
  title  = {Tf_mab: Transformer-based Reward Modeling with Adaptive Exploration for Multi-Armed Bandits},
  year   = {2026},
  url    = {https://github.com/Leorhh/Tf_mab}
}
```
