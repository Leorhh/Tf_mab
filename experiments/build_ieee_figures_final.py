import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIXED_BETA_SWEEP_PATH = ("outputs/experiments/""fixed_beta_sweep_history_matched/""final_summary.csv")
MECHANISM_TABLE_PATH = ("outputs/paper_tables/""table_4_mechanism_ablation.csv")
TABLE_DIR = "outputs/paper_tables"
FIGURE_DIR = "outputs/paper_figures_ieee_final"
os.makedirs(FIGURE_DIR, exist_ok=True)

DOUBLE_COL_WIDTH = 7.16

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.1,
    "lines.markersize": 4,
    "hatch.linewidth": 0.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

COLORS = {"Amazon": "#0072B2", "KuaiRand": "#D55E00"}
MARKERS = {"Amazon": "o", "KuaiRand": "s"}
HATCHES = {"Amazon": "", "KuaiRand": "//"}


def load_table(filename):
    path = os.path.join(TABLE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required table not found: {path}")
    return pd.read_csv(path)


def style_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", which="major", direction="out", width=0.7, length=3, pad=2)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", linewidth=0.4, color="0.82")


def add_panel_label(ax, label, dataset):
    ax.text(0.5, 1.015, f"({label}) {dataset}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8)


def save_figure(fig, filename):
    png = os.path.join(FIGURE_DIR, filename + ".png")
    pdf = os.path.join(FIGURE_DIR, filename + ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


def normalize_method_name(value):
    if pd.isna(value):
        return value
    s = str(value).strip()
    key = re.sub(r"[^a-z0-9]+", "", s.lower())
    aliases = {
        "random": "Random",
        "transformergreedy": "Transformer-Greedy",
        "greedy": "Transformer-Greedy",
        "epsilongreedy": "Epsilon-Greedy",
        "egreedy": "Epsilon-Greedy",
        "predictiveucb": "Predictive UCB",
        "ucb": "Predictive UCB",
        "hybridthompsonsampling": "Hybrid Thompson Sampling",
        "thompson": "Hybrid Thompson Sampling",
        "thompsonsampling": "Hybrid Thompson Sampling",
        "adaptivemab": "Adaptive-MAB",
        "fixedbetamab": "Fixed-Beta-MAB",
        "fixedbeta": "Fixed-Beta-MAB",
        "adaptivemabwithoutuncertainty": "Adaptive-MAB-w/o-Uncertainty",
        "adaptivemabwouncertainty": "Adaptive-MAB-w/o-Uncertainty",
        "adaptivemabnouncertainty": "Adaptive-MAB-w/o-Uncertainty",
    }
    return aliases.get(key, s)


def normalize_dataset_name(value):
    if pd.isna(value):
        return value
    s = str(value).strip().lower()
    if "amazon" in s:
        return "Amazon"
    if "kuai" in s:
        return "KuaiRand"
    return str(value).strip()


def normalize_table(df):
    out = df.copy()
    if "Dataset" in out.columns:
        out["Dataset"] = out["Dataset"].map(normalize_dataset_name)
    if "dataset" in out.columns:
        out["dataset"] = out["dataset"].map(normalize_dataset_name)
    if "Method" in out.columns:
        out["Method"] = out["Method"].map(normalize_method_name)
    if "method" in out.columns:
        out["method"] = out["method"].map(normalize_method_name)
    return out


def require_columns(df, columns, context):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing columns {missing}. Available: {list(df.columns)}")


def ordered_subset(df, dataset, methods, context):
    df = normalize_table(df)
    require_columns(df, ["Dataset", "Method"], context)
    sub = df[df["Dataset"] == dataset].copy()
    available = list(sub["Method"].dropna().unique())
    missing = [m for m in methods if m not in available]
    if missing:
        raise ValueError(
            f"{context} / {dataset}: required methods missing: {missing}\n"
            f"Available methods: {available}\n"
            "This script stops instead of silently creating NaN/empty plots."
        )
    sub = sub.set_index("Method").loc[methods].reset_index()
    return sub


def assert_finite(df, columns, context):
    for col in columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals.to_numpy()).all():
            bad = df.loc[vals.isna(), [c for c in ["Dataset", "Method", col] if c in df.columns]]
            raise ValueError(f"{context}: non-finite values in {col}.\n{bad}")


def to_percent(series):
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if len(finite) == 0:
        return s
    # Fractional rates are normally in [0, 1]. Already-percent values are > 1.
    return s * 100.0 if finite.abs().max() <= 1.5 else s


# -----------------------------------------------------------------------------
# Figure 1
# -----------------------------------------------------------------------------
def figure_1():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="0.97"):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.03,rounding_size=0.07",
                           linewidth=0.8, edgecolor="0.25", facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=7.5)

    def arrow(x1, y1, x2, y2, rad=0.0):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                            linewidth=0.8, color="0.25",
                            connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(a)

    box(0.2, 4.2, 1.7, 0.9, "Historical\nInteraction Sequence")
    box(2.3, 4.2, 1.6, 0.9, "Candidate\nGeneration")
    box(4.3, 4.2, 2.0, 0.9, "Transformer +\nMC-Dropout Scoring", "#EAF2F8")
    box(6.7, 4.2, 2.0, 0.9, "Predicted Reward\n+ Uncertainty", "#EAF2F8")
    box(9.1, 4.2, 1.8, 0.9, r"Adaptive $\beta$\nController", "#FDF2E9")
    box(9.1, 2.3, 1.8, 0.9, "Adaptive-MAB\nArm Selection", "#FDEDEC")
    box(6.7, 2.3, 2.0, 0.9, "Selected\nRecommendation")
    box(4.3, 2.3, 2.0, 0.9, "Logged-Target\nFeedback", "#E8F8F5")
    box(2.3, 2.3, 1.6, 0.9, "Bandit State\nUpdate", "#E8F8F5")

    arrow(1.9, 4.65, 2.3, 4.65)
    arrow(3.9, 4.65, 4.3, 4.65)
    arrow(6.3, 4.65, 6.7, 4.65)
    arrow(8.7, 4.65, 9.1, 4.65)
    arrow(10.0, 4.2, 10.0, 3.2)
    arrow(9.1, 2.75, 8.7, 2.75)
    arrow(6.7, 2.75, 6.3, 2.75)
    arrow(4.3, 2.75, 3.9, 2.75)
    arrow(2.3, 2.75, 1.0, 4.15, rad=-0.2)

    ax.text(5.3, 1.15,
            "Reward is observed only when the selected item matches the logged target",
            ha="center", va="center", fontsize=6.8, style="italic", color="0.35")

    save_figure(fig, "figure_1_proposed_framework")


# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------
def figure_2():
    df = normalize_table(load_table("table_3_main_mab.csv"))
    methods = ["Random", "Transformer-Greedy", "Epsilon-Greedy", "Predictive UCB",
               "Hybrid Thompson Sampling", "Adaptive-MAB"]
    labels = ["Random", "Greedy", r"$\epsilon$-Greedy", "UCB", "Thompson", "Adaptive"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.55), constrained_layout=True)

    for i, dataset in enumerate(["Amazon", "KuaiRand"]):
        ax = axes[i]
        sub = ordered_subset(df, dataset, methods, "Figure 2")
        require_columns(sub, ["Reward Mean", "Reward SD"], "Figure 2")
        assert_finite(sub, ["Reward Mean", "Reward SD"], f"Figure 2 / {dataset}")
        ax.errorbar(x, sub["Reward Mean"], yerr=sub["Reward SD"],
                    fmt=MARKERS[dataset], linestyle="none", markersize=5,
                    color=COLORS[dataset], markerfacecolor=COLORS[dataset],
                    markeredgecolor="black", markeredgewidth=0.5,
                    elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", rotation_mode="anchor")
        if i == 0:
            ax.set_ylabel("Mean Observed Reward")
        ymin = np.min(sub["Reward Mean"] - sub["Reward SD"])
        ymax = np.max(sub["Reward Mean"] + sub["Reward SD"])
        margin = max((ymax - ymin) * 0.08, 1e-4)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(97+i), dataset)
        style_axis(ax)

    save_figure(fig, "figure_2_main_mab_reward")


# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------
def figure_3():
    df = normalize_table(load_table("table_3_main_mab.csv"))
    methods = ["Random", "Transformer-Greedy", "Epsilon-Greedy", "Predictive UCB",
               "Hybrid Thompson Sampling", "Adaptive-MAB"]
    labels = ["Random", "Greedy", r"$\epsilon$-Greedy", "UCB", "Thompson", "Adaptive"]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.40), constrained_layout=True)

    for i, dataset in enumerate(["Amazon", "KuaiRand"]):
        sub = ordered_subset(df, dataset, methods, "Figure 3")
        require_columns(sub,["Non-Greedy Selection Rate Mean","Non-Greedy Selection Rate SD",],"Figure 3",)
        assert_finite(sub,["Non-Greedy Selection Rate Mean","Non-Greedy Selection Rate SD",],f"Figure 3 / {dataset}",)
        mean = to_percent(sub["Non-Greedy Selection Rate Mean"])
        sd = to_percent(sub["Non-Greedy Selection Rate SD"])
        offset = (i - 0.5) * width
        ax.bar(x + offset, mean, width=width, yerr=sd, capsize=2,
               color=COLORS[dataset], edgecolor="black", linewidth=0.5,
               hatch=HATCHES[dataset], label=dataset,
               error_kw={"elinewidth": 0.75, "capthick": 0.75}, zorder=3)

    ax.set_ylabel("Non-Greedy Selection Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              handlelength=1.3, handletextpad=0.5, columnspacing=1.1, borderaxespad=0)
    save_figure(fig, "figure_3_exploration_comparison")


# Figure 4
def figure_4():

    sources = {
        "Amazon": os.path.join(
            "outputs",
            "uncertainty",
            "amazon_uncertainty_groups.csv",
        ),
        "KuaiRand": os.path.join(
            "outputs",
            "uncertainty",
            "kuairand_uncertainty_groups.csv",
        ),
    }

    datasets = {}

    for dataset, path in sources.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Figure 4 source not found for {dataset}:\n{path}\n"
                f"Run the uncertainty experiment first."
            )

        df = pd.read_csv(path)

        required_columns = {
            "group",
            "uncertainty_mean",
            "prediction_error_mean",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Figure 4 / {dataset}: missing columns "
                f"{sorted(missing)}\n"
                f"Available columns: {list(df.columns)}"
            )

        df = df.sort_values("group")

        if len(df) != 5:
            raise ValueError(
                f"Figure 4 / {dataset}: expected 5 uncertainty groups, "
                f"found {len(df)}."
            )

        expected_groups = [1, 2, 3, 4, 5]

        actual_groups = df["group"].astype(int).tolist()

        if actual_groups != expected_groups:
            raise ValueError(
                f"Figure 4 / {dataset}: expected groups "
                f"{expected_groups}, found {actual_groups}."
            )

        datasets[dataset] = df

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL_WIDTH, 2.40),
        constrained_layout=True,
    )

    for idx, dataset in enumerate(
        ["Amazon", "KuaiRand"]
    ):

        ax = axes[idx]

        df = datasets[dataset]

        groups = df["group"].to_numpy()

        mae = (
            df["prediction_error_mean"]
            .to_numpy()
        )

        ax.plot(
            groups,
            mae,
            marker=MARKERS[dataset],
            markersize=4.5,
            linewidth=1.2,
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.45,
            zorder=3,
        )

        ax.set_xlabel(
            "Uncertainty Group"
        )

        if idx == 0:
            ax.set_ylabel(
                "MAE"
            )

        ax.set_xticks(
            groups
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=5
            )
        )

        add_panel_label(
            ax,
            chr(97 + idx),
            dataset,
        )

        style_axis(ax)

    save_figure(
        fig,
        "figure_4_uncertainty_prediction_error",
    )


# -----------------------------------------------------------------------------
# Figure 5
# -----------------------------------------------------------------------------
def figure_5():
    df = normalize_table(load_table("table_6_candidate_size.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.40), constrained_layout=True)
    for i, dataset in enumerate(["Amazon", "KuaiRand"]):
        ax = axes[i]
        sub = df[df["Dataset"] == dataset].sort_values("Candidate Size").copy()
        require_columns(sub, ["Candidate Size", "Reward Mean", "Reward SD"], "Figure 5")
        if sub.empty:
            raise ValueError(f"Figure 5: no rows for {dataset}")
        assert_finite(sub, ["Candidate Size", "Reward Mean", "Reward SD"], f"Figure 5 / {dataset}")
        ax.errorbar(sub["Candidate Size"], sub["Reward Mean"], yerr=sub["Reward SD"],
                    marker=MARKERS[dataset], markersize=4.5, linewidth=1.2, capsize=2.5,
                    color=COLORS[dataset], markerfacecolor=COLORS[dataset],
                    markeredgecolor="black", markeredgewidth=0.45,
                    elinewidth=0.8, capthick=0.8, zorder=3)
        ax.set_xlabel("Candidate Set Size")
        if i == 0:
            ax.set_ylabel("Mean Observed Reward")
        ax.set_xticks(sub["Candidate Size"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(97+i), dataset)
        style_axis(ax)
    save_figure(fig, "figure_5_candidate_size_sensitivity")


# -----------------------------------------------------------------------------
# Figure 6
# -----------------------------------------------------------------------------
def figure_6():
    df = normalize_table(load_table("table_7_mc_dropout.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.40), constrained_layout=True)
    for i, dataset in enumerate(["Amazon", "KuaiRand"]):
        ax = axes[i]
        sub = df[df["Dataset"] == dataset].sort_values("MC Samples").copy()
        require_columns(sub, ["MC Samples", "Reward Mean", "Reward SD"], "Figure 6")
        if sub.empty:
            raise ValueError(f"Figure 6: no rows for {dataset}")
        assert_finite(sub, ["MC Samples", "Reward Mean", "Reward SD"], f"Figure 6 / {dataset}")
        ax.errorbar(sub["MC Samples"], sub["Reward Mean"], yerr=sub["Reward SD"],
                    marker=MARKERS[dataset], markersize=4.5, linewidth=1.2, capsize=2.5,
                    color=COLORS[dataset], markerfacecolor=COLORS[dataset],
                    markeredgecolor="black", markeredgewidth=0.45,
                    elinewidth=0.8, capthick=0.8, zorder=3)
        ax.set_xlabel("MC Dropout Samples")
        if i == 0:
            ax.set_ylabel("Mean Observed Reward")
        ax.set_xticks(sub["MC Samples"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(97+i), dataset)
        style_axis(ax)
    save_figure(fig, "figure_6_mc_dropout_sensitivity")


# -----------------------------------------------------------------------------
# Figure 7: Mechanism ablation
# Target selection + non-greedy rate (both are fully defined for every round)
# -----------------------------------------------------------------------------
def figure_7():
    df = normalize_table(
        load_table("table_4_mechanism_ablation.csv")
    )

    methods = [
        "Transformer-Greedy",
        "Fixed-Beta-MAB",
        "Adaptive-MAB",
    ]

    labels = [
        "Greedy",
        r"Fixed-$\beta$",
        "Adaptive",
    ]

    x = np.arange(len(methods))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COL_WIDTH, 4.35),
        constrained_layout=True,
    )

    for i, dataset in enumerate(
        ["Amazon", "KuaiRand"]
    ):
        sub = ordered_subset(
            df,
            dataset,
            methods,
            "Figure 7",
        )

        require_columns(
            sub,
            [
                "Target Selection Rate",
                "Non-Greedy Selection Rate",
            ],
            "Figure 7",
        )

        assert_finite(
            sub,
            [
                "Target Selection Rate",
                "Non-Greedy Selection Rate",
            ],
            f"Figure 7 / {dataset}",
        )

        # -------------------------------------------------
        # Top row: Target Selection Rate
        # -------------------------------------------------
        ax = axes[0, i]

        vals = to_percent(
            sub["Target Selection Rate"]
        )

        ax.bar(
            x,
            vals,
            width=0.55,
            color=COLORS[dataset],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        if i == 0:
            ax.set_ylabel(
                "Target Selection Rate (%)"
            )

        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5)
        )

        add_panel_label(
            ax,
            chr(97 + i),
            dataset,
        )

        style_axis(ax)

        # -------------------------------------------------
        # Bottom row: Non-Greedy Selection Rate
        # -------------------------------------------------
        ax = axes[1, i]

        vals = to_percent(
            sub["Non-Greedy Selection Rate"]
        )

        ax.bar(
            x,
            vals,
            width=0.55,
            color=COLORS[dataset],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        if i == 0:
            ax.set_ylabel(
                "Non-Greedy Selection Rate (%)"
            )

        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5)
        )

        add_panel_label(
            ax,
            chr(99 + i),
            dataset,
        )

        style_axis(ax)

    save_figure(
        fig,
        "figure_7_mechanism_ablation",
    )

# -----------------------------------------------------------------------------
# Figure 8: Uncertainty ablation
# -----------------------------------------------------------------------------
def figure_8():
    df = normalize_table(load_table("table_5_uncertainty_ablation.csv"))
    methods = ["Adaptive-MAB", "Adaptive-MAB-w/o-Uncertainty"]
    labels = ["With\nUncertainty", "Without\nUncertainty"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 4.35), constrained_layout=True)

    for i, dataset in enumerate(["Amazon", "KuaiRand"]):
        sub = ordered_subset(df, dataset, methods, "Figure 8")
        require_columns(sub,["Target Selection Rate","Non-Greedy Selection Rate",], "Figure 8",)
        assert_finite(sub,["Target Selection Rate", "Non-Greedy Selection Rate",],f"Figure 8 / {dataset}",)
        ax = axes[0, i]
        means = to_percent(
            sub["Target Selection Rate"]
        )

        err = (
            to_percent(
                sub["Target Selection Rate SD"]
            )
            if "Target Selection Rate SD" in sub.columns
            else None
        )
        ax.bar(x, means, yerr=err, width=0.52, color=COLORS[dataset], edgecolor="black",
               linewidth=0.5, capsize=2.5,
               error_kw={"elinewidth": 0.8, "capthick": 0.8}, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        if i == 0:
            ax.set_ylabel("Target Selection Rate (%)")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(97+i), dataset)
        style_axis(ax)

        ax = axes[1, i]
        means = to_percent(
            sub["Non-Greedy Selection Rate"]
        )

        err = (
            to_percent(
                sub["Non-Greedy Selection Rate SD"]
            )
            if "Non-Greedy Selection Rate SD" in sub.columns
            else None
        )
        ax.bar(x, means, yerr=err, width=0.52, color=COLORS[dataset], edgecolor="black",
               linewidth=0.5, capsize=2.5,
               error_kw={"elinewidth": 0.8, "capthick": 0.8}, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        if i == 0:
            ax.set_ylabel("Non-Greedy Selection Rate (%)")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(99+i), dataset)
        style_axis(ax)

    save_figure(fig, "figure_8_uncertainty_ablation")


# -----------------------------------------------------------------------------
# Figure 9
# -----------------------------------------------------------------------------
def figure_9():
    candidates = [
        os.path.join("outputs", "uncertainty", "adaptive_exploration_group_summary.csv"),
        os.path.join(TABLE_DIR, "adaptive_exploration_group_summary.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("Adaptive exploration summary not found. Tried:\n" + "\n".join(candidates))

    df = pd.read_csv(path)
    dataset_col = "dataset" if "dataset" in df.columns else "Dataset"
    df[dataset_col] = df[dataset_col].map(normalize_dataset_name)

    aliases = {
        "group": ["uncertainty_group", "Uncertainty Group", "group"],
        "beta": ["mean_beta", "Mean Beta", "beta_mean"],
        "beta_sd": ["std_beta", "Beta SD", "beta_std"],
        "explore": ["exploration_rate", "Exploration Rate", "mean_exploration_rate"],
    }
    def find_col(options, required=True):
        c = next((x for x in options if x in df.columns), None)
        if required and c is None:
            raise ValueError(f"Figure 9: none of {options} found. Available: {list(df.columns)}")
        return c

    group_col = find_col(aliases["group"])
    beta_col = find_col(aliases["beta"])
    beta_sd_col = find_col(aliases["beta_sd"], required=False)
    explore_col = find_col(aliases["explore"])

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 4.35), constrained_layout=True)
    for i, dataset in enumerate(["Amazon", "KuaiRand"]):
        sub = df[df[dataset_col] == dataset].sort_values(group_col).copy()
        if sub.empty:
            raise ValueError(f"Figure 9: no rows for {dataset}")
        groups = pd.to_numeric(sub[group_col], errors="raise")

        ax = axes[0, i]
        beta = pd.to_numeric(sub[beta_col], errors="raise")
        beta_sd = pd.to_numeric(sub[beta_sd_col], errors="raise") if beta_sd_col else None
        ax.errorbar(groups, beta, yerr=beta_sd, marker=MARKERS[dataset], markersize=4.5,
                    linewidth=1.2, capsize=2, color=COLORS[dataset],
                    markerfacecolor=COLORS[dataset], markeredgecolor="black",
                    markeredgewidth=0.45, elinewidth=0.75, capthick=0.75, zorder=3)
        ax.set_xticks(groups)
        if i == 0:
            ax.set_ylabel(r"Adaptive $\beta$")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(97+i), dataset)
        style_axis(ax)

        ax = axes[1, i]
        explore = to_percent(sub[explore_col])
        ax.plot(groups, explore, marker=MARKERS[dataset], markersize=4.5, linewidth=1.2,
                color=COLORS[dataset], markerfacecolor=COLORS[dataset],
                markeredgecolor="black", markeredgewidth=0.45, zorder=3)
        ax.set_xticks(groups)
        ax.set_xlabel("Uncertainty Group")
        if i == 0:
            ax.set_ylabel("Non-Greedy Selection Rate (%)")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(99+i), dataset)
        style_axis(ax)

    save_figure(fig, "figure_9_adaptive_exploration")

def figure_10():
    sweep = pd.read_csv(FIXED_BETA_SWEEP_PATH)
    mechanism = pd.read_csv(MECHANISM_TABLE_PATH)

    required_sweep = [
        "dataset",
        "beta",
        "target_selection_rate_mean",
        "target_selection_rate_std",
        "non_greedy_rate_mean",
        "non_greedy_rate_std",
    ]

    missing = [
        c for c in required_sweep
        if c not in sweep.columns
    ]

    if missing:
        raise ValueError(
            "Figure 10 fixed-beta sweep is missing columns: "
            f"{missing}. "
            f"Available: {list(sweep.columns)}"
        )

    required_mechanism = [
        "Dataset",
        "Method",
        "Target Selection Rate",
        "Non-Greedy Selection Rate",
    ]

    missing = [
        c for c in required_mechanism
        if c not in mechanism.columns
    ]

    if missing:
        raise ValueError(
            "Figure 10 mechanism table is missing columns: "
            f"{missing}. "
            f"Available: {list(mechanism.columns)}"
        )

    datasets = [
        ("Amazon", COLORS["Amazon"], MARKERS["Amazon"]),
        ("KuaiRand", COLORS["KuaiRand"], MARKERS["KuaiRand"]),
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.16, 4.7),
    )

    # ---------------------------------------------------------
    # Top row: target-selection rate
    # ---------------------------------------------------------

    for col, (dataset, color, marker) in enumerate(datasets):
        ax = axes[0, col]

        sub = (
            sweep[
                sweep["dataset"] == dataset
            ]
            .sort_values("beta")
        )

        if sub.empty:
            raise ValueError(
                f"Figure 10: no fixed-beta results for {dataset}"
            )

        adaptive = mechanism[
            (mechanism["Dataset"] == dataset)
            &
            (mechanism["Method"] == "Adaptive-MAB")
        ]

        if len(adaptive) != 1:
            raise ValueError(
                f"Figure 10: expected exactly one "
                f"Adaptive-MAB row for {dataset}, "
                f"found {len(adaptive)}"
            )

        x = sub["beta"].to_numpy(dtype=float)

        y = (
            sub["target_selection_rate_mean"]
            .to_numpy(dtype=float)
            * 100.0
        )

        yerr = (
            sub["target_selection_rate_std"]
            .to_numpy(dtype=float)
            * 100.0
        )

        adaptive_y = (
            float(
                adaptive[
                    "Target Selection Rate"
                ].iloc[0]
            )
            * 100.0
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=marker,
            markersize=3.8,
            linewidth=1.2,
            capsize=2.2,
            color=color,
            label=r"Fixed-$\beta$",
        )

        ax.axhline(
            adaptive_y,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Adaptive-MAB",
        )

        ax.set_title(
            f"({chr(97 + col)}) {dataset}",
            pad=4,
        )

        ax.set_ylabel(
            "Target Selection Rate (%)"
        )

        ax.set_xlabel(
            r"Fixed Exploration Coefficient $\beta$"
        )

        ax.set_xticks(
            [
                0.0,
                0.25,
                0.50,
                0.75,
                1.0,
            ]
        )

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.5,
            alpha=0.45,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == 0:
            ax.legend(
                frameon=False,
                fontsize=7,
                loc="best",
            )

    # ---------------------------------------------------------
    # Bottom row: non-greedy rate
    # ---------------------------------------------------------

    for col, (dataset, color, marker) in enumerate(datasets):
        ax = axes[1, col]

        sub = (
            sweep[
                sweep["dataset"] == dataset
            ]
            .sort_values("beta")
        )

        adaptive = mechanism[
            (mechanism["Dataset"] == dataset)
            &
            (mechanism["Method"] == "Adaptive-MAB")
        ]

        x = sub["beta"].to_numpy(dtype=float)

        y = (
            sub["non_greedy_rate_mean"]
            .to_numpy(dtype=float)
            * 100.0
        )

        yerr = (
            sub["non_greedy_rate_std"]
            .to_numpy(dtype=float)
            * 100.0
        )

        adaptive_y = (
            float(
                adaptive[
                    "Non-Greedy Selection Rate"
                ].iloc[0]
            )
            * 100.0
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=marker,
            markersize=3.8,
            linewidth=1.2,
            capsize=2.2,
            color=color,
            label=r"Fixed-$\beta$",
        )

        ax.axhline(
            adaptive_y,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Adaptive-MAB",
        )

        ax.set_title(
            f"({chr(99 + col)}) {dataset}",
            pad=4,
        )

        ax.set_ylabel(
            "Non-Greedy Selection Rate (%)"
        )

        ax.set_xlabel(
            r"Fixed Exploration Coefficient $\beta$"
        )

        ax.set_xticks(
            [
                0.0,
                0.25,
                0.50,
                0.75,
                1.0,
            ]
        )

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.5,
            alpha=0.45,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.12,
        top=0.94,
        wspace=0.28,
        hspace=0.42,
    )

    save_figure(
        fig,
        "figure_10_fixed_beta_sensitivity",
    )


def main():
    print("=" * 80)
    print("BUILDING VALIDATED IEEE-STYLE PAPER FIGURES")
    print("=" * 80)

    builders = [
        figure_1,
        figure_2,
        figure_3,
        figure_4,
        figure_5,
        figure_6,
        figure_7,
        figure_8,
        figure_9,
        figure_10,
    ]

    total = len(builders)

    for i, fn in enumerate(builders, start=1):
        print(f"\n[{i}/{total}] {fn.__name__}")
        fn()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()