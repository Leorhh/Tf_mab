import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

TABLE_DIR = "outputs/paper_tables"
FIGURE_DIR = "outputs/paper_figures_ieee"
os.makedirs(FIGURE_DIR, exist_ok=True)

SINGLE_COL_WIDTH = 3.50
DOUBLE_COL_WIDTH = 7.16

rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],
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

AMAZON_COLOR = "#0072B2"
KUAIRAND_COLOR = "#D55E00"
COLORS = {
    "Amazon": AMAZON_COLOR,
    "KuaiRand": KUAIRAND_COLOR,
}
MARKERS = {
    "Amazon": "o",
    "KuaiRand": "s",
}
HATCHES = {
    "Amazon": "",
    "KuaiRand": "//",
}


def load_table(filename):
    path = os.path.join(TABLE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required table not found:\n{path}"
        )
    return pd.read_csv(path)


def get_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            "None of the following columns were found:\n"
            + "\n".join(candidates)
            + "\n\nAvailable columns:\n"
            + "\n".join(df.columns)
        )
    return None


def style_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=0.7,
        length=3,
        pad=2,
    )
    if grid:
        ax.set_axisbelow(True)
        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.4,
            color="0.82",
        )


def add_panel_label(ax, label, dataset):
    ax.text(
        0.5,
        1.015,
        f"({label}) {dataset}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )


def save_figure(fig, filename):
    png_path = os.path.join(FIGURE_DIR, filename + ".png")
    pdf_path = os.path.join(FIGURE_DIR, filename + ".pdf")
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)
    print(f"[OK] {png_path}")
    print(f"[OK] {pdf_path}")


def figure_1():
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def add_box(x, y, w, h, text, facecolor="0.96", edgecolor="0.25", fontsize=8):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=0.8,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    def add_arrow(x1, y1, x2, y2, connectionstyle="arc3"):
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.8,
            color="0.25",
            connectionstyle=connectionstyle,
        )
        ax.add_patch(arrow)

    add_box(0.25, 4.4, 1.45, 0.85, "User–Item\nInteraction Sequence")
    add_box(2.05, 4.4, 1.45, 0.85, "Transformer\nReward Model", facecolor="#EAF2F8")
    add_box(3.85, 4.4, 1.35, 0.85, "Candidate\nGenerator")
    add_box(5.55, 4.4, 1.45, 0.85, "MC Dropout\nScoring", facecolor="#EAF2F8")
    add_box(7.35, 4.4, 1.55, 0.85, "Predicted Reward\n+ Uncertainty", facecolor="#EAF2F8")
    add_box(7.35, 2.6, 1.55, 0.85, "Adaptive\nExploration Controller", facecolor="#FDF2E9")
    add_box(5.55, 2.6, 1.45, 0.85, r"Adaptive $\beta$", facecolor="#FDF2E9")
    add_box(3.75, 2.6, 1.55, 0.85, "Adaptive-MAB\nArm Selection", facecolor="#FDEDEC")
    add_box(1.95, 2.6, 1.45, 0.85, "Selected\nRecommendation")
    add_box(0.25, 2.6, 1.35, 0.85, "Observed\nReward", facecolor="#E8F8F5")
    add_box(3.75, 0.75, 1.55, 0.85, "Bandit State\nUpdate", facecolor="#E8F8F5")

    add_arrow(1.70, 4.825, 2.05, 4.825)
    add_arrow(3.50, 4.825, 3.85, 4.825)
    add_arrow(5.20, 4.825, 5.55, 4.825)
    add_arrow(7.00, 4.825, 7.35, 4.825)
    add_arrow(8.125, 4.40, 8.125, 3.45)
    add_arrow(7.35, 3.025, 7.00, 3.025)
    add_arrow(5.55, 3.025, 5.30, 3.025)
    add_arrow(3.75, 3.025, 3.40, 3.025)
    add_arrow(1.95, 3.025, 1.60, 3.025)
    add_arrow(0.925, 2.60, 3.75, 1.18, connectionstyle="arc3,rad=0.18")
    add_arrow(4.525, 1.60, 4.525, 2.60)

    ax.text(6.28, 3.75, "uncertainty‑aware", ha="center", va="center", fontsize=7, style="italic", color="0.35")
    ax.text(2.45, 1.15, "feedback", ha="center", va="center", fontsize=7, style="italic", color="0.35")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.03)
    save_figure(fig, "figure_1_proposed_framework")


def figure_2():
    df = load_table("table_3_main_mab.csv")
    methods = [
        "Random",
        "Transformer‑Greedy",
        "Epsilon‑Greedy",
        "Predictive UCB",
        "Hybrid Thompson Sampling",
        "Adaptive‑MAB",
    ]
    labels = [
        "Random",
        "Greedy",
        r"$\epsilon$-Greedy",
        "UCB",
        "Thompson",
        "Adaptive",
    ]
    datasets = ["Amazon", "KuaiRand"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.55), constrained_layout=True)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["Dataset"] == dataset].set_index("Method").reindex(methods).reset_index()
        ax.errorbar(
            x,
            sub["Reward Mean"],
            yerr=sub["Reward SD"],
            fmt=MARKERS[dataset],
            linestyle="none",
            markersize=5,
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.5,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", rotation_mode="anchor")
        if idx == 0:
            ax.set_ylabel("Mean Reward")
        ymin = np.min(sub["Reward Mean"] - sub["Reward SD"])
        ymax = np.max(sub["Reward Mean"] + sub["Reward SD"])
        data_range = ymax - ymin
        margin = data_range * 0.08 if data_range > 0 else abs(ymax) * 0.05
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(97 + idx), dataset)
        style_axis(ax)
    save_figure(fig, "figure_2_main_mab_reward")


def figure_3():
    df = load_table("table_3_main_mab.csv")
    methods = [
        "Random",
        "Transformer‑Greedy",
        "Epsilon‑Greedy",
        "Predictive UCB",
        "Hybrid Thompson Sampling",
        "Adaptive‑MAB",
    ]
    labels = [
        "Random",
        "Greedy",
        r"$\epsilon$-Greedy",
        "UCB",
        "Thompson",
        "Adaptive",
    ]
    datasets = ["Amazon", "KuaiRand"]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 2.40), constrained_layout=True)
    for idx, dataset in enumerate(datasets):
        sub = df[df["Dataset"] == dataset].set_index("Method").reindex(methods).reset_index()
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            sub["Exploration Rate Mean"] * 100,
            width=width,
            yerr=sub["Exploration Rate SD"] * 100,
            capsize=2,
            color=COLORS[dataset],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[dataset],
            label=dataset,
            error_kw={"elinewidth": 0.75, "capthick": 0.75},
            zorder=3,
        )
    ax.set_ylabel("Exploration Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", rotation_mode="anchor")
    ax.set_ylim(-2, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=1.1,
        borderaxespad=0,
    )
    save_figure(fig, "figure_3_exploration_comparison")


def figure_4():
    datasets = {
        "Amazon": {
            "uncertainty": [0.004382, 0.006837, 0.008489, 0.010321, 0.015810],
            "mae": [0.264074, 0.228003, 0.220972, 0.219196, 0.242769],
        },
        "KuaiRand": {
            "uncertainty": [0.003239, 0.004751, 0.005909, 0.007192, 0.009616],
            "mae": [0.076897, 0.102882, 0.117908, 0.134600, 0.150060],
        },
    }
    groups = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.40), constrained_layout=True)
    for idx, dataset in enumerate(["Amazon", "KuaiRand"]):
        ax = axes[idx]
        values = datasets[dataset]
        ax.plot(
            groups,
            values["mae"],
            marker=MARKERS[dataset],
            markersize=4.5,
            linewidth=1.2,
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.45,
            zorder=3,
        )
        ax.set_xlabel("Uncertainty Group")
        if idx == 0:
            ax.set_ylabel("MAE")
        ax.set_xticks(groups)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(97 + idx), dataset)
        style_axis(ax)
    save_figure(fig, "figure_4_uncertainty_prediction_error")


def figure_5():
    df = load_table("table_6_candidate_size.csv")
    datasets = ["Amazon", "KuaiRand"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.40), constrained_layout=True)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["Dataset"] == dataset].sort_values("Candidate Size")
        ax.errorbar(
            sub["Candidate Size"],
            sub["Reward Mean"],
            yerr=sub["Reward SD"],
            marker=MARKERS[dataset],
            markersize=4.5,
            linewidth=1.2,
            capsize=2.5,
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.45,
            elinewidth=0.8,
            capthick=0.8,
            zorder=3,
        )
        ax.set_xlabel("Candidate Set Size")
        if idx == 0:
            ax.set_ylabel("Mean Reward")
        ax.set_xticks(sub["Candidate Size"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(97 + idx), dataset)
        style_axis(ax)
    save_figure(fig, "figure_5_candidate_size_sensitivity")


def figure_6():
    df = load_table("table_7_mc_dropout.csv")
    datasets = ["Amazon", "KuaiRand"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.40), constrained_layout=True)
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["Dataset"] == dataset].sort_values("MC Samples")
        ax.errorbar(
            sub["MC Samples"],
            sub["Reward Mean"],
            yerr=sub["Reward SD"],
            marker=MARKERS[dataset],
            markersize=4.5,
            linewidth=1.2,
            capsize=2.5,
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.45,
            elinewidth=0.8,
            capthick=0.8,
            zorder=3,
        )
        ax.set_xlabel("MC Dropout Samples")
        if idx == 0:
            ax.set_ylabel("Mean Reward")
        ax.set_xticks(sub["MC Samples"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        add_panel_label(ax, chr(97 + idx), dataset)
        style_axis(ax)
    save_figure(fig, "figure_6_mc_dropout_sensitivity")


def figure_7():
    df = load_table("table_4_mechanism_ablation.csv")
    reward_mean_col = get_column(df, ["Reward Mean", "Mean Reward"])
    reward_sd_col = get_column(df, ["Reward SD", "Reward Std", "Reward Standard Deviation"], required=False)
    exploration_col = get_column(df, ["Exploration Rate", "Exploration Rate Mean", "Mean Exploration Rate"])
    methods = ["Transformer‑Greedy", "Fixed‑Beta‑MAB", "Adaptive‑MAB"]
    labels = ["Greedy", r"Fixed‑$\beta$", "Adaptive"]
    datasets = ["Amazon", "KuaiRand"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 4.4), constrained_layout=True)
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df["Dataset"] == dataset].set_index("Method").reindex(methods).reset_index()
        yerr = sub[reward_sd_col] if reward_sd_col is not None else None
        ax.errorbar(
            x,
            sub[reward_mean_col],
            yerr=yerr,
            fmt=MARKERS[dataset],
            linestyle="none",
            markersize=5,
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.45,
            elinewidth=0.85,
            capsize=2.5,
            capthick=0.85,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if idx == 0:
            ax.set_ylabel("Mean Reward")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(ax)
        add_panel_label(ax, chr(97 + idx), dataset)
    for idx, dataset in enumerate(datasets):
        ax = axes[1, idx]
        sub = df[df["Dataset"] == dataset].set_index("Method").reindex(methods).reset_index()
        values = sub[exploration_col] * 100
        ax.bar(
            x,
            values,
            width=0.55,
            color=COLORS[dataset],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if idx == 0:
            ax.set_ylabel("Exploration Rate (%)")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(ax)
        add_panel_label(ax, chr(99 + idx), dataset)
    save_figure(fig, "figure_7_mechanism_ablation")


def figure_8():
    df = load_table("table_5_uncertainty_ablation.csv")
    reward_mean_col = get_column(df, ["Reward Mean", "Mean Reward"])
    reward_sd_col = get_column(df, ["Reward SD", "Reward Std", "Reward Standard Deviation"], required=False)
    exploration_mean_col = get_column(df, ["Exploration Rate", "Exploration Rate Mean", "Mean Exploration Rate"])
    exploration_sd_col = get_column(df, ["Exploration Rate SD", "Exploration Rate Std"], required=False)
    methods = ["Adaptive‑MAB", "Adaptive‑MAB‑w/o‑Uncertainty"]
    labels = ["With\nUncertainty", "Without\nUncertainty"]
    datasets = ["Amazon", "KuaiRand"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 4.4), constrained_layout=True)
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df["Dataset"] == dataset].set_index("Method").reindex(methods).reset_index()
        reward_error = sub[reward_sd_col] if reward_sd_col is not None else None
        ax.errorbar(
            x,
            sub[reward_mean_col],
            yerr=reward_error,
            fmt=MARKERS[dataset],
            linestyle="none",
            markersize=5,
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.45,
            elinewidth=0.85,
            capsize=2.5,
            capthick=0.85,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if idx == 0:
            ax.set_ylabel("Mean Reward")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(ax)
        add_panel_label(ax, chr(97 + idx), dataset)
    for idx, dataset in enumerate(datasets):
        ax = axes[1, idx]
        sub = df[df["Dataset"] == dataset].set_index("Method").reindex(methods).reset_index()
        exploration_mean = sub[exploration_mean_col] * 100
        exploration_error = sub[exploration_sd_col] * 100 if exploration_sd_col is not None else None
        ax.bar(
            x,
            exploration_mean,
            yerr=exploration_error,
            width=0.52,
            color=COLORS[dataset],
            edgecolor="black",
            linewidth=0.5,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if idx == 0:
            ax.set_ylabel("Exploration Rate (%)")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(ax)
        add_panel_label(ax, chr(99 + idx), dataset)
    save_figure(fig, "figure_8_uncertainty_ablation")


def figure_9():
    possible_paths = [
        os.path.join("outputs", "uncertainty", "adaptive_exploration_group_summary.csv"),
        os.path.join(TABLE_DIR, "adaptive_exploration_group_summary.csv"),
    ]
    path = None
    for candidate in possible_paths:
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        raise FileNotFoundError(
            "Adaptive exploration summary not found.\n\n"
            "Expected one of:\n"
            + "\n".join(possible_paths)
        )
    df = pd.read_csv(path)
    dataset_col = get_column(df, ["dataset", "Dataset"])
    group_col = get_column(df, ["uncertainty_group", "Uncertainty Group", "group"])
    beta_mean_col = get_column(df, ["mean_beta", "Mean Beta", "beta_mean"])
    beta_sd_col = get_column(df, ["std_beta", "Beta SD", "beta_std"], required=False)
    exploration_col = get_column(df, ["exploration_rate", "Exploration Rate", "mean_exploration_rate"])
    datasets = ["Amazon", "KuaiRand"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 4.4), constrained_layout=True)
    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        sub = df[df[dataset_col] == dataset].sort_values(group_col)
        groups = sub[group_col].to_numpy()
        beta_error = sub[beta_sd_col] if beta_sd_col is not None else None
        ax.errorbar(
            groups,
            sub[beta_mean_col],
            yerr=beta_error,
            marker=MARKERS[dataset],
            markersize=4.5,
            linewidth=1.2,
            capsize=2,
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.45,
            elinewidth=0.75,
            capthick=0.75,
            zorder=3,
        )
        ax.set_xticks(groups)
        if idx == 0:
            ax.set_ylabel(r"Adaptive $\beta$")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(ax)
        add_panel_label(ax, chr(97 + idx), dataset)
    for idx, dataset in enumerate(datasets):
        ax = axes[1, idx]
        sub = df[df[dataset_col] == dataset].sort_values(group_col)
        groups = sub[group_col].to_numpy()
        exploration_percent = sub[exploration_col] * 100
        ax.plot(
            groups,
            exploration_percent,
            marker=MARKERS[dataset],
            markersize=4.5,
            linewidth=1.2,
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.45,
            zorder=3,
        )
        ax.set_xticks(groups)
        ax.set_xlabel("Uncertainty Group")
        if idx == 0:
            ax.set_ylabel("Exploration Rate (%)")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(ax)
        add_panel_label(ax, chr(99 + idx), dataset)
    save_figure(fig, "figure_9_adaptive_exploration")


def main():
    print("=" * 80)
    print("BUILDING FINAL IEEE‑STYLE PAPER FIGURES")
    print("=" * 80)
    print("\n[1/9] Figure 1: Proposed framework")
    figure_1()
    print("\n[2/9] Figure 2: Main MAB reward")
    figure_2()
    print("\n[3/9] Figure 3: Exploration comparison")
    figure_3()
    print("\n[4/9] Figure 4: Uncertainty vs prediction error")
    figure_4()
    print("\n[5/9] Figure 5: Candidate‑size sensitivity")
    figure_5()
    print("\n[6/9] Figure 6: MC‑dropout sensitivity")
    figure_6()
    print("\n[7/9] Figure 7: Mechanism ablation")
    figure_7()
    print("\n[8/9] Figure 8: Uncertainty ablation")
    figure_8()
    print("\n[9/9] Figure 9: Adaptive exploration mechanism")
    figure_9()
    print("\n" + "=" * 80)
    print("IEEE FIGURE GENERATION FINISHED")
    print("=" * 80)
    files = sorted([f for f in os.listdir(FIGURE_DIR) if f.endswith((".png", ".pdf"))])
    print(f"\nGenerated {len(files)} files:")
    for filename in files:
        print("  " + os.path.join(FIGURE_DIR, filename))


if __name__ == "__main__":
    main()
