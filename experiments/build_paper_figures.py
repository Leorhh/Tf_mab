import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

TABLE_DIR = "outputs/paper_tables"
FIGURE_DIR = "outputs/paper_figures_ieee"
os.makedirs(FIGURE_DIR, exist_ok=True)

SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.16
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
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
HATCHES = {
    "Amazon": "",
    "KuaiRand": "//",
}
MARKERS = {
    "Amazon": "o",
    "KuaiRand": "s",
}


def load_table(filename):
    path = os.path.join(TABLE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Table not found: {path}")
    return pd.read_csv(path)


def style_axis(ax, grid=True):
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
            linewidth=0.45,
            alpha=0.45,
        )


def save_figure(fig, filename):
    png_path = os.path.join(FIGURE_DIR, filename + ".png")
    pdf_path = os.path.join(FIGURE_DIR, filename + ".pdf")
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print(f"[OK] {png_path}")
    print(f"[OK] {pdf_path}")


def figure_2():

    df = load_table("table_3_main_mab.csv")

    methods = [
        "Random",
        "Transformer-Greedy",
        "Epsilon-Greedy",
        "Predictive UCB",
        "Hybrid Thompson Sampling",
        "Adaptive-MAB",
    ]

    short_names = [
        "Random",
        "Greedy",
        r"$\epsilon$-Greedy",
        "UCB",
        "Thompson",
        "Adaptive",
    ]

    datasets = ["Amazon", "KuaiRand"]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.55),
        constrained_layout=True,
    )

    for idx, dataset in enumerate(datasets):

        ax = axes[idx]

        sub = (
            df[df["Dataset"] == dataset]
            .set_index("Method")
            .reindex(methods)
            .reset_index()
        )

        ax.errorbar(
            x,
            sub["Reward Mean"],
            yerr=sub["Reward SD"],
            fmt=MARKERS[dataset],
            linestyle="none",
            markersize=5,
            capsize=2.5,
            color=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.5,
            elinewidth=0.9,
            capthick=0.9,
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            short_names,
            rotation=20,
            ha="right",
            rotation_mode="anchor",
        )

        if idx == 0:
            ax.set_ylabel("Mean Reward")

        ax.text(
            0.5,
            1.03,
            f"({chr(97 + idx)}) {dataset}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )

        ymin = np.min(
            sub["Reward Mean"] - sub["Reward SD"]
        )

        ymax = np.max(
            sub["Reward Mean"] + sub["Reward SD"]
        )

        margin = 0.08 * (ymax - ymin)

        ax.set_ylim(
            ymin - margin,
            ymax + margin,
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5)
        )

        style_axis(ax)

    save_figure(
        fig,
        "figure_2_main_mab_reward"
    )


def figure_3():
    df = load_table("table_3_main_mab.csv")
    methods = [
        "Random",
        "Transformer-Greedy",
        "Epsilon-Greedy",
        "Predictive UCB",
        "Hybrid Thompson Sampling",
        "Adaptive-MAB",
    ]
    short_names = [
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
    fig, ax = plt.subplots(
        figsize=(SINGLE_COL_WIDTH, 2.45),
        constrained_layout=True,
    )
    for i, dataset in enumerate(datasets):
        sub = (
            df[df["Dataset"] == dataset]
            .set_index("Method")
            .reindex(methods)
            .reset_index()
        )
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            sub["Exploration Rate Mean"] * 100,
            width=width,
            yerr=sub["Exploration Rate SD"] * 100,
            capsize=1.8,
            color=COLORS[dataset],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[dataset],
            label=dataset,
            error_kw={
                "elinewidth": 0.7,
                "capthick": 0.7,
            },
        )
    ax.set_ylabel("Exploration Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        short_names,
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    save_figure(fig, "figure_3_exploration_comparison")


def figure_4():
    datasets = {
        "Amazon": {
            "uncertainty": [
                0.004382,
                0.006837,
                0.008489,
                0.010321,
                0.015810,
            ],
            "mae": [
                0.264074,
                0.228003,
                0.220972,
                0.219196,
                0.242769,
            ],
        },
        "KuaiRand": {
            "uncertainty": [
                0.003239,
                0.004751,
                0.005909,
                0.007192,
                0.009616,
            ],
            "mae": [
                0.076897,
                0.102882,
                0.117908,
                0.134600,
                0.150060,
            ],
        },
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(SINGLE_COL_WIDTH, 2.15),
        constrained_layout=True,
    )
    groups = np.arange(1, 6)
    for idx, dataset in enumerate(["Amazon", "KuaiRand"]):
        ax = axes[idx]
        values = datasets[dataset]
        ax.plot(
            groups,
            values["mae"],
            marker=MARKERS[dataset],
            markersize=3.8,
            linewidth=1.15,
            color=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.35,
        )
        ax.set_xlabel("Uncertainty Group")
        if idx == 0:
            ax.set_ylabel("MAE")
        ax.set_xticks(groups)
        ax.text(
            0.5,
            1.03,
            f"({chr(97 + idx)}) {dataset}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(ax)
    save_figure(fig, "figure_4_uncertainty_prediction_error")


def figure_5():
    df = load_table("table_6_candidate_size.csv")
    datasets = ["Amazon", "KuaiRand"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(SINGLE_COL_WIDTH, 2.15),
        constrained_layout=True,
    )
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["Dataset"] == dataset].sort_values("Candidate Size")
        ax.errorbar(
            sub["Candidate Size"],
            sub["Reward Mean"],
            yerr=sub["Reward SD"],
            marker=MARKERS[dataset],
            markersize=3.8,
            linewidth=1.15,
            capsize=2,
            color=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.35,
            elinewidth=0.75,
            capthick=0.75,
        )
        ax.set_xlabel("Candidate Set Size")
        if idx == 0:
            ax.set_ylabel("Mean Reward")
        ax.set_xticks(sub["Candidate Size"])
        ax.text(
            0.5,
            1.03,
            f"({chr(97 + idx)}) {dataset}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(ax)
    save_figure(fig, "figure_5_candidate_size_sensitivity")


def figure_6():
    df = load_table("table_7_mc_dropout.csv")
    datasets = ["Amazon", "KuaiRand"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(SINGLE_COL_WIDTH, 2.15),
        constrained_layout=True,
    )
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = df[df["Dataset"] == dataset].sort_values("MC Samples")
        ax.errorbar(
            sub["MC Samples"],
            sub["Reward Mean"],
            yerr=sub["Reward SD"],
            marker=MARKERS[dataset],
            markersize=3.8,
            linewidth=1.15,
            capsize=2,
            color=COLORS[dataset],
            markeredgecolor="black",
            markeredgewidth=0.35,
            elinewidth=0.75,
            capthick=0.75,
        )
        ax.set_xlabel("MC Dropout Samples")
        if idx == 0:
            ax.set_ylabel("Mean Reward")
        ax.set_xticks(sub["MC Samples"])
        ax.text(
            0.5,
            1.03,
            f"({chr(97 + idx)}) {dataset}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        style_axis(ax)
    save_figure(fig, "figure_6_mc_dropout_sensitivity")


def main():
    print("=" * 80)
    print("BUILDING IEEE‑STYLE PAPER FIGURES")
    print("=" * 80)
    print("\n[1/5] Figure 2")
    figure_2()
    print("\n[2/5] Figure 3")
    figure_3()
    print("\n[3/5] Figure 4")
    figure_4()
    print("\n[4/5] Figure 5")
    figure_5()
    print("\n[5/5] Figure 6")
    figure_6()
    print("\n" + "=" * 80)
    print("IEEE FIGURE GENERATION FINISHED")
    print("=" * 80)
    files = sorted(f for f in os.listdir(FIGURE_DIR) if f.endswith((".png", ".pdf")))
    print(f"\nGenerated {len(files)} files:")
    for filename in files:
        print(f"  {os.path.join(FIGURE_DIR, filename)}")


if __name__ == "__main__":
    main()
