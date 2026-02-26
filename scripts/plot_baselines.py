"""Generate box plots from baseline benchmark results.

Usage:
    python scripts/plot_baselines.py
    python scripts/plot_baselines.py --input results/baselines_per_episode.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


# Strategy display order (worst → best expected)
STRATEGY_ORDER = ["MinimumOnly", "Snowball", "Avalanche", "NormalizedAvalanche"]

# Color palette — muted, professional
COLORS = {
    "MinimumOnly": "#e74c3c",
    "Snowball": "#3498db",
    "Avalanche": "#9b59b6",
    "NormalizedAvalanche": "#2ecc71",
}


def make_boxplots(df: pd.DataFrame, output_path: str = "results/baselines_boxplots.png") -> None:
    """Create a 4-panel box plot comparing baselines.

    Panels: Total Interest, Months to Payoff, Avg Utilization, Credit Proxy Score
    """
    # Filter to strategies present in the data, in display order
    strategies = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]
    colors = [COLORS.get(s, "#333333") for s in strategies]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline Strategy Comparison", fontsize=16, fontweight="bold", y=0.98)

    metrics = [
        ("total_interest", "Total Interest Paid ($)", axes[0, 0]),
        ("months", "Months to Payoff", axes[0, 1]),
        ("avg_utilization", "Average Utilization", axes[1, 0]),
        ("credit_proxy_score", "Credit Proxy Score", axes[1, 1]),
    ]

    for col, title, ax in metrics:
        data = [df[df["strategy"] == s][col].values for s in strategies]

        bp = ax.boxplot(
            data,
            tick_labels=strategies,
            patch_artist=True,
            notch=True,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Box plots saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate baseline box plots")
    parser.add_argument(
        "--input",
        type=str,
        default="results/baselines_per_episode.csv",
        help="Path to per-episode CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baselines_boxplots.png",
        help="Path to save box plot image",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run run_baselines.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    make_boxplots(df, args.output)


if __name__ == "__main__":
    main()
