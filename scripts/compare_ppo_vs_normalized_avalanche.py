"""Compare PPO vs Normalized Avalanche across in-distribution and held-out test sets.

Reads summary CSVs, outputs side-by-side comparison and saves report + optional figure.

Usage:
    python scripts/compare_ppo_vs_normalized_avalanche.py
    python scripts/compare_ppo_vs_normalized_avalanche.py --results-dir results --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def main():
    parser = argparse.ArgumentParser(description="Compare PPO vs Normalized Avalanche")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating comparison figure")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / args.results_dir

    # Load in-distribution
    in_dist_path = results_dir / "ppo_vs_baselines_summary.csv"
    if not in_dist_path.exists():
        print(f"Missing {in_dist_path}. Run evaluate_ppo_vs_baselines.py first.")
        return
    in_dist = pd.read_csv(in_dist_path).set_index("strategy")

    # Load held-out summaries
    test_sets = [("In-distribution", in_dist)]
    for f in sorted(results_dir.glob("held_out_summary_*.csv")):
        preset = f.stem.replace("held_out_summary_", "")
        test_sets.append((preset, pd.read_csv(f).set_index("strategy")))

    # PPO vs Normalized Avalanche (baseline strategy name in summary CSVs)
    BASELINE = "NormalizedAvalanche"
    metrics = [
        ("interest_mean", "Total interest ($)", "lower"),
        ("months_mean", "Months to payoff", "lower"),
        ("util_mean", "Avg utilization", "lower"),
        ("paid_off_pct", "Paid off (%)", "higher"),
        ("credit_score_mean", "Credit score", "higher"),
    ]

    rows = []
    for test_name, df in test_sets:
        if "PPO" not in df.index or BASELINE not in df.index:
            continue
        ppo = df.loc["PPO"]
        bl = df.loc[BASELINE]
        for col, label, _ in metrics:
            if col not in ppo or col not in bl:
                continue
            diff = ppo[col] - bl[col]
            # Lower is better: interest, months, util. Higher is better: paid_off_pct, credit_score
            if col in ("interest_mean", "months_mean", "util_mean"):
                winner = "PPO" if diff < 0 else "NormalizedAvalanche"
            else:
                winner = "PPO" if diff > 0 else "NormalizedAvalanche"
            rows.append({
                "test_set": test_name,
                "metric": label,
                "PPO": round(ppo[col], 4),
                "NormalizedAvalanche": round(bl[col], 4),
                "PPO_minus_NormalizedAvalanche": round(diff, 4),
                "better": winner,
            })

    comp_df = pd.DataFrame(rows)
    output_dir = results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ppo_vs_normalized_avalanche_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # Print report
    print("\n" + "=" * 90)
    print("  PPO vs Normalized Avalanche comparison")
    print("=" * 90)
    for test_name, _ in test_sets:
        sub = comp_df[comp_df["test_set"] == test_name]
        if sub.empty:
            continue
        print(f"\n  --- {test_name} ---")
        for _, r in sub.iterrows():
            diff = r["PPO_minus_NormalizedAvalanche"]
            print(f"    {r['metric']:25s}  PPO: {r['PPO']:12.4f}  NormalizedAvalanche: {r['NormalizedAvalanche']:12.4f}  Diff (PPO-NA): {diff:+.4f}  -> {r['better']} better")
    print()

    # Summary: who wins on interest (primary metric) per test set
    interest_rows = comp_df[(comp_df["metric"] == "Total interest ($)")]
    print("  --- Summary: Total interest (lower is better) ---")
    for _, r in interest_rows.iterrows():
        print(f"    {r['test_set']:28s}  PPO ${r['PPO']:,.0f}  NormalizedAvalanche ${r['NormalizedAvalanche']:,.0f}  -> {r['better']} wins")
    print()

    # Markdown report
    md_path = output_dir / "ppo_vs_normalized_avalanche_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PPO vs Normalized Avalanche\n\n")
        f.write("| Test set | Metric | PPO | NormalizedAvalanche | Diff (PPO - NA) | Better |\n")
        f.write("|----------|--------|-----|--------------------|-----------------|--------|\n")
        for _, r in comp_df.iterrows():
            d = r["PPO_minus_NormalizedAvalanche"]
            f.write(f"| {r['test_set']} | {r['metric']} | {r['PPO']:.4f} | {r['NormalizedAvalanche']:.4f} | {d:+.4f} | {r['better']} |\n")
        f.write("\n## Total interest (primary)\n\n")
        for _, r in interest_rows.iterrows():
            f.write(f"- **{r['test_set']}**: PPO ${r['PPO']:,.0f}, NormalizedAvalanche ${r['NormalizedAvalanche']:,.0f} -> **{r['better']}** wins.\n")
    print(f"Saved {md_path}")

    # Bar chart: Total interest by test set (PPO vs Normalized Avalanche)
    if _HAS_MPL and not args.no_plot:
        test_names = interest_rows["test_set"].unique().tolist()
        ppo_vals = [comp_df[(comp_df["test_set"] == t) & (comp_df["metric"] == "Total interest ($)")]["PPO"].values[0] for t in test_names]
        na_vals = [comp_df[(comp_df["test_set"] == t) & (comp_df["metric"] == "Total interest ($)")]["NormalizedAvalanche"].values[0] for t in test_names]
        x = range(len(test_names))
        w = 0.35
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.bar([i - w/2 for i in x], ppo_vals, w, label="PPO", color="C0")
        ax.bar([i + w/2 for i in x], na_vals, w, label="Normalized Avalanche", color="C1")
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=20, ha="right")
        ax.set_ylabel("Total interest ($)")
        ax.set_title("PPO vs Normalized Avalanche: Total interest paid (lower is better)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / "ppo_vs_normalized_avalanche_interest.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_dir / 'ppo_vs_normalized_avalanche_interest.png'}")


if __name__ == "__main__":
    main()
