"""Report performance gap: in-distribution vs held-out (no retraining).

Reads in-distribution summary (ppo_vs_baselines_summary.csv) and held-out
summaries (held_out_summary_<preset>.csv), computes gaps for PPO and
optionally baselines, and writes results/performance_gap_report.csv.

Usage:
    python scripts/report_performance_gap.py
    python scripts/report_performance_gap.py --results-dir results --output results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_in_distribution(path: Path) -> pd.DataFrame:
    """Load in-distribution summary (one row per strategy)."""
    if not path.exists():
        raise FileNotFoundError(
            f"In-distribution summary not found: {path}. "
            "Run: python scripts/evaluate_ppo_vs_baselines.py"
        )
    return pd.read_csv(path)


def load_held_out_summaries(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all held_out_summary_*.csv files. Key = preset name (from filename)."""
    out = {}
    for f in results_dir.glob("held_out_summary_*.csv"):
        preset = f.stem.replace("held_out_summary_", "")
        out[preset] = pd.read_csv(f)
    return out


def main():
    parser = argparse.ArgumentParser(description="Report performance gap (in-dist vs held-out)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / args.results_dir
    output_dir = root / args.output

    in_dist_path = results_dir / "ppo_vs_baselines_summary.csv"
    try:
        in_dist = load_in_distribution(in_dist_path)
    except FileNotFoundError as e:
        print(e)
        return

    held_out = load_held_out_summaries(results_dir)
    if not held_out:
        print(
            "No held-out summaries found in {}. Run:\n"
            "  python scripts/evaluate_held_out.py --preset low_apr_low_income\n"
            "  python scripts/evaluate_held_out.py --preset high_apr_high_income\n"
            "  python scripts/evaluate_held_out.py --preset extended_ranges".format(
                results_dir
            )
        )
        return

    # Metrics to report and their direction (lower is better for interest, months; higher for paid_off, credit)
    metrics = [
        ("interest_mean", "Total interest ($)", "lower"),
        ("months_mean", "Months to payoff", "lower"),
        ("util_mean", "Avg utilization", "lower"),
        ("paid_off_pct", "Paid off (%)", "higher"),
        ("credit_score_mean", "Credit score", "higher"),
    ]

    # Build gap table: for each (strategy, preset), gap = held_out - in_dist
    in_dist_by_strategy = in_dist.set_index("strategy")
    rows = []
    for preset, ho_df in held_out.items():
        for _, r in ho_df.iterrows():
            strat = r["strategy"].strip()
            if strat not in in_dist_by_strategy.index:
                continue
            id_row = in_dist_by_strategy.loc[strat]
            row = {"preset": preset, "strategy": strat}
            for col, label, _ in metrics:
                if col not in r or col not in id_row:
                    continue
                val_id = id_row[col]
                val_ho = r[col]
                gap = val_ho - val_id
                row[f"{col}_in_dist"] = round(val_id, 4)
                row[f"{col}_held_out"] = round(val_ho, 4)
                row[f"{col}_gap"] = round(gap, 4)
            rows.append(row)

    gap_df = pd.DataFrame(rows)
    if gap_df.empty:
        print("No matching strategies between in-dist and held-out.")
        return

    # Summary table: PPO only, one row per preset
    ppo_gap = gap_df[gap_df["strategy"] == "PPO"].copy()
    ppo_gap = ppo_gap.drop(columns=["strategy"])

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "performance_gap_report.csv"
    gap_df.to_csv(report_path, index=False)
    print(f"Full gap table saved to {report_path}")

    # Print performance gap report (PPO focus)
    print("\n" + "=" * 80)
    print("  PERFORMANCE GAP: In-distribution vs held-out (no retraining)")
    print("=" * 80)
    print("\nIn-distribution: fixed test set (seed 999, 1000 episodes, 3-card, APR 12–29%, income $3k–$8k)")
    print("Held-out: different APR/income/cards per preset (seed 4242, 1000 episodes)\n")

    for preset in sorted(held_out.keys()):
        sub = ppo_gap[ppo_gap["preset"] == preset]
        if sub.empty:
            continue
        print(f"  --- Held-out preset: {preset} ---")
        r = sub.iloc[0]
        for col, label, direction in metrics:
            id_val = r.get(f"{col}_in_dist")
            ho_val = r.get(f"{col}_held_out")
            gap_val = r.get(f"{col}_gap")
            if id_val is None or ho_val is None or gap_val is None:
                continue
            print(f"    {label:25s}  In-dist: {id_val:12.4f}  Held-out: {ho_val:12.4f}  Gap: {gap_val:+.4f}")
        print()

    # One-line summary table
    print("  --- PPO performance gap summary (held_out - in_dist) ---")
    summary_rows = []
    for preset in sorted(held_out.keys()):
        sub = ppo_gap[ppo_gap["preset"] == preset]
        if sub.empty:
            continue
        r = sub.iloc[0]
        summary_rows.append({
            "preset": preset,
            "interest_gap": r.get("interest_mean_gap"),
            "months_gap": r.get("months_mean_gap"),
            "paid_off_gap_pct": r.get("paid_off_pct_gap"),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_path = output_dir / "performance_gap_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}\n")


if __name__ == "__main__":
    main()
