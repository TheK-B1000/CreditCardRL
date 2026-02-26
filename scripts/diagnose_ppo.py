"""Diagnose PPO vs Avalanche when PPO underperforms (P2.5).

Reads results from evaluate_ppo_vs_baselines.py (ppo_vs_baselines_per_episode.csv)
and suggests next steps: diverse scenarios, reward/obs tweaks.

Usage:
    python scripts/diagnose_ppo.py
    python scripts/diagnose_ppo.py --input results/ppo_vs_baselines_per_episode.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Diagnose PPO underperformance vs Avalanche")
    parser.add_argument(
        "--input",
        type=str,
        default="results/ppo_vs_baselines_per_episode.csv",
        help="CSV from evaluate_ppo_vs_baselines.py",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        print("Run: python scripts/evaluate_ppo_vs_baselines.py")
        sys.exit(1)

    df = pd.read_csv(path)
    df["strategy"] = df["strategy"].str.strip()

    ppo_mean = df[df["strategy"] == "PPO"]["total_interest"].mean()
    aval_mean = df[df["strategy"] == "Avalanche"]["total_interest"].mean()
    norm_aval_mean = df[df["strategy"] == "NormalizedAvalanche"]["total_interest"].mean()

    print("=" * 60)
    print("  PPO vs Avalanche — Diagnostic (P2.5)")
    print("=" * 60)
    print(f"  PPO (mean total interest):     ${ppo_mean:,.0f}")
    print(f"  Avalanche (mean total interest): ${aval_mean:,.0f}")
    print(f"  NormalizedAvalanche (mean):    ${norm_aval_mean:,.0f}")
    print()

    if ppo_mean <= aval_mean:
        print("  [OK] PPO does not underperform Avalanche on this test set.")
        print("  No diagnosis needed.")
        return

    print("  PPO underperforms Avalanche. Next steps (already applied: dense")
    print("  interest penalty eta, utilization bonus beta_below, affordability obs,")
    print("  interest-to-date obs, stronger payoff bonus delta, diverse_scenarios):\n")
    print("  1. FURTHER REWARD / OBS")
    print("     → Tune eta, delta, or add months-to-payoff in obs (see src/envs/).\n")
    print("  2. TRAINING")
    print("     → Train longer (e.g. 2M steps) or tune lr, n_steps, batch_size.\n")
    print("  After changes: re-train, re-run evaluate_ppo_vs_baselines.py, then this script.")
    print("=" * 60)


if __name__ == "__main__":
    main()
