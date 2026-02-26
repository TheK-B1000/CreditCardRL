"""Diagnose PPO vs Avalanche when PPO underperforms (P2.5).

Reads results from evaluate_ppo_vs_baselines.py (ppo_vs_baselines_per_episode.csv)
and suggests common fixes: reward shaping, discrete action space, observation features.

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
    print("  interest penalty eta, utilization bonus beta_below, affordability obs):\n")
    print("  1. DISCRETE ACTION SPACE")
    print("     → configs/env/default_3card.yaml set action_mode: discrete")
    print("     → Re-train PPO; discrete allocations can be easier to learn\n")
    print("  2. TRAIN LONGER")
    print("     → python scripts/train_ppo.py --timesteps 1000000\n")
    print("  3. TRAIN ON DIVERSE SCENARIOS")
    print("     → PPO is currently trained on a single scenario (default_3card).")
    print("     → Add curriculum or vectorized envs with random 3-card scenarios")
    print("     → so the policy generalizes to the 1,000-scenario test set.\n")
    print("  4. FURTHER REWARD / OBS (if still behind)")
    print("     → Per-card payoff bonus; interest-to-date or months-to-payoff in obs.\n")
    print("  After changes: re-train, re-run evaluate_ppo_vs_baselines.py, then this script.")
    print("=" * 60)


if __name__ == "__main__":
    main()
