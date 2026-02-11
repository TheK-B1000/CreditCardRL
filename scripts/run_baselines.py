"""Run all baselines on configurable scenarios and print comparison table.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --config configs/env/stress_5card.yaml --episodes 200
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.baselines import (
    AvalanchePolicy,
    MinimumOnlyPolicy,
    NormalizedAvalanchePolicy,
    RandomPolicy,
    SnowballPolicy,
)
from src.envs.credit_env import CreditCardDebtEnv
from src.utils.config import load_env_config, EnvConfig


def run_baselines(env_config: EnvConfig, num_episodes: int = 100, seed: int = 42):
    """Run all baselines and collect metrics."""
    policies = [
        MinimumOnlyPolicy(),
        SnowballPolicy(),
        AvalanchePolicy(),
        NormalizedAvalanchePolicy(),
        RandomPolicy(seed=seed),
    ]

    results = []
    for policy in policies:
        interests = []
        months_list = []
        utils_list = []
        paid_count = 0

        for ep in range(num_episodes):
            env = CreditCardDebtEnv(config=env_config)
            result = policy.run_episode(env, seed=seed + ep)
            interests.append(result["total_interest"])
            months_list.append(result["months"])
            utils_list.append(result["avg_utilization"])
            if result["all_paid"]:
                paid_count += 1

        results.append({
            "Strategy": policy.name,
            "Interest (mean)": f"${np.mean(interests):,.2f}",
            "Interest (std)": f"${np.std(interests):,.2f}",
            "Months (mean)": f"{np.mean(months_list):.1f}",
            "Months (std)": f"{np.std(months_list):.1f}",
            "Avg Util (mean)": f"{np.mean(utils_list):.3f}",
            "Paid Off %": f"{100 * paid_count / num_episodes:.0f}%",
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Run baseline strategies")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/env/default_3card.yaml",
        help="Path to env config YAML",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Save CSV to path")
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    env_config = load_env_config(args.config)

    print(f"Running {args.episodes} episodes per baseline...\n")
    df = run_baselines(env_config, args.episodes, args.seed)

    print(df.to_string(index=False))
    print()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
