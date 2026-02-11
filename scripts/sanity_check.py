"""Sanity check — step through episodes with rendered output.

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --config configs/env/single_card.yaml --episodes 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.baselines import AvalanchePolicy, MinimumOnlyPolicy, SnowballPolicy
from src.envs.credit_env import CreditCardDebtEnv
from src.utils.config import load_env_config


def main():
    parser = argparse.ArgumentParser(description="Sanity check: walk through episodes")
    parser.add_argument("--config", type=str, default="configs/env/default_3card.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env_config = load_env_config(args.config)
    policies = [MinimumOnlyPolicy(), SnowballPolicy(), AvalanchePolicy()]

    for ep, policy in enumerate(policies[:args.episodes]):
        print(f"\n{'#'*60}")
        print(f"  EPISODE {ep + 1}: Strategy = {policy.name}")
        print(f"{'#'*60}")

        env = CreditCardDebtEnv(config=env_config, render_mode="human")
        obs, info = env.reset(seed=args.seed + ep)

        terminated = truncated = False
        prev_debt = info["total_debt"]

        while not (terminated or truncated):
            action = policy.allocate(env)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            # Sanity checks
            current_debt = info["total_debt"]
            if current_debt > prev_debt + 100:
                print(f"  ⚠️ WARNING: Debt increased by ${current_debt - prev_debt:.2f}!")
            prev_debt = current_debt

        status = "✓ ALL PAID OFF" if all(c.is_paid_off for c in env.cards) else "✗ TIMED OUT"
        print(f"\n  Result: {status} after {info['month']} months")
        print(f"  Final debt: ${info['total_debt']:,.2f}")


if __name__ == "__main__":
    main()
