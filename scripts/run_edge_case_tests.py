"""Run edge-case tests: single card, same APR, 0% intro rate, income shock mid-episode.

These tests stress the agent and baselines on scenarios that differ from typical training.
Single-card runs baselines only (PPO is trained for 3-card obs/action).

Usage:
    python scripts/run_edge_case_tests.py
    python scripts/run_edge_case_tests.py --model models/best/best_model.zip --episodes 50 --output results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.baselines import ALL_BASELINES
from src.envs.credit_env import CreditCardDebtEnv
from src.evaluation.metrics import compute_credit_proxy_score
from src.utils.config import CardConfig, EnvConfig
from src.envs.reward import RewardConfig


def make_edge_case_configs() -> dict[str, EnvConfig]:
    """Define 2–3 edge-case scenarios (and income shock)."""
    reward = RewardConfig()
    cases = {}

    # 1. Single card — PPO is 3-card so we run baselines only
    cases["single_card"] = EnvConfig(
        cards=[
            CardConfig("Solo Card", apr=0.18, balance=5000.0, credit_limit=10000.0),
        ],
        monthly_income=5000.0,
        fixed_expenses=3200.0,
        action_mode="continuous",
        max_months=60,
        utilization_target=0.30,
        reward=reward,
    )

    # 2. All cards same APR — no “avalanche” advantage; allocation is about utilization/time
    cases["same_apr"] = EnvConfig(
        cards=[
            CardConfig("Card A", apr=0.18, balance=3000.0, credit_limit=6000.0),
            CardConfig("Card B", apr=0.18, balance=4000.0, credit_limit=8000.0),
            CardConfig("Card C", apr=0.18, balance=5000.0, credit_limit=10000.0),
        ],
        monthly_income=5000.0,
        fixed_expenses=3200.0,
        action_mode="continuous",
        max_months=60,
        utilization_target=0.30,
        reward=reward,
    )

    # 3. One card at 0% intro rate — optimal is to pay down high-APR cards first while min on 0% card
    cases["zero_pct_intro"] = EnvConfig(
        cards=[
            CardConfig("Intro 0%", apr=0.00, balance=4000.0, credit_limit=8000.0),
            CardConfig("Standard", apr=0.18, balance=4000.0, credit_limit=8000.0),
            CardConfig("High APR", apr=0.24, balance=4000.0, credit_limit=8000.0),
        ],
        monthly_income=5000.0,
        fixed_expenses=3200.0,
        action_mode="continuous",
        max_months=60,
        utilization_target=0.30,
        reward=reward,
    )

    # 4. Income shock mid-episode — income drops after month 6
    # First 6 months $5000, then $3500 (tight budget)
    income_schedule = [5000.0] * 6 + [3500.0] * 54
    cases["income_shock"] = EnvConfig(
        cards=[
            CardConfig("Card A", apr=0.18, balance=3000.0, credit_limit=6000.0),
            CardConfig("Card B", apr=0.20, balance=4000.0, credit_limit=8000.0),
            CardConfig("Card C", apr=0.22, balance=3000.0, credit_limit=6000.0),
        ],
        monthly_income=5000.0,
        fixed_expenses=3200.0,
        action_mode="continuous",
        max_months=60,
        utilization_target=0.30,
        reward=reward,
        income_schedule=income_schedule,
    )

    return cases


def run_episode_ppo(env: CreditCardDebtEnv, model: PPO, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total_interest = 0.0
    utilization_history = []
    term = trunc = False
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        total_interest += sum(info.get("interests", []))
        utilization_history.append(info.get("overall_utilization", 0.0))
    avg_util = float(np.mean(utilization_history)) if utilization_history else 0.0
    initial = env.initial_total_debt
    final = info.get("total_debt", 0.0)
    debt_red = (initial - final) / initial if initial > 0 else 1.0
    score = compute_credit_proxy_score(avg_utilization=avg_util, missed_min_ratio=0.0, debt_reduction_ratio=debt_red)
    return {
        "total_interest": total_interest,
        "months": info.get("month", env.month),
        "avg_utilization": avg_util,
        "all_paid": all(c.is_paid_off for c in env.cards),
        "credit_proxy_score": score,
    }


def run_episode_baseline(env: CreditCardDebtEnv, policy, seed: int) -> dict:
    res = policy.run_episode(env, seed=seed)
    initial = env.initial_total_debt
    debt_red = (initial - res["final_debt"]) / initial if initial > 0 else 1.0
    score = compute_credit_proxy_score(
        avg_utilization=res["avg_utilization"], missed_min_ratio=0.0, debt_reduction_ratio=debt_red
    )
    return {
        "total_interest": res["total_interest"],
        "months": res["months"],
        "avg_utilization": res["avg_utilization"],
        "all_paid": res["all_paid"],
        "credit_proxy_score": score,
    }


def main():
    parser = argparse.ArgumentParser(description="Run edge-case tests")
    parser.add_argument("--model", type=str, default="models/best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--seed", type=int, default=8888)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    model_path = root / args.model
    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = make_edge_case_configs()
    all_rows = []

    for case_name, config in configs.items():
        n_cards = config.num_cards
        run_ppo = n_cards == 3 and model_path.exists()
        env = CreditCardDebtEnv(config=config)
        model = None
        if run_ppo:
            model = PPO.load(
                str(model_path),
                custom_objects={
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                },
            )

        strategies = ([("PPO", None)] if run_ppo else []) + [(p().name, p()) for p in ALL_BASELINES]
        for ep in range(args.episodes):
            seed = args.seed + hash(case_name) % 100000 + ep
            for strat_name, policy in strategies:
                env_i = CreditCardDebtEnv(config=config)
                if policy is None:
                    res = run_episode_ppo(env_i, model, seed)
                else:
                    res = run_episode_baseline(env_i, policy, seed)
                all_rows.append({
                    "edge_case": case_name,
                    "strategy": strat_name.strip(),
                    "episode": ep,
                    "total_interest": round(res["total_interest"], 2),
                    "months": res["months"],
                    "avg_utilization": round(res["avg_utilization"], 4),
                    "all_paid": res["all_paid"],
                    "credit_proxy_score": round(res["credit_proxy_score"], 1),
                })
        print(f"  {case_name}: {len(strategies)} strategies x {args.episodes} episodes")

    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "edge_case_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Summary per edge case and strategy
    summary = df.groupby(["edge_case", "strategy"]).agg(
        interest_mean=("total_interest", "mean"),
        interest_std=("total_interest", "std"),
        months_mean=("months", "mean"),
        paid_off_pct=("all_paid", "mean"),
        credit_mean=("credit_proxy_score", "mean"),
        n=("episode", "count"),
    ).reset_index()
    summary_path = output_dir / "edge_case_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    print("\n" + "=" * 85)
    print("  Edge-case test summary (mean total interest, months, paid off %)")
    print("=" * 85)
    for case_name in configs:
        sub = summary[summary["edge_case"] == case_name]
        if sub.empty:
            continue
        print(f"\n  --- {case_name} ---")
        for _, r in sub.iterrows():
            print(
                f"    {r['strategy']:22s}  Interest: ${r['interest_mean']:,.0f}  "
                f"Months: {r['months_mean']:.1f}  Paid off: {r['paid_off_pct']*100:.1f}%  "
                f"Credit: {r['credit_mean']:.1f}"
            )
    print()


if __name__ == "__main__":
    main()
