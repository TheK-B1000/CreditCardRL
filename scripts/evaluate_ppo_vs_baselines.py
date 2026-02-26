"""Evaluate trained PPO vs all baselines on a fixed 1,000-scenario test set (P2.4).

Uses a fixed RNG seed so the same 1,000 scenarios are used for every strategy.
Scenarios are 3-card to match the default PPO training env.

Usage:
    python scripts/evaluate_ppo_vs_baselines.py
    python scripts/evaluate_ppo_vs_baselines.py --model models/best/best_model.zip
    python scripts/evaluate_ppo_vs_baselines.py --episodes 1000 --seed 999 --output results
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
from src.envs.scenario_sampler import ScenarioSampler
from src.evaluation.metrics import compute_credit_proxy_score

# Fixed test seed for reproducible 1,000-scenario test set
DEFAULT_TEST_SEED = 999
NUM_TEST_EPISODES = 1000
# PPO is trained on 3-card env; test set must match
NUM_CARDS_TEST = 3


def run_ppo_episode(env: CreditCardDebtEnv, model: PPO, seed: int) -> dict:
    """Run one episode with the PPO model. Returns same shape as baseline EpisodeResult."""
    obs, info = env.reset(seed=seed)
    total_interest = 0.0
    utilization_history = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_interest += sum(info.get("interests", []))
        utilization_history.append(info.get("overall_utilization", 0.0))
    avg_util = float(np.mean(utilization_history)) if utilization_history else 0.0
    return {
        "strategy": "PPO",
        "total_interest": total_interest,
        "months": info.get("month", env.month),
        "avg_utilization": avg_util,
        "final_debt": info.get("total_debt", 0.0),
        "all_paid": all(c.is_paid_off for c in env.cards),
        "credit_proxy_score": 0.0,  # filled below
    }


def run_benchmark(
    model_path: str | Path,
    num_episodes: int = NUM_TEST_EPISODES,
    test_seed: int = DEFAULT_TEST_SEED,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run PPO + all baselines on the same fixed set of scenarios."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = PPO.load(str(model_path))

    # Fixed 3-card scenarios so PPO (trained on 3-card) works
    sampler = ScenarioSampler(num_cards_range=(NUM_CARDS_TEST, NUM_CARDS_TEST))
    rng = np.random.default_rng(test_seed)
    scenarios = [sampler.sample(rng) for _ in range(num_episodes)]

    rows: list[dict] = []
    strategies = [("PPO", None)] + [(p().name, p()) for p in ALL_BASELINES]
    total_runs = len(strategies) * num_episodes
    completed = 0
    t0 = time.time()

    for ep_idx, scenario in enumerate(scenarios):
        env = CreditCardDebtEnv(config=scenario)
        seed_ep = test_seed + ep_idx

        # PPO
        result = run_ppo_episode(env, model, seed_ep)
        initial_debt = scenario.total_initial_debt
        debt_reduction = (
            (initial_debt - result["final_debt"]) / initial_debt if initial_debt > 0 else 1.0
        )
        result["credit_proxy_score"] = round(
            compute_credit_proxy_score(
                avg_utilization=result["avg_utilization"],
                missed_min_ratio=0.0,
                debt_reduction_ratio=debt_reduction,
            ),
            1,
        )
        rows.append({
            "strategy": result["strategy"],
            "seed": test_seed,
            "episode": ep_idx,
            "total_interest": round(result["total_interest"], 2),
            "months": result["months"],
            "avg_utilization": round(result["avg_utilization"], 4),
            "final_debt": round(result["final_debt"], 2),
            "all_paid": result["all_paid"],
            "credit_proxy_score": result["credit_proxy_score"],
        })
        completed += 1

        # Baselines
        for name, policy in strategies[1:]:
            if policy is None:
                continue
            env_b = CreditCardDebtEnv(config=scenario)
            res = policy.run_episode(env_b, seed=seed_ep)
            initial_debt = scenario.total_initial_debt
            debt_reduction = (
                (initial_debt - res["final_debt"]) / initial_debt if initial_debt > 0 else 1.0
            )
            credit_score = compute_credit_proxy_score(
                avg_utilization=res["avg_utilization"],
                missed_min_ratio=0.0,
                debt_reduction_ratio=debt_reduction,
            )
            rows.append({
                "strategy": res["strategy"].strip(),
                "seed": test_seed,
                "episode": ep_idx,
                "total_interest": round(res["total_interest"], 2),
                "months": res["months"],
                "avg_utilization": round(res["avg_utilization"], 4),
                "final_debt": round(res["final_debt"], 2),
                "all_paid": res["all_paid"],
                "credit_proxy_score": round(credit_score, 1),
            })
            completed += 1

        if completed % 2000 == 0 or completed == total_runs:
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"  [{completed}/{total_runs}] {elapsed:.0f}s elapsed")

    df = pd.DataFrame(rows)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "ppo_vs_baselines_per_episode.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary stats by strategy."""
    df = df.copy()
    df["strategy"] = df["strategy"].str.strip()
    summary_rows = []
    for strategy, group in df.groupby("strategy", sort=False):
        summary_rows.append({
            "Strategy": strategy,
            "Interest (mean±std)": f"${group['total_interest'].mean():,.0f} ± ${group['total_interest'].std():,.0f}",
            "Months (mean±std)": f"{group['months'].mean():.1f} ± {group['months'].std():.1f}",
            "Util (mean)": f"{group['avg_utilization'].mean():.3f}",
            "Paid Off %": f"{group['all_paid'].mean() * 100:.1f}%",
            "Credit Score (mean)": f"{group['credit_proxy_score'].mean():.0f}",
        })
    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print("  PPO vs BASELINES — Fixed 1,000-scenario test set")
    print("=" * 90)
    print(summary.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO vs baselines on fixed test set")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best/best_model.zip",
        help="Path to trained PPO model (.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_TEST_EPISODES,
        help=f"Number of test scenarios (default {NUM_TEST_EPISODES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TEST_SEED,
        help=f"RNG seed for fixed test set (default {DEFAULT_TEST_SEED})",
    )
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    print(f"Loading PPO from {args.model}")
    print(f"Fixed test set: {args.episodes} scenarios (3-card), seed={args.seed}")
    print(f"Strategies: PPO + {len(ALL_BASELINES)} baselines\n")

    df = run_benchmark(
        model_path=args.model,
        num_episodes=args.episodes,
        test_seed=args.seed,
        output_dir=args.output,
    )
    print_summary(df)


if __name__ == "__main__":
    main()
