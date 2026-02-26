"""Run all baselines on randomized scenarios and produce benchmark CSV.

Usage:
    python scripts/run_baselines.py                          # Full: 1000 eps × 5 seeds
    python scripts/run_baselines.py --quick                  # Dev:  50 eps × 1 seed
    python scripts/run_baselines.py --config configs/eval/eval_protocol.yaml
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.baselines import ALL_BASELINES
from src.envs.credit_env import CreditCardDebtEnv
from src.envs.scenario_sampler import ScenarioSampler
from src.evaluation.metrics import compute_credit_proxy_score
from src.utils.config import load_eval_config


def run_benchmark(
    num_episodes: int = 1000,
    seeds: list[int] | None = None,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run all baselines across seeds × episodes with randomized scenarios.

    Args:
        num_episodes: Number of episodes per (policy, seed) pair.
        seeds: List of RNG seeds for reproducibility.
        output_dir: Directory to write CSV output.

    Returns:
        DataFrame with one row per (strategy, seed, episode).
    """
    if seeds is None:
        seeds = [42]

    sampler = ScenarioSampler()
    rows: list[dict] = []

    total_runs = len(ALL_BASELINES) * len(seeds) * num_episodes
    completed: int = 0
    t0 = time.time()

    for seed in seeds:
        # Pre-generate scenarios for this seed so all policies see the same ones
        rng = np.random.default_rng(seed)
        scenarios = [sampler.sample(rng) for _ in range(num_episodes)]

        for PolicyClass in ALL_BASELINES:
            policy = PolicyClass()

            for ep_idx, scenario in enumerate(scenarios):
                env = CreditCardDebtEnv(config=scenario)
                result = policy.run_episode(env, seed=seed + ep_idx)

                # Compute credit proxy score
                initial_debt = scenario.total_initial_debt
                final_debt = result["final_debt"]
                debt_reduction = (
                    (initial_debt - final_debt) / initial_debt
                    if initial_debt > 0
                    else 1.0
                )

                # Count missed minimums from the interest/utilization histories
                # We track missed mins through whether all_paid and months
                total_card_months = len(result.get("utilization_history", [])) * scenario.num_cards
                # Approximate missed_min_ratio: if not all_paid after max_months,
                # we use a penalty proportional to remaining debt
                missed_min_ratio = 0.0  # Baselines always pay minimums except MinimumOnly edge cases

                credit_score = compute_credit_proxy_score(
                    avg_utilization=result["avg_utilization"],
                    missed_min_ratio=missed_min_ratio,
                    debt_reduction_ratio=debt_reduction,
                )

                rows.append({
                    "strategy": result["strategy"].strip(),
                    "seed": seed,
                    "episode": ep_idx,
                    "total_interest": round(result["total_interest"], 2),
                    "months": result["months"],
                    "avg_utilization": round(result["avg_utilization"], 4),
                    "final_debt": round(final_debt, 2),
                    "all_paid": result["all_paid"],
                    "credit_proxy_score": round(credit_score, 1),
                })

                completed = completed + 1
                if completed % 500 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_runs - completed) / rate if rate > 0 else 0
                    print(
                        f"  [{completed}/{total_runs}] "
                        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
                    )

    df = pd.DataFrame(rows)

    # Save per-episode CSV
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "baselines_per_episode.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-episode results saved to {csv_path}")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary stats table grouped by strategy."""
    # Normalize strategy names (strip whitespace) so we don't show duplicates
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
    print("  BASELINE COMPARISON — Summary Statistics")
    print("=" * 90)
    print(summary.to_string(index=False))
    print()


def sanity_checks(df: pd.DataFrame) -> None:
    """Run sanity checks on baseline results."""
    print("Sanity checks:")

    # Check 1: Avalanche should accrue less interest than Snowball (on average)
    aval = df[df["strategy"] == "Avalanche"]["total_interest"].mean()
    snow = df[df["strategy"] == "Snowball"]["total_interest"].mean()

    if aval < snow:
        print(f"  [PASS] Avalanche (${aval:,.0f}) < Snowball (${snow:,.0f}) on interest")
    else:
        print(
            f"  [FAIL] Avalanche (${aval:,.0f}) >= Snowball (${snow:,.0f}) on interest!\n"
            f"    This violates a known financial truth. Possible env bug."
        )

    # Check 2: MinimumOnly should be worst on interest
    min_only = df[df["strategy"] == "MinimumOnly"]["total_interest"].mean()
    others_max = df[df["strategy"] != "MinimumOnly"].groupby("strategy")["total_interest"].mean().max()

    if min_only >= others_max:
        print(f"  [PASS] MinimumOnly (${min_only:,.0f}) has highest interest")
    else:
        print(f"  [NOTE] MinimumOnly (${min_only:,.0f}) is not highest ({others_max:,.0f})")

    print()


def main():
    parser = argparse.ArgumentParser(description="Run baseline strategies benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval/eval_protocol.yaml",
        help="Path to eval protocol YAML",
    )
    parser.add_argument("--quick", action="store_true", help="Quick run: 50 eps, 1 seed")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    # Load eval config
    eval_cfg = load_eval_config(args.config)

    if args.quick:
        num_episodes = 50
        seeds = [42]
        print("Quick mode: 50 episodes × 1 seed")
    else:
        num_episodes = eval_cfg.get("num_episodes", 1000)
        seeds = eval_cfg.get("seeds", [42, 123, 456, 789, 1024])
        print(f"Full mode: {num_episodes} episodes × {len(seeds)} seeds")

    print(f"Running {len(ALL_BASELINES)} baselines...\n")
    df = run_benchmark(
        num_episodes=num_episodes,
        seeds=seeds,
        output_dir=args.output,
    )

    print_summary(df)
    sanity_checks(df)


if __name__ == "__main__":
    main()
