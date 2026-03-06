"""P4.1 — Evaluate PPO and baselines on a held-out test set.

The held-out set uses different APR ranges, income levels, and optionally
different number of cards than the training distribution, to measure
generalization and robustness.

Training (in-distribution): 3 cards, APR 12–29%, income $3k–$8k.
Held-out presets: see configs/eval/held_out_test.yaml.

Usage:
    python scripts/evaluate_held_out.py
    python scripts/evaluate_held_out.py --config configs/eval/held_out_test.yaml --preset high_apr_high_income
    python scripts/evaluate_held_out.py --preset extended_ranges --episodes 500 --output results/held_out
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.baselines import ALL_BASELINES
from src.envs.credit_env import CreditCardDebtEnv
from src.envs.scenario_sampler import ScenarioSampler
from src.evaluation.metrics import compute_credit_proxy_score


def load_held_out_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def run_ppo_episode(env: CreditCardDebtEnv, model: PPO, seed: int) -> dict:
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
        "credit_proxy_score": 0.0,
    }


def run_held_out_benchmark(
    model_path: Path,
    preset_config: dict,
    seed: int,
    num_episodes: int,
    output_dir: Path,
    file_suffix: str = "",
) -> pd.DataFrame:
    """Run PPO (if 3-card) + all baselines on the held-out scenario set."""
    model_path = Path(model_path)
    num_cards_range = tuple(preset_config["num_cards_range"])
    assert len(num_cards_range) == 2
    num_cards_min, num_cards_max = num_cards_range[0], num_cards_range[1]
    # For fixed card count, use same min/max
    sampler = ScenarioSampler(
        num_cards_range=(num_cards_min, num_cards_max),
        apr_range=tuple(preset_config["apr_range"]),
        balance_range=tuple(preset_config["balance_range"]),
        income_range=tuple(preset_config["income_range"]),
        expense_ratio_range=tuple(preset_config["expense_ratio_range"]),
    )
    rng = np.random.default_rng(seed)
    scenarios = [sampler.sample(rng) for _ in range(num_episodes)]

    # PPO only supports 3-card (trained obs/action space)
    run_ppo = num_cards_min == 3 and num_cards_max == 3 and model_path.exists()
    model = None
    if run_ppo:
        temp_env = CreditCardDebtEnv(config=scenarios[0])
        model = PPO.load(
            str(model_path),
            custom_objects={
                "observation_space": temp_env.observation_space,
                "action_space": temp_env.action_space,
            },
        )

    strategies = ([("PPO", None)] if run_ppo else []) + [(p().name, p()) for p in ALL_BASELINES]
    total_runs = len(strategies) * num_episodes
    rows: list[dict] = []
    completed = 0
    t0 = time.time()

    for ep_idx, scenario in enumerate(scenarios):
        env = CreditCardDebtEnv(config=scenario)
        seed_ep = seed + ep_idx
        initial_debt = scenario.total_initial_debt

        if run_ppo:
            result = run_ppo_episode(env, model, seed_ep)
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
                "strategy": result["strategy"].strip(),
                "seed": seed,
                "episode": ep_idx,
                "total_interest": round(result["total_interest"], 2),
                "months": result["months"],
                "avg_utilization": round(result["avg_utilization"], 4),
                "final_debt": round(result["final_debt"], 2),
                "all_paid": result["all_paid"],
                "credit_proxy_score": result["credit_proxy_score"],
            })
            completed += 1

        for name, policy in strategies[1:] if run_ppo else strategies:
            if policy is None:
                continue
            env_b = CreditCardDebtEnv(config=scenario)
            res = policy.run_episode(env_b, seed=seed_ep)
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
                "seed": seed,
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
            print(f"  [{completed}/{total_runs}] {elapsed:.0f}s elapsed")

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"held_out_per_episode{file_suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nHeld-out results saved to {csv_path}")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["strategy"] = df["strategy"].str.strip()
    summary_rows = []
    for strategy, group in df.groupby("strategy", sort=False):
        summary_rows.append({
            "strategy": strategy,
            "interest_mean": round(group["total_interest"].mean(), 2),
            "interest_std": round(group["total_interest"].std(), 2),
            "months_mean": round(group["months"].mean(), 2),
            "months_std": round(group["months"].std(), 2),
            "util_mean": round(group["avg_utilization"].mean(), 4),
            "paid_off_pct": round(group["all_paid"].mean() * 100, 2),
            "credit_score_mean": round(group["credit_proxy_score"].mean(), 2),
            "n_episodes": len(group),
        })
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description="P4.1 — Evaluate on held-out test set (different APR, income, cards)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval/held_out_test.yaml",
        help="Path to held-out test config YAML",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name (e.g. low_apr_low_income, high_apr_high_income, fewer_cards, more_cards). Default from config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best/best_model.zip",
        help="Path to PPO model (used only for 3-card held-out)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes (default from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory; summary and per-episode CSVs written here",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for output files (e.g. _low_apr) so multiple presets don't overwrite",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    config_path = root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_held_out_config(config_path)
    seed = int(cfg.get("seed", 4242))
    num_episodes = args.episodes or int(cfg.get("num_episodes", 1000))
    preset_name = args.preset or cfg.get("default_preset", "low_apr_low_income")

    if preset_name not in cfg or not isinstance(cfg.get(preset_name), dict):
        valid = [k for k in cfg if isinstance(cfg.get(k), dict)]
        raise ValueError(f"Unknown preset {preset_name!r}. Valid presets: {valid}")
    preset_config = cfg[preset_name]

    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    num_cards = preset_config["num_cards_range"]
    run_ppo = num_cards[0] == 3 and num_cards[1] == 3
    print(f"Held-out test set (P4.1)")
    print(f"  Config: {config_path}")
    print(f"  Preset: {preset_name}")
    print(f"  Episodes: {num_episodes}, seed: {seed}")
    print(f"  Num cards: {num_cards}")
    print(f"  APR range: {preset_config['apr_range']}, income range: {preset_config['income_range']}")
    if run_ppo:
        print(f"  Model: {args.model} (3-card; will run PPO + baselines)")
    else:
        print(f"  PPO skipped (model is 3-card; this preset uses {num_cards} cards). Baselines only.")
    print()

    df = run_held_out_benchmark(
        model_path=root / args.model,
        preset_config=preset_config,
        seed=seed,
        num_episodes=num_episodes,
        output_dir=output_dir,
        file_suffix=args.suffix or f"_{preset_name}",
    )

    summary_df = build_summary(df)
    suffix = args.suffix or f"_{preset_name}"
    summary_path = output_dir / f"held_out_summary{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"  HELD-OUT TEST — {preset_name} (mean ± std)")
    print("=" * 80)
    for _, r in summary_df.iterrows():
        print(
            f"  {r['strategy']:20s}  Interest: ${r['interest_mean']:,.0f} ± ${r['interest_std']:,.0f}  "
            f"Months: {r['months_mean']:.1f} ± {r['months_std']:.1f}  "
            f"Util: {r['util_mean']:.4f}  Paid off: {r['paid_off_pct']:.1f}%  "
            f"Credit: {r['credit_score_mean']:.1f}"
        )
    print()


if __name__ == "__main__":
    main()
