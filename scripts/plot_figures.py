"""Generate publication-style figures: training curve, method comparison, trajectories, scenario breakdown.

Requires: results from evaluate_ppo_vs_baselines.py (per_episode + summary CSVs).
Optional: TensorBoard logs in runs/ for training curve; PPO model for trajectory/breakdown.

Usage:
    python scripts/plot_figures.py
    python scripts/plot_figures.py --results-dir results --figures-dir figures --logdir runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

DEFAULT_TEST_SEED = 999
NUM_CARDS_TEST = 3
# Main baselines for trajectory plots (keep figure readable)
MAIN_BASELINES = ["PPO", "MinimumOnly", "Snowball", "Avalanche"]


def _ensure_mpl():
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required; pip install matplotlib")


# ---------- Plot 1: Training curve (return vs timesteps, mean + shaded CI) ----------


def plot_training_curve(logdir: Path, output_dir: Path) -> None:
    """Plot PPO return vs timesteps with mean and shaded confidence interval across runs/seeds."""
    _ensure_mpl()
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[Plot 1] Skipping: tensorboard not installed. pip install tensorboard")
        return

    if not logdir.exists():
        print(f"[Plot 1] Skipping: logdir {logdir} not found.")
        return

    event_dirs = []
    for d in logdir.iterdir():
        if d.is_dir():
            for e in d.rglob("events.out.tfevents.*"):
                event_dirs.append(e.parent)
                break
    if not event_dirs:
        for e in logdir.rglob("events.out.tfevents.*"):
            event_dirs.append(e.parent)
            break

    if not event_dirs:
        print(f"[Plot 1] No TensorBoard events under {logdir}")
        return

    rew_tag = "rollout/ep_rew_mean"
    steps_tag = "time/total_timesteps"
    all_steps: list[np.ndarray] = []
    all_values: list[np.ndarray] = []

    for event_dir in event_dirs:
        acc = EventAccumulator(str(event_dir))
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if rew_tag not in tags:
            rew_tag_alt = next((t for t in tags if "rew" in t.lower() or "reward" in t.lower()), None)
            if not rew_tag_alt:
                continue
            rew_tag = rew_tag_alt
        events = acc.Scalars(rew_tag)
        if not events:
            continue
        steps = np.array([e.step for e in events])
        if steps_tag in tags and steps_tag != rew_tag:
            step_ev = acc.Scalars(steps_tag)
            if step_ev and len(step_ev) == len(events):
                steps = np.array([e.value for e in step_ev])
        values = np.array([e.value for e in events])
        all_steps.append(steps)
        all_values.append(values)

    if not all_steps:
        print("[Plot 1] No reward data found in any run.")
        return

    # Single run: no CI, just mean line
    if len(all_steps) == 1:
        steps = all_steps[0]
        values = all_values[0]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(steps, values, color="C0", linewidth=2, label="Mean episode return")
    else:
        # Align to common step grid (min step to max step, 500 points)
        all_steps_arr = np.concatenate(all_steps)
        step_min, step_max = all_steps_arr.min(), all_steps_arr.max()
        grid = np.linspace(step_min, step_max, 500)
        interpolated = []
        for s, v in zip(all_steps, all_values):
            if len(s) < 2:
                continue
            vi = np.interp(grid, s, v)
            interpolated.append(vi)
        if not interpolated:
            steps = all_steps[0]
            values = all_values[0]
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.plot(steps, values, color="C0", linewidth=2, label="Mean episode return")
        else:
            interp = np.array(interpolated)
            mean_val = np.mean(interp, axis=0)
            std_val = np.std(interp, axis=0, ddof=1)
            n = interp.shape[0]
            ci = 1.96 * std_val / np.sqrt(n) if n > 1 else 0 * std_val
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.plot(grid, mean_val, color="C0", linewidth=2, label="Mean return")
            ax.fill_between(grid, mean_val - ci, mean_val + ci, color="C0", alpha=0.3, label="95% CI")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode return (mean)")
    ax.set_title("PPO training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "training_curve.png", dpi=150)
    plt.close(fig)
    print(f"[Plot 1] Saved {output_dir / 'training_curve.png'}")


# ---------- Plot 2: Method comparison (interest, months, utilization) with error bars ----------


def plot_method_comparison(per_episode_path: Path, output_dir: Path) -> None:
    """Bar chart: PPO vs baselines on total interest, months to payoff, utilization (with error bars)."""
    _ensure_mpl()
    if not per_episode_path.exists():
        print(f"[Plot 2] Skipping: {per_episode_path} not found.")
        return

    df = pd.read_csv(per_episode_path)
    df["strategy"] = df["strategy"].str.strip()
    summary = df.groupby("strategy").agg(
        interest_mean=("total_interest", "mean"),
        interest_std=("total_interest", "std"),
        months_mean=("months", "mean"),
        months_std=("months", "std"),
        util_mean=("avg_utilization", "mean"),
        util_std=("avg_utilization", "std"),
    ).reset_index()

    strategies = summary["strategy"].tolist()
    x = np.arange(len(strategies))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Total interest paid
    ax = axes[0]
    ax.bar(x - width / 2, summary["interest_mean"], width, yerr=summary["interest_std"], capsize=4, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.set_ylabel("Total interest paid ($)")
    ax.set_title("Total interest paid")
    ax.grid(True, alpha=0.3, axis="y")

    # Months to payoff
    ax = axes[1]
    ax.bar(x - width / 2, summary["months_mean"], width, yerr=summary["months_std"], capsize=4, color="seagreen", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.set_ylabel("Months to payoff")
    ax.set_title("Months to payoff")
    ax.grid(True, alpha=0.3, axis="y")

    # Avg utilization
    ax = axes[2]
    ax.bar(x - width / 2, summary["util_mean"], width, yerr=summary["util_std"], capsize=4, color="coral", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.set_ylabel("Avg utilization")
    ax.set_title("Average utilization")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Method comparison (fixed test set, mean ± std)", fontsize=12, y=1.02)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot 2] Saved {output_dir / 'method_comparison.png'}")


# ---------- Helpers for trajectory plots: run one episode and record balance + payments ----------


def _run_episode_ppo(env, model, seed: int) -> tuple[list[int], list[list[float]], list[list[float]]]:
    """Run one episode with PPO; return (months, balances_per_step, payments_per_step)."""
    obs, _ = env.reset(seed=seed)
    months = [0]
    balances = [[c.balance for c in env.cards]]
    payments_log = []
    term = trunc = False
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        months.append(env.month)
        balances.append([c.balance for c in env.cards])
        payments_log.append(list(info.get("payments", [0] * env.num_cards)))
    return months, balances, payments_log


def _run_episode_baseline(env, policy, seed: int) -> tuple[list[int], list[list[float]], list[list[float]]]:
    """Run one episode with a baseline policy; return (months, balances_per_step, payments_per_step)."""
    obs, _ = env.reset(seed=seed)
    months = [0]
    balances = [[c.balance for c in env.cards]]
    payments_log = []
    term = trunc = False
    while not (term or trunc):
        action = policy.allocate(env)
        obs, _, term, trunc, info = env.step(action)
        months.append(env.month)
        balances.append([c.balance for c in env.cards])
        payments_log.append(list(info.get("payments", [0] * env.num_cards)))
    return months, balances, payments_log


# ---------- Plot 3: Balance trajectory (one scenario, PPO + main baselines) ----------


def plot_balance_trajectory(
    model_path: Path,
    output_dir: Path,
    scenario_episode: int = 0,
    seed: int = DEFAULT_TEST_SEED,
) -> None:
    """Plot remaining balance over time for one scenario: PPO and main baselines."""
    _ensure_mpl()
    if not model_path.exists():
        print(f"[Plot 3] Skipping: model {model_path} not found.")
        return

    from stable_baselines3 import PPO
    from src.envs.credit_env import CreditCardDebtEnv
    from src.envs.scenario_sampler import ScenarioSampler
    from src.baselines import ALL_BASELINES

    sampler = ScenarioSampler(num_cards_range=(NUM_CARDS_TEST, NUM_CARDS_TEST))
    rng = np.random.default_rng(seed)
    for _ in range(scenario_episode):
        sampler.sample(rng)
    scenario = sampler.sample(rng)
    ep_seed = seed + scenario_episode

    env0 = CreditCardDebtEnv(config=scenario)
    model = PPO.load(
        str(model_path),
        custom_objects={
            "observation_space": env0.observation_space,
            "action_space": env0.action_space,
        },
    )

    strategies_to_plot = [s for s in MAIN_BASELINES if s != "NormalizedAvalanche"]  # PPO + 3 baselines
    baseline_map = {p().name: p() for p in ALL_BASELINES}

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    colors = {"PPO": "C0", "MinimumOnly": "C1", "Snowball": "C2", "Avalanche": "C3"}
    linestyles = {"PPO": "-", "MinimumOnly": "--", "Snowball": "-.", "Avalanche": ":"}

    for strat in strategies_to_plot:
        env = CreditCardDebtEnv(config=scenario)
        if strat == "PPO":
            months, balances, _ = _run_episode_ppo(env, model, ep_seed)
        else:
            pol = baseline_map.get(strat)
            if not pol:
                continue
            months, balances, _ = _run_episode_baseline(env, pol, ep_seed)
        months = np.array(months)
        total_balance = np.array([sum(b) for b in balances])
        ax.plot(
            months,
            total_balance,
            color=colors.get(strat, "gray"),
            linestyle=linestyles.get(strat, "-"),
            linewidth=2,
            label=strat,
        )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total remaining balance ($)")
    ax.set_title("Balance trajectory (one representative scenario)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "balance_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"[Plot 3] Saved {output_dir / 'balance_trajectory.png'}")


# ---------- Plot 4: Payment allocation trajectory ----------


def plot_payment_allocation(
    model_path: Path,
    output_dir: Path,
    scenario_episode: int = 0,
    seed: int = DEFAULT_TEST_SEED,
) -> None:
    """Plot how each method allocates payments across cards over time (stacked or per-card)."""
    _ensure_mpl()
    if not model_path.exists():
        print(f"[Plot 4] Skipping: model {model_path} not found.")
        return

    from stable_baselines3 import PPO
    from src.envs.credit_env import CreditCardDebtEnv
    from src.envs.scenario_sampler import ScenarioSampler
    from src.baselines import ALL_BASELINES

    sampler = ScenarioSampler(num_cards_range=(NUM_CARDS_TEST, NUM_CARDS_TEST))
    rng = np.random.default_rng(seed)
    for _ in range(scenario_episode):
        sampler.sample(rng)
    scenario = sampler.sample(rng)
    ep_seed = seed + scenario_episode

    env0 = CreditCardDebtEnv(config=scenario)
    model = PPO.load(
        str(model_path),
        custom_objects={
            "observation_space": env0.observation_space,
            "action_space": env0.action_space,
        },
    )
    baseline_map = {p().name: p() for p in ALL_BASELINES}
    strategies_to_plot = [s for s in MAIN_BASELINES if s != "NormalizedAvalanche"]

    n_methods = len(strategies_to_plot)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, strat in enumerate(strategies_to_plot):
        if idx >= 4:
            break
        ax = axes[idx]
        env = CreditCardDebtEnv(config=scenario)
        if strat == "PPO":
            _, _, payments_log = _run_episode_ppo(env, model, ep_seed)
        else:
            pol = baseline_map.get(strat)
            if not pol:
                continue
            _, _, payments_log = _run_episode_baseline(env, pol, ep_seed)
        payments = np.array(payments_log)  # (T, n_cards)
        months = np.arange(1, payments.shape[0] + 1)
        n_cards = payments.shape[1]
        ax.stackplot(
            months,
            *[payments[:, i] for i in range(n_cards)],
            labels=[f"Card {i+1}" for i in range(n_cards)],
            alpha=0.8,
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Payment ($)")
        ax.set_title(strat)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Payment allocation across cards over time (one scenario)", fontsize=12, y=1.02)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "payment_allocation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot 4] Saved {output_dir / 'payment_allocation.png'}")


# ---------- Plot 5: Scenario breakdown / generalization ----------


def _classify_scenario(scenario: Any) -> dict[str, str]:
    """Label scenario by type: APR imbalance, high utilization, low budget."""
    from src.envs.financial_model import CardState, compute_interest, compute_min_payment
    cfg = scenario
    if not hasattr(cfg, "cards"):
        return {"apr_type": "unknown", "util_type": "unknown", "budget_type": "unknown"}
    cards = cfg.cards
    aprs = [c.apr for c in cards]
    balances = [c.balance for c in cards]
    limits = [c.credit_limit for c in cards]
    total_debt = sum(balances)
    total_limit = sum(limits) or 1
    initial_util = total_debt / total_limit
    total_min = 0.0
    for c in cards:
        st = CardState(
            name=c.name,
            apr=c.apr,
            balance=c.balance,
            credit_limit=c.credit_limit,
            min_payment_floor=c.min_payment_floor,
        )
        total_min += compute_min_payment(st, compute_interest(st))
    surplus = max(0.0, cfg.monthly_income - cfg.fixed_expenses - total_min)
    surplus_ratio = surplus / total_debt if total_debt > 0 else 1.0

    apr_range = max(aprs) - min(aprs) if len(aprs) > 1 else 0
    if apr_range >= 0.10:
        apr_type = "High APR spread"
    elif apr_range >= 0.05:
        apr_type = "Medium APR spread"
    else:
        apr_type = "Low APR spread"

    if initial_util >= 0.5:
        util_type = "High utilization"
    elif initial_util >= 0.3:
        util_type = "Medium utilization"
    else:
        util_type = "Low utilization"

    if surplus_ratio < 0.05:
        budget_type = "Tight budget"
    elif surplus_ratio < 0.15:
        budget_type = "Moderate budget"
    else:
        budget_type = "Comfortable budget"

    return {"apr_type": apr_type, "util_type": util_type, "budget_type": budget_type}


def plot_scenario_breakdown(
    per_episode_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_TEST_SEED,
) -> None:
    """Plot PPO performance across scenario types (APR spread, utilization, budget)."""
    _ensure_mpl()
    if not per_episode_path.exists():
        print(f"[Plot 5] Skipping: {per_episode_path} not found.")
        return

    from src.envs.scenario_sampler import ScenarioSampler

    df = pd.read_csv(per_episode_path)
    df["strategy"] = df["strategy"].str.strip()
    ppo = df[df["strategy"] == "PPO"].copy()

    sampler = ScenarioSampler(num_cards_range=(NUM_CARDS_TEST, NUM_CARDS_TEST))
    rng = np.random.default_rng(seed)
    episode_indices = ppo["episode"].values
    types_apr = []
    types_util = []
    types_budget = []
    for ep_idx in episode_indices:
        rng_ep = np.random.default_rng(seed + ep_idx)
        sc = sampler.sample(rng_ep)
        t = _classify_scenario(sc)
        types_apr.append(t["apr_type"])
        types_util.append(t["util_type"])
        types_budget.append(t["budget_type"])
    ppo["scenario_apr"] = types_apr
    ppo["scenario_util"] = types_util
    ppo["scenario_budget"] = types_budget

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, col, title in [
        (axes[0], "scenario_apr", "By APR spread"),
        (axes[1], "scenario_util", "By initial utilization"),
        (axes[2], "scenario_budget", "By payment budget"),
    ]:
        order = ppo[col].value_counts().index.tolist()
        means = ppo.groupby(col)["total_interest"].mean().reindex(order)
        stds = ppo.groupby(col)["total_interest"].std().reindex(order)
        counts = ppo.groupby(col).size().reindex(order)
        se = stds / np.sqrt(counts)
        x = np.arange(len(order))
        ax.bar(x, means, yerr=1.96 * se, capsize=4, color="steelblue", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Mean total interest ($)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("PPO generalization across scenario types (mean ± 95% CI)", fontsize=12, y=1.02)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "scenario_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot 5] Saved {output_dir / 'scenario_breakdown.png'}")


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser(description="Generate figures for report")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="results")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--model", type=str, default="models/best/best_model.zip")
    parser.add_argument("--seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument("--scenario-episode", type=int, default=0, help="Episode index for trajectory plots")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / args.results_dir
    figures_dir = root / args.figures_dir
    logdir = root / args.logdir
    model_path = root / args.model
    per_episode = results_dir / "ppo_vs_baselines_per_episode.csv"

    print("Plot 1: Training curve (return vs timesteps, mean + CI)")
    plot_training_curve(logdir, figures_dir)

    print("Plot 2: Method comparison (interest, months, util) with error bars")
    plot_method_comparison(per_episode, figures_dir)

    print("Plot 3: Balance trajectory (one scenario, PPO + baselines)")
    plot_balance_trajectory(model_path, figures_dir, scenario_episode=args.scenario_episode, seed=args.seed)

    print("Plot 4: Payment allocation trajectory")
    plot_payment_allocation(model_path, figures_dir, scenario_episode=args.scenario_episode, seed=args.seed)

    print("Plot 5: Scenario breakdown / generalization")
    plot_scenario_breakdown(per_episode, figures_dir, seed=args.seed)

    print("Done.")


if __name__ == "__main__":
    main()
