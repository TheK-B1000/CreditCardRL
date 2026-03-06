"""Generate report assets: comparison table, training curve, and balance trajectory.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --results-dir results --output-dir results --logdir runs
    python scripts/generate_report.py --model models/best/best_model.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# Step 1: comparison table (no extra deps)
# Step 2: training curve (tensorboard)
# Step 3: balance trajectory (matplotlib, env, PPO)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

DEFAULT_TEST_SEED = 999
NUM_CARDS_TEST = 3


def step1_comparison_table(
    summary_path: Path,
    output_dir: Path,
) -> None:
    """Generate comparison table (mean ± std) and save as Markdown and CSV."""
    if not summary_path.exists():
        print(f"[Step 1] Skipping: {summary_path} not found. Run evaluate_ppo_vs_baselines.py first.")
        return

    df = pd.read_csv(summary_path)
    df = df.rename(columns={
        "interest_mean": "Interest ($) mean",
        "interest_std": "Interest ($) std",
        "months_mean": "Months mean",
        "months_std": "Months std",
        "util_mean": "Utilization mean",
        "paid_off_pct": "Paid off (%)",
        "credit_score_mean": "Credit score mean",
        "n_episodes": "N episodes",
    })

    # Formatted table for display (mean ± std for interest and months)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Strategy": r["strategy"],
            "Interest ($)": f"{r['Interest ($) mean']:,.0f} ± {r['Interest ($) std']:,.0f}",
            "Months": f"{r['Months mean']:.2f} ± {r['Months std']:.2f}",
            "Utilization": f"{r['Utilization mean']:.4f}",
            "Paid off (%)": f"{r['Paid off (%)']:.1f}",
            "Credit score": f"{r['Credit score mean']:.1f}",
            "N episodes": int(r["N episodes"]),
        })
    table_df = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_md = output_dir / "comparison_table.md"
    out_csv = output_dir / "comparison_table.csv"

    def to_md(df: pd.DataFrame) -> str:
        lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        return "\n".join(lines)

    with open(out_md, "w") as f:
        f.write("# PPO vs Baselines — Fixed test set\n\n")
        f.write(to_md(table_df))
        f.write("\n")

    table_df.to_csv(out_csv, index=False)
    print(f"[Step 1] Comparison table saved to {out_md} and {out_csv}")
    print(table_df.to_string(index=False))
    print()


def step2_training_curve(logdir: Path, output_dir: Path) -> None:
    """Plot training curve: episode return vs timesteps from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[Step 2] Skipping: tensorboard not installed. pip install tensorboard")
        return

    if not logdir.exists():
        print(f"[Step 2] Skipping: logdir {logdir} not found. Train with train_ppo.py first.")
        return

    # Find latest run with PPO events (SB3 writes rollout/ep_rew_mean, time/total_timesteps)
    event_dirs = []
    for d in logdir.iterdir():
        if d.is_dir():
            for e in d.rglob("events.out.tfevents.*"):
                event_dirs.append(e.parent)
                break

    if not event_dirs:
        # Maybe events are directly in logdir
        for e in logdir.rglob("events.out.tfevents.*"):
            event_dirs.append(e.parent)
            break

    if not event_dirs:
        print(f"[Step 2] Skipping: no TensorBoard events under {logdir}")
        return

    if not _HAS_MPL:
        print("[Step 2] Skipping: matplotlib not installed")
        return

    # Use the first (or we could merge multiple runs)
    event_dir = event_dirs[0]
    acc = EventAccumulator(str(event_dir))
    acc.Reload()

    # Prefer rollout/ep_rew_mean (SB3 VecMonitor); fallback to eval if present
    tags = acc.Tags().get("scalars", [])
    rew_tag = "rollout/ep_rew_mean" if "rollout/ep_rew_mean" in tags else None
    if not rew_tag and "eval/mean_reward" in tags:
        rew_tag = "eval/mean_reward"
    if not rew_tag:
        for t in tags:
            if "rew" in t.lower() or "reward" in t.lower():
                rew_tag = t
                break
    if not rew_tag:
        print(f"[Step 2] No reward scalar found. Available: {tags[:20]}")
        return

    steps_tag = "time/total_timesteps" if "time/total_timesteps" in tags else None
    if not steps_tag:
        for t in tags:
            if "timestep" in t.lower() or "step" in t.lower():
                steps_tag = t
                break

    events = acc.Scalars(rew_tag)
    if not events:
        print(f"[Step 2] No data for {rew_tag}")
        return

    steps = np.array([e.step for e in events])
    # If we have time/total_timesteps, we need to align; SB3 often logs rew at same step as step counter
    if steps_tag and steps_tag != rew_tag:
        step_events = acc.Scalars(steps_tag)
        if step_events and len(step_events) == len(events):
            steps = np.array([e.value for e in step_events])
    values = np.array([e.value for e in events])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(steps, values, color="C0", linewidth=1.5, label="Mean episode return")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Return (mean)")
    ax.set_title("PPO training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "training_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Step 2] Training curve saved to {out_path}")
    print()


def step3_balance_trajectory(
    model_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_TEST_SEED,
) -> None:
    """Run one episode with PPO, record balances each month, plot trajectory."""
    if not _HAS_MPL:
        print("[Step 3] Skipping: matplotlib not installed")
        return

    if not model_path.exists():
        print(f"[Step 3] Skipping: model {model_path} not found")
        return

    from stable_baselines3 import PPO
    from src.envs.credit_env import CreditCardDebtEnv
    from src.envs.scenario_sampler import ScenarioSampler

    sampler = ScenarioSampler(num_cards_range=(NUM_CARDS_TEST, NUM_CARDS_TEST))
    rng = np.random.default_rng(seed)
    scenario = sampler.sample(rng)
    env = CreditCardDebtEnv(config=scenario)

    model = PPO.load(
        str(model_path),
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
    )

    obs, _ = env.reset(seed=seed)
    months = [0]
    balances = [[c.balance for c in env.cards]]
    card_names = [c.name for c in env.cards]
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        months.append(env.month)
        balances.append([c.balance for c in env.cards])

    months = np.array(months)
    balances = np.array(balances)  # (T+1, n_cards)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for i in range(balances.shape[1]):
        ax.plot(months, balances[:, i], marker="o", markersize=3, label=card_names[i], linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Balance ($)")
    ax.set_title("Example balance trajectory (PPO, one scenario)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "balance_trajectory.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Step 3] Balance trajectory saved to {out_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate report: table, training curve, trajectory")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory with ppo_vs_baselines_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Where to write comparison_table.*, training_curve.png, balance_trajectory.png",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs",
        help="TensorBoard log directory for training curve",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best/best_model.zip",
        help="PPO model for balance trajectory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TEST_SEED,
        help="RNG seed for balance trajectory scenario",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / args.results_dir
    output_dir = root / args.output_dir
    logdir = root / args.logdir
    model_path = root / args.model

    summary_path = results_dir / "ppo_vs_baselines_summary.csv"

    print("Step 1 — Comparison table (mean ± std)")
    step1_comparison_table(summary_path, output_dir)

    print("Step 2 — Training curve (return vs timesteps)")
    step2_training_curve(logdir, output_dir)

    print("Step 3 — Example balance trajectory (one scenario)")
    step3_balance_trajectory(model_path, output_dir, seed=args.seed)

    print("Done.")


if __name__ == "__main__":
    main()
