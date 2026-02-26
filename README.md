# Credit-RL: RL for Optimal Credit Card Debt Repayment

> Train a reinforcement learning agent to optimally allocate monthly payments across multiple credit cards, minimizing total interest paid and time-to-payoff while maintaining healthy credit utilization.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Sanity check — visual walk-through of 3 episodes
python scripts/sanity_check.py

# Run all baselines (100 episodes)
python scripts/run_baselines.py --episodes 100
```

## Project Structure

```
credit-rl/
├── configs/           # YAML configs for env, training, evaluation
│   ├── env/           # Environment scenarios (3-card, 5-card, single)
│   ├── train/         # Algorithm hyperparameters (PPO, SAC, DQN)
│   └── eval/          # Evaluation protocol (seeds, metrics)
├── src/
│   ├── envs/          # Gymnasium environment + financial model
│   ├── baselines/     # Heuristic strategies (MinOnly, Snowball, Avalanche, etc.)
│   ├── agents/        # RL training scripts (Phase 1)
│   ├── evaluation/    # Metrics, statistical tests (Phase 2)
│   └── utils/         # Config loading, logging, visualization
├── scripts/           # CLI entry points
├── tests/             # pytest test suite
├── runs/              # Training artifacts (git-ignored)
├── results/           # Aggregated results & figures
└── paper/             # LaTeX source for report
```

## Environment

**`CreditCardDebtEnv`** — Gymnasium-compliant env simulating monthly debt repayment.

- **Observation**: Normalized vector of per-card features (balance, limit, APR, min payment, utilization) + global features (surplus, month, total debt, avg APR)
- **Action**: Continuous (softmax proportions) or discrete ($50 chunks) allocation of surplus budget
- **Reward**: Interest penalty + utilization penalty + missed-minimum penalty + payoff bonus + terminal bonus

## Baselines

| Strategy | Logic |
|----------|-------|
| MinimumOnly | Pay only minimums — worst case |
| Snowball | All surplus → smallest balance |
| Avalanche | All surplus → highest APR |
| NormalizedAvalanche | Weighted by APR×balance with utilization ceiling |

## Algorithms (Phase 1)

- **PPO** — Primary agent (continuous actions)
- **SAC** — Continuous-action alternative
- **DQN** — Discrete fallback

## Evaluation (Phase 2)

1,000-episode rollouts × 5 seeds. Metrics: total interest, months-to-payoff, avg utilization, credit health score. Report mean ± std and 95% CI.
