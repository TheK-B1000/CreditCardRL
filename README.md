# CreditCardRL: RL for Optimal Credit Card Debt Repayment

Train a reinforcement learning agent (PPO) to allocate monthly payments across multiple credit cards, minimizing total interest and time-to-payoff while keeping utilization in a healthy range. The project includes a Gymnasium environment, baseline heuristics (Avalanche, Snowball, Minimum-Only, Normalized Avalanche), and evaluation on in-distribution and held-out test sets.

## Setup

**Requirements:** Python 3.10+ (recommended), NumPy 2.x, PyTorch 2.3+, Stable-Baselines3, Gymnasium.

```bash
# From project root
cd CreditCardRL
pip install -r requirements.txt
```

Optional (for development and tests):

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Quick start: run evaluation

After setup, you can run the main evaluation without training (baselines only if no model is present):

```bash
pip install -r requirements.txt && python run_eval.py
```

- **With a trained model:** Evaluates PPO and all baselines on a fixed test set (50 episodes by default) and writes `results/ppo_vs_baselines_per_episode.csv` and `results/ppo_vs_baselines_summary.csv`.
- **Without a model:** Runs with `--no-ppo` so the command still succeeds; train first for full PPO + baselines (see below).

---

## Reproducing all results

Run from the **project root** (`CreditCardRL/`). Order matters where later steps use outputs from earlier ones.

### 1. Train PPO (optional)

Train a PPO agent on the 3-card environment. TensorBoard logs go to `runs/`; the best model is saved under `models/best/`.

```bash
python scripts/train_ppo.py
# Or with custom config / short run:
python scripts/train_ppo.py --config configs/train/ppo_default.yaml --timesteps 5000
```

### 2. Main evaluation (fixed test set)

Evaluates PPO (if model exists) and baselines on a fixed 1,000-episode test set (seed 999, 3-card scenarios).

```bash
python scripts/evaluate_ppo_vs_baselines.py --episodes 1000 --output results
```

Produces:

- `results/ppo_vs_baselines_per_episode.csv`
- `results/ppo_vs_baselines_summary.csv`

To run baselines only (e.g. no model): add `--no-ppo`.

### 3. Report assets (comparison table, training curve, balance trajectory)

```bash
python scripts/generate_report.py --results-dir results --output-dir results --logdir runs
```

Optional: `--model models/best/best_model.zip` for balance trajectory. Produces:

- `results/comparison_table.md`, `results/comparison_table.csv`
- `results/training_curve.png` (if TensorBoard logs exist in `runs/`)
- Balance trajectory plot (if model and matplotlib available)

### 4. Publication-style figures

Requires the per-episode and summary CSVs from step 2; optional TensorBoard `runs/` and PPO model for training curve and trajectory plots.

```bash
python scripts/plot_figures.py --results-dir results --figures-dir results --logdir runs
```

Produces in `results/`:

- `training_curve.png` — PPO return vs timesteps (mean ± CI if multiple runs)
- `method_comparison.png` — Bar chart of total interest by strategy with error bars
- `balance_trajectory.png` — Total balance over time for PPO and main baselines
- `payment_allocation.png` — Stacked payment allocation over months (PPO)
- `scenario_breakdown.png` — Performance by APR / utilization / budget buckets

### 5. Held-out evaluation (generalization)

Runs PPO (for 3-card presets) and baselines on held-out scenario distributions (different APR/income/card count). Config: `configs/eval/held_out_test.yaml`.

```bash
python scripts/evaluate_held_out.py --config configs/eval/held_out_test.yaml --output results
```

To run a single preset (e.g. `extended_ranges`):

```bash
python scripts/evaluate_held_out.py --preset extended_ranges --episodes 500 --output results
```

Produces per preset: `results/held_out_summary_<preset>.csv`, `results/held_out_per_episode_<preset>.csv`.

### 6. Performance gap report (in-dist vs held-out)

Expects in-distribution and held-out summary CSVs (e.g. from steps 2 and 5). Computes gap (held-out − in-dist) for PPO and writes a short report.

```bash
python scripts/report_performance_gap.py
```

Produces: `results/performance_gap_report.csv`, `results/performance_gap_summary.csv`, `results/performance_gap_report.md`.

### 7. PPO vs Normalized Avalanche comparison

Compares PPO to Normalized Avalanche on in-distribution and all held-out presets; writes table and interest figure.

```bash
python scripts/compare_ppo_vs_normalized_avalanche.py --output results
```

Produces: `results/ppo_vs_normalized_avalanche_comparison.csv`, `results/ppo_vs_normalized_avalanche_comparison.md`, `results/ppo_vs_normalized_avalanche_interest.png`.

### 8. Edge-case tests

Runs single-card, same-APR, 0% intro card, and income-shock scenarios; PPO only for 3-card cases.

```bash
python scripts/run_edge_case_tests.py --output results
```

Produces: `results/edge_case_results.csv`, `results/edge_case_summary.csv`.

---

## Figure descriptions

| Figure | Description |
|--------|-------------|
| **training_curve.png** | PPO mean episode return vs training timesteps. If multiple TensorBoard runs exist, shows mean line with shaded confidence interval. |
| **method_comparison.png** | Bar chart of total interest paid by strategy (PPO and baselines) on the fixed test set, with error bars (e.g. std or CI). |
| **balance_trajectory.png** | Total debt over months for one scenario: PPO and main baselines (e.g. MinimumOnly, Snowball, Avalanche). Shows how quickly each strategy pays down balance. |
| **payment_allocation.png** | Stacked area or bar chart of PPO’s monthly payment allocation across cards over time for a single scenario. |
| **scenario_breakdown.png** | Performance (e.g. total interest) broken down by scenario groups: APR range, utilization level, or budget tier. |
| **ppo_vs_normalized_avalanche_interest.png** | Comparison of total interest (or related metric) between PPO and Normalized Avalanche across test sets (in-dist and held-out presets). |

---

## Project structure

```
CreditCardRL/
├── configs/
│   ├── env/              # Environment configs (e.g. default_3card.yaml)
│   ├── train/            # PPO/SAC/DQN training configs
│   └── eval/             # Evaluation and held-out test configs
├── src/
│   ├── envs/             # CreditCardDebtEnv, financial model, scenario sampler
│   ├── baselines/        # MinimumOnly, Snowball, Avalanche, NormalizedAvalanche
│   ├── agents/           # Training callbacks
│   ├── evaluation/       # Metrics (e.g. credit proxy score)
│   └── utils/            # Config loading
├── scripts/              # Training, evaluation, reporting, figures
├── results/              # CSVs, tables, and figures (generated)
├── models/               # Saved PPO model (e.g. models/best/best_model.zip)
├── runs/                 # TensorBoard logs (git-ignored)
├── tests/                # pytest suite
├── run_eval.py           # Entry point: pip install -r requirements.txt && python run_eval.py
└── requirements.txt
```

## Environment and baselines

- **CreditCardDebtEnv:** Gymnasium env with observation (normalized per-card + global features), continuous action (payment proportions), and reward combining interest, utilization, and payoff terms.
- **Baselines:** MinimumOnly, Snowball (smallest balance first), Avalanche (highest APR first), NormalizedAvalanche (APR×balance with utilization cap).

## License and citation

See repository for license. If you use this code, please cite the project or paper as appropriate.
