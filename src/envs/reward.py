"""Reward function for the credit card debt environment.

Configurable via RewardConfig dataclass. Computes per-step and terminal rewards.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.envs.financial_model import CardState, compute_overall_utilization


@dataclass
class RewardConfig:
    """Reward shaping coefficients. Defaults from the project brief."""

    alpha: float = 1.0      # Interest penalty weight
    beta: float = 0.3       # Utilization penalty weight
    gamma_: float = 5.0     # Missed minimum penalty (per card)
    delta: float = 0.5      # Per-card payoff bonus
    epsilon: float = 10.0   # All-debt-cleared terminal bonus
    zeta: float = 1.0       # Time pressure terminal penalty

    utilization_target: float = 0.30  # Target utilization threshold


def compute_step_reward(
    cfg: RewardConfig,
    interest_accrued: float,
    initial_total_debt: float,
    cards: list[CardState],
    missed_minimum_count: int,
    cards_paid_off_this_step: int,
) -> float:
    """Compute the per-step (monthly) reward.

    reward_monthly = (
        - α × (interest_this_month / initial_total_debt)
        - β × max(0, avg_utilization − utilization_target)
        - γ × missed_minimum_count
        + δ × cards_paid_off_this_step
    )

    Args:
        cfg: Reward coefficients.
        interest_accrued: Total interest accrued across all cards this month.
        initial_total_debt: Total debt at episode start (for normalization).
        cards: Current card states (for utilization calc).
        missed_minimum_count: Number of cards where payment < minimum this step.
        cards_paid_off_this_step: Number of cards newly zeroed this step.

    Returns:
        Scalar reward for this time step.
    """
    # Normalize interest penalty by initial debt to keep reward scale consistent
    if initial_total_debt > 0:
        interest_penalty = cfg.alpha * (interest_accrued / initial_total_debt)
    else:
        interest_penalty = 0.0

    # Utilization penalty (only kicks in above target)
    avg_util = compute_overall_utilization(cards)
    util_penalty = cfg.beta * max(0.0, avg_util - cfg.utilization_target)

    # Missed minimum hard penalty
    missed_penalty = cfg.gamma_ * missed_minimum_count

    # Payoff bonus
    payoff_bonus = cfg.delta * cards_paid_off_this_step

    return -interest_penalty - util_penalty - missed_penalty + payoff_bonus


def compute_terminal_reward(
    cfg: RewardConfig,
    all_debt_paid: bool,
    months_elapsed: int,
    max_months: int,
) -> float:
    """Compute the terminal (end-of-episode) reward.

    reward_terminal = (
        + ε × (1.0 if all_debt_paid else 0.0)
        - ζ × (months_elapsed / max_months)
    )

    Args:
        cfg: Reward coefficients.
        all_debt_paid: True if every card balance is 0.
        months_elapsed: How many months the episode lasted.
        max_months: Maximum allowed months (horizon).

    Returns:
        Terminal reward bonus/penalty.
    """
    completion_bonus = cfg.epsilon * (1.0 if all_debt_paid else 0.0)
    time_penalty = cfg.zeta * (months_elapsed / max_months)
    return completion_bonus - time_penalty
