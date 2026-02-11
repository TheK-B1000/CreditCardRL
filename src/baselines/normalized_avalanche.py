"""Normalized Avalanche with utilization ceiling — the "smart default" baseline.

Spreads surplus proportionally by (APR × utilization) while enforcing a 30%
utilization ceiling: if any card is above the target, extra budget goes there
to bring it down before optimizing for interest.
"""

from __future__ import annotations

import numpy as np

from src.baselines.base_policy import BaselinePolicy
from src.envs.credit_env import CreditCardDebtEnv


class NormalizedAvalanchePolicy(BaselinePolicy):
    """Hybrid strategy: interest-aware allocation with utilization rebalancing.

    Algorithm:
        1. Identify cards over the utilization target (default 30%).
        2. If any card is over target, allocate proportional to the excess
           utilization to bring them under the ceiling.
        3. Remaining surplus (if any) is allocated proportional to APR × balance
           across all active cards.
    """

    def __init__(self, utilization_target: float = 0.30):
        self.utilization_target = utilization_target

    @property
    def name(self) -> str:
        return "NormalizedAvalanche"

    def allocate(self, env: CreditCardDebtEnv) -> np.ndarray:
        n = env.num_cards
        weights = np.zeros(n, dtype=np.float64)

        active_indices = [
            i for i, c in enumerate(env.cards)
            if not c.is_paid_off and c.balance > 0
        ]

        if not active_indices:
            if env.action_mode == "continuous":
                return np.zeros(n, dtype=np.float32)
            else:
                return np.zeros(n, dtype=np.int64)

        # Step 1: Check for over-utilized cards
        over_util = {}
        for i in active_indices:
            card = env.cards[i]
            if card.utilization > self.utilization_target:
                excess = card.utilization - self.utilization_target
                over_util[i] = excess

        if over_util:
            # Allocate proportional to excess utilization
            total_excess = sum(over_util.values())
            for i, excess in over_util.items():
                weights[i] = excess / total_excess if total_excess > 0 else 0
        else:
            # Step 2: Allocate proportional to APR × balance (interest-weighted)
            for i in active_indices:
                card = env.cards[i]
                weights[i] = card.apr * card.balance

            total_w = weights.sum()
            if total_w > 0:
                weights /= total_w

        if env.action_mode == "continuous":
            return weights.astype(np.float32)
        else:
            from src.envs.financial_model import compute_interest, compute_min_payment
            total_min = sum(
                compute_min_payment(c, compute_interest(c)) for c in env.cards
            )
            surplus = max(0, env.monthly_income - env.fixed_expenses - total_min)
            total_chunks = int(surplus / env.discrete_chunk_size)
            chunks = (weights * total_chunks).astype(np.int64)
            return chunks
