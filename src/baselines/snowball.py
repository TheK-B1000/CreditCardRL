"""Snowball strategy — all surplus to smallest balance first."""

from __future__ import annotations

import numpy as np

from src.baselines.base_policy import BaselinePolicy
from src.envs.credit_env import CreditCardDebtEnv


class SnowballPolicy(BaselinePolicy):
    """Debt snowball: direct all surplus to the card with the smallest balance.

    Psychologically motivated strategy — quick wins by eliminating small debts.
    """

    @property
    def name(self) -> str:
        return "Snowball"

    def allocate(self, env: CreditCardDebtEnv) -> np.ndarray:
        if env.action_mode == "continuous":
            action = np.zeros(env.num_cards, dtype=np.float32)
        else:
            action = np.zeros(env.num_cards, dtype=np.int64)

        # Find the active card with the smallest balance
        active_cards = [
            (i, c.balance) for i, c in enumerate(env.cards)
            if not c.is_paid_off and c.balance > 0
        ]
        if not active_cards:
            return action

        # Sort by balance ascending, pick smallest
        target_idx = min(active_cards, key=lambda x: x[1])[0]

        if env.action_mode == "continuous":
            action[target_idx] = 1.0  # All surplus to this card
        else:
            # Compute available surplus in chunks
            from src.envs.financial_model import compute_interest, compute_min_payment
            total_min = sum(
                compute_min_payment(c, compute_interest(c)) for c in env.cards
            )
            surplus = max(0, env.monthly_income - env.fixed_expenses - total_min)
            chunks = int(surplus / env.discrete_chunk_size)
            action[target_idx] = chunks

        return action
