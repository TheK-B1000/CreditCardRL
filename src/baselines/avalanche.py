"""Avalanche strategy — all surplus to highest APR card first."""

from __future__ import annotations

import numpy as np

from src.baselines.base_policy import BaselinePolicy
from src.envs.credit_env import CreditCardDebtEnv


class AvalanchePolicy(BaselinePolicy):
    """Debt avalanche: direct all surplus to the card with the highest APR.

    Mathematically optimal single-target strategy for minimizing total interest.
    """

    @property
    def name(self) -> str:
        return "Avalanche"

    def allocate(self, env: CreditCardDebtEnv) -> np.ndarray:
        if env.action_mode == "continuous":
            action = np.zeros(env.num_cards, dtype=np.float32)
        else:
            action = np.zeros(env.num_cards, dtype=np.int64)

        # Find the active card with the highest APR
        active_cards = [
            (i, c.apr) for i, c in enumerate(env.cards)
            if not c.is_paid_off and c.balance > 0
        ]
        if not active_cards:
            return action

        # Sort by APR descending, pick highest
        target_idx = max(active_cards, key=lambda x: x[1])[0]

        if env.action_mode == "continuous":
            action[target_idx] = 1.0
        else:
            from src.envs.financial_model import compute_interest, compute_min_payment
            total_min = sum(
                compute_min_payment(c, compute_interest(c)) for c in env.cards
            )
            surplus = max(0, env.monthly_income - env.fixed_expenses - total_min)
            chunks = int(surplus / env.discrete_chunk_size)
            action[target_idx] = chunks

        return action
