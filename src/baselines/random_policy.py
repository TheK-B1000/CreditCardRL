"""Random policy — uniformly random allocations for lower-bound reference."""

from __future__ import annotations

import numpy as np

from src.baselines.base_policy import BaselinePolicy
from src.envs.credit_env import CreditCardDebtEnv


class RandomPolicy(BaselinePolicy):
    """Allocate surplus randomly across active cards."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "Random"

    def allocate(self, env: CreditCardDebtEnv) -> np.ndarray:
        n = env.num_cards

        if env.action_mode == "continuous":
            action = self._rng.random(n).astype(np.float32)
            # Mask paid-off cards
            for i, c in enumerate(env.cards):
                if c.is_paid_off:
                    action[i] = 0.0
            return action
        else:
            from src.envs.financial_model import compute_interest, compute_min_payment
            total_min = sum(
                compute_min_payment(c, compute_interest(c)) for c in env.cards
            )
            surplus = max(0, env.monthly_income - env.fixed_expenses - total_min)
            total_chunks = int(surplus / env.discrete_chunk_size)

            if total_chunks <= 0:
                return np.zeros(n, dtype=np.int64)

            # Randomly distribute chunks
            action = np.zeros(n, dtype=np.int64)
            active = [i for i, c in enumerate(env.cards) if not c.is_paid_off]
            if active:
                for _ in range(total_chunks):
                    idx = self._rng.choice(active)
                    action[idx] += 1
            return action
