"""Minimum-payment-only baseline — worst-case reference strategy."""

from __future__ import annotations

import numpy as np

from src.baselines.base_policy import BaselinePolicy
from src.envs.credit_env import CreditCardDebtEnv


class MinimumOnlyPolicy(BaselinePolicy):
    """Pay only the minimum on each card, never allocate surplus."""

    @property
    def name(self) -> str:
        return "MinimumOnly"

    def allocate(self, env: CreditCardDebtEnv) -> np.ndarray:
        # Zero surplus allocation — env already handles minimums
        if env.action_mode == "continuous":
            return np.zeros(env.num_cards, dtype=np.float32)
        else:
            return np.zeros(env.num_cards, dtype=np.int64)
