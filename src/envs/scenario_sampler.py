"""ScenarioSampler — generates diverse consumer profiles for training robustness.

Produces random EnvConfig instances with varied numbers of cards (1–5),
APRs (12%–29%), balances ($500–$15,000), incomes, and expenses.
Also provides named presets for reproducible benchmarking.
"""

from __future__ import annotations

import numpy as np

from src.envs.reward import RewardConfig
from src.utils.config import CardConfig, EnvConfig


# Card name pools for variety
_CARD_NAMES = [
    "Visa Platinum", "Mastercard Gold", "Store Card", "Rewards Card",
    "Travel Card", "Gas Card", "Medical Card", "Cash Back Card",
    "Student Card", "Department Store", "Airline Card", "Hotel Card",
]


class ScenarioSampler:
    """Generate randomized or preset consumer debt scenarios."""

    def __init__(
        self,
        num_cards_range: tuple[int, int] = (1, 5),
        apr_range: tuple[float, float] = (0.12, 0.29),
        balance_range: tuple[float, float] = (500.0, 15000.0),
        limit_multiplier_range: tuple[float, float] = (1.2, 3.0),
        income_range: tuple[float, float] = (3000.0, 8000.0),
        expense_ratio_range: tuple[float, float] = (0.50, 0.80),
        action_mode: str = "continuous",
        max_months: int = 60,
        utilization_target: float = 0.30,
        reward_config: RewardConfig | None = None,
    ):
        self.num_cards_range = num_cards_range
        self.apr_range = apr_range
        self.balance_range = balance_range
        self.limit_multiplier_range = limit_multiplier_range
        self.income_range = income_range
        self.expense_ratio_range = expense_ratio_range
        self.action_mode = action_mode
        self.max_months = max_months
        self.utilization_target = utilization_target
        self.reward_config = reward_config or RewardConfig()

    def sample(self, rng: np.random.Generator | None = None) -> EnvConfig:
        """Sample a random consumer profile.

        Args:
            rng: Numpy random Generator for reproducibility.

        Returns:
            A randomized EnvConfig.
        """
        if rng is None:
            rng = np.random.default_rng()

        num_cards = rng.integers(self.num_cards_range[0], self.num_cards_range[1] + 1)

        # Pick unique card names
        name_indices = rng.choice(len(_CARD_NAMES), size=num_cards, replace=False)
        names = [_CARD_NAMES[i] for i in name_indices]

        cards = []
        for name in names:
            apr = round(rng.uniform(*self.apr_range), 3)
            balance = round(rng.uniform(*self.balance_range), 2)
            limit_mult = rng.uniform(*self.limit_multiplier_range)
            credit_limit = round(balance * limit_mult, 2)
            cards.append(
                CardConfig(
                    name=name,
                    apr=apr,
                    balance=balance,
                    credit_limit=credit_limit,
                    min_payment_floor=25.0,
                )
            )

        income = round(rng.uniform(*self.income_range), 2)
        expense_ratio = rng.uniform(*self.expense_ratio_range)
        fixed_expenses = round(income * expense_ratio, 2)

        return EnvConfig(
            cards=cards,
            monthly_income=income,
            fixed_expenses=fixed_expenses,
            action_mode=self.action_mode,
            max_months=self.max_months,
            utilization_target=self.utilization_target,
            reward=self.reward_config,
        )

    @staticmethod
    def preset(name: str) -> EnvConfig:
        """Return a named preset scenario for reproducible experiments.

        Available presets:
            - "easy_3card": Low APRs, moderate balances, generous income
            - "hard_5card": High APRs, high balances, tight budget
            - "single_high_apr": One card with 28.9% APR

        Args:
            name: Preset name.

        Returns:
            EnvConfig for the named scenario.

        Raises:
            ValueError: If preset name is unknown.
        """
        presets = {
            "easy_3card": EnvConfig(
                cards=[
                    CardConfig("Visa Basic", apr=0.139, balance=2000, credit_limit=5000),
                    CardConfig("MC Standard", apr=0.159, balance=1500, credit_limit=4000),
                    CardConfig("Store Card", apr=0.199, balance=800, credit_limit=2000),
                ],
                monthly_income=6000,
                fixed_expenses=3000,
            ),
            "hard_5card": EnvConfig(
                cards=[
                    CardConfig("Platinum", apr=0.289, balance=12000, credit_limit=15000),
                    CardConfig("Medical", apr=0.249, balance=8500, credit_limit=10000),
                    CardConfig("Rewards", apr=0.199, balance=5000, credit_limit=12000),
                    CardConfig("Dept Store", apr=0.269, balance=3000, credit_limit=4000),
                    CardConfig("Gas Card", apr=0.229, balance=1500, credit_limit=2500),
                ],
                monthly_income=5500,
                fixed_expenses=3800,
            ),
            "single_high_apr": EnvConfig(
                cards=[
                    CardConfig("High APR Card", apr=0.289, balance=10000, credit_limit=12000),
                ],
                monthly_income=4500,
                fixed_expenses=2800,
            ),
        }

        if name not in presets:
            valid = ", ".join(sorted(presets.keys()))
            raise ValueError(f"Unknown preset {name!r}. Valid: {valid}")

        return presets[name]
