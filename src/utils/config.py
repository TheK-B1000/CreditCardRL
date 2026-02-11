"""YAML configuration loader and dataclasses for environment setup."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.envs.reward import RewardConfig


@dataclass
class CardConfig:
    """Configuration for a single credit card."""

    name: str = "Card"
    apr: float = 0.199
    balance: float = 5000.0
    credit_limit: float = 10000.0
    min_payment_floor: float = 25.0


@dataclass
class EnvConfig:
    """Full environment configuration."""

    cards: list[CardConfig] = field(default_factory=lambda: [CardConfig()])
    monthly_income: float = 5000.0
    fixed_expenses: float = 3200.0
    action_mode: str = "continuous"  # "continuous" or "discrete"
    max_months: int = 60
    utilization_target: float = 0.30
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def num_cards(self) -> int:
        return len(self.cards)

    @property
    def total_initial_debt(self) -> float:
        return sum(c.balance for c in self.cards)

    @property
    def total_credit_limit(self) -> float:
        return sum(c.credit_limit for c in self.cards)


def load_env_config(path: str | Path) -> EnvConfig:
    """Load an EnvConfig from a YAML file.

    Args:
        path: Path to a YAML config file (e.g., configs/env/default_3card.yaml).

    Returns:
        Populated EnvConfig instance.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # Build card configs
    card_dicts = raw.get("cards", [])
    cards = [CardConfig(**cd) for cd in card_dicts]

    # Build reward config
    reward_dict = raw.get("reward", {})
    # Handle the gamma_ naming (YAML uses 'gamma', Python uses 'gamma_' to avoid builtin clash)
    if "gamma" in reward_dict:
        reward_dict["gamma_"] = reward_dict.pop("gamma")
    reward_cfg = RewardConfig(**reward_dict)

    return EnvConfig(
        cards=cards,
        monthly_income=raw.get("monthly_income", 5000.0),
        fixed_expenses=raw.get("fixed_expenses", 3200.0),
        action_mode=raw.get("action_mode", "continuous"),
        max_months=raw.get("max_months", 60),
        utilization_target=raw.get("utilization_target", 0.30),
        reward=reward_cfg,
    )


def load_train_config(path: str | Path) -> dict[str, Any]:
    """Load training hyperparameters from a YAML file.

    Returns a plain dict since training configs vary by algorithm.
    """
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_eval_config(path: str | Path) -> dict[str, Any]:
    """Load evaluation protocol from a YAML file."""
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)
