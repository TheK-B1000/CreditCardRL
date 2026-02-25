"""YAML configuration loader and dataclasses for environment setup."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.envs.reward import RewardConfig


def _resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path, anchoring relative paths to the project root.

    The project root is identified as the nearest ancestor directory that
    contains ``pyproject.toml``.  If the file exists as-is (e.g. an absolute
    path or the CWD happens to be the project root already), it is returned
    unchanged.
    """
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p

    # Walk up from this file's location to find the project root
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            candidate = parent / p
            if candidate.exists():
                return candidate
            # Return it anyway — open() will give the descriptive FileNotFoundError
            return candidate

    return p  # Fallback: return as-is


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
    path = _resolve_config_path(path)
    with open(path, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # Build card configs
    card_dicts = raw.get("cards", [])
    cards = [
        CardConfig(
            name=str(cd.get("name", "Card")),
            apr=float(cd.get("apr", 0.199)),
            balance=float(cd.get("balance", 5000.0)),
            credit_limit=float(cd.get("credit_limit", 10000.0)),
            min_payment_floor=float(cd.get("min_payment_floor", 25.0)),
        )
        for cd in card_dicts
    ]

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
    path = _resolve_config_path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_eval_config(path: str | Path) -> dict[str, Any]:
    """Load evaluation protocol from a YAML file."""
    path = _resolve_config_path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)
