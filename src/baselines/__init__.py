"""Baseline policies for credit card debt repayment."""

from src.baselines.base_policy import BaselinePolicy
from src.baselines.minimum_only import MinimumOnlyPolicy
from src.baselines.snowball import SnowballPolicy
from src.baselines.avalanche import AvalanchePolicy
from src.baselines.normalized_avalanche import NormalizedAvalanchePolicy

ALL_BASELINES = [
    MinimumOnlyPolicy,
    SnowballPolicy,
    AvalanchePolicy,
    NormalizedAvalanchePolicy,
]

__all__ = [
    "BaselinePolicy",
    "MinimumOnlyPolicy",
    "SnowballPolicy",
    "AvalanchePolicy",
    "NormalizedAvalanchePolicy",
    "ALL_BASELINES",
]
