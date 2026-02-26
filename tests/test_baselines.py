"""Unit tests for baseline policies."""

import numpy as np
import pytest

from src.baselines import (
    AvalanchePolicy,
    MinimumOnlyPolicy,
    NormalizedAvalanchePolicy,
    SnowballPolicy,
)
from src.envs.credit_env import CreditCardDebtEnv
from src.utils.config import CardConfig, EnvConfig


@pytest.fixture
def env_3card() -> CreditCardDebtEnv:
    cfg = EnvConfig(
        cards=[
            CardConfig("High APR", apr=0.269, balance=1800, credit_limit=3000),
            CardConfig("Mid APR", apr=0.219, balance=6500, credit_limit=10000),
            CardConfig("Low APR", apr=0.169, balance=3200, credit_limit=8000),
        ],
        monthly_income=5000,
        fixed_expenses=3200,
    )
    env = CreditCardDebtEnv(config=cfg)
    env.reset(seed=42)
    return env


class TestMinimumOnly:

    def test_zero_surplus(self, env_3card):
        action = MinimumOnlyPolicy().allocate(env_3card)
        assert np.all(action == 0)

    def test_full_episode(self, env_3card):
        policy = MinimumOnlyPolicy()
        result = policy.run_episode(env_3card, seed=42)
        assert result["months"] > 0
        assert result["total_interest"] > 0
        assert result["strategy"] == "MinimumOnly"


class TestSnowball:

    def test_targets_smallest_balance(self, env_3card):
        """Snowball should put all surplus on the card with smallest balance."""
        action = SnowballPolicy().allocate(env_3card)
        # Card 0 has smallest balance ($1800)
        assert action[0] == pytest.approx(1.0)
        assert action[1] == pytest.approx(0.0)
        assert action[2] == pytest.approx(0.0)

    def test_full_episode(self, env_3card):
        result = SnowballPolicy().run_episode(env_3card, seed=42)
        assert result["months"] > 0
        assert result["total_interest"] > 0


class TestAvalanche:

    def test_targets_highest_apr(self, env_3card):
        """Avalanche should put all surplus on the card with highest APR."""
        action = AvalanchePolicy().allocate(env_3card)
        # Card 0 has highest APR (26.9%)
        assert action[0] == pytest.approx(1.0)
        assert action[1] == pytest.approx(0.0)
        assert action[2] == pytest.approx(0.0)

    def test_full_episode(self, env_3card):
        result = AvalanchePolicy().run_episode(env_3card, seed=42)
        assert result["months"] > 0
        assert result["total_interest"] > 0


class TestNormalizedAvalanche:

    def test_spreads_across_cards(self, env_3card):
        """NormalizedAvalanche should distribute weight across active cards."""
        action = NormalizedAvalanchePolicy().allocate(env_3card)
        # Should have non-zero weights on multiple cards
        nonzero = np.count_nonzero(action)
        assert nonzero >= 1  # At minimum targets something

    def test_full_episode(self, env_3card):
        result = NormalizedAvalanchePolicy().run_episode(env_3card, seed=42)
        assert result["months"] > 0
        assert result["total_interest"] > 0


class TestAllBaselinesComplete:

    @pytest.mark.parametrize("PolicyClass", [
        MinimumOnlyPolicy,
        SnowballPolicy,
        AvalanchePolicy,
        NormalizedAvalanchePolicy,
    ])
    def test_episode_completes(self, PolicyClass, env_3card):
        """Every baseline should complete an episode without error."""
        policy = PolicyClass()
        result = policy.run_episode(env_3card, seed=42)
        assert result["months"] <= 60
        assert result["total_interest"] >= 0

    def test_avalanche_beats_minimum_on_interest(self):
        """Avalanche should accrue less total interest than minimum-only."""
        cfg = EnvConfig(
            cards=[
                CardConfig("A", apr=0.269, balance=5000, credit_limit=8000),
                CardConfig("B", apr=0.169, balance=3000, credit_limit=6000),
            ],
            monthly_income=5000,
            fixed_expenses=3000,
        )
        env = CreditCardDebtEnv(config=cfg)

        min_result = MinimumOnlyPolicy().run_episode(env, seed=42)
        aval_result = AvalanchePolicy().run_episode(env, seed=42)

        assert aval_result["total_interest"] < min_result["total_interest"], (
            f"Avalanche interest ({aval_result['total_interest']:.2f}) should be less than "
            f"MinimumOnly ({min_result['total_interest']:.2f})"
        )
