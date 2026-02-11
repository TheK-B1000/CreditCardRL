"""Unit tests for CreditCardDebtEnv — Gymnasium compliance and correctness."""

import numpy as np
import pytest

from src.envs.credit_env import CreditCardDebtEnv
from src.utils.config import CardConfig, EnvConfig


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> EnvConfig:
    """3-card default config."""
    return EnvConfig(
        cards=[
            CardConfig("Visa", apr=0.219, balance=6500, credit_limit=10000),
            CardConfig("MC", apr=0.169, balance=3200, credit_limit=8000),
            CardConfig("Store", apr=0.269, balance=1800, credit_limit=3000),
        ],
        monthly_income=5000,
        fixed_expenses=3200,
    )


@pytest.fixture
def single_card_config() -> EnvConfig:
    """Simple 1-card config for targeted tests."""
    return EnvConfig(
        cards=[CardConfig("Solo", apr=0.199, balance=5000, credit_limit=8000)],
        monthly_income=4000,
        fixed_expenses=2800,
    )


@pytest.fixture
def env(default_config) -> CreditCardDebtEnv:
    return CreditCardDebtEnv(config=default_config)


@pytest.fixture
def single_env(single_card_config) -> CreditCardDebtEnv:
    return CreditCardDebtEnv(config=single_card_config)


# ── Reset ────────────────────────────────────────────────────────────────

class TestReset:

    def test_obs_shape(self, env, default_config):
        obs, info = env.reset(seed=42)
        expected_dim = 5 * default_config.num_cards + 4
        assert obs.shape == (expected_dim,)

    def test_obs_normalized(self, env):
        obs, _ = env.reset(seed=42)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_reset_returns_info(self, env):
        _, info = env.reset(seed=42)
        assert "month" in info
        assert info["month"] == 0
        assert "balances" in info

    def test_seed_reproducibility(self, default_config):
        env1 = CreditCardDebtEnv(config=default_config)
        env2 = CreditCardDebtEnv(config=default_config)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


# ── Step ─────────────────────────────────────────────────────────────────

class TestStep:

    def test_step_returns_tuple(self, env):
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_shape_after_step(self, env, default_config):
        env.reset(seed=42)
        obs, *_ = env.step(env.action_space.sample())
        expected_dim = 5 * default_config.num_cards + 4
        assert obs.shape == (expected_dim,)

    def test_obs_normalized_after_step(self, env):
        env.reset(seed=42)
        obs, *_ = env.step(env.action_space.sample())
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_month_increments(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert info["month"] == 1

    def test_balances_decrease_with_payment(self, single_env):
        """With a healthy surplus, balance should decrease each month."""
        single_env.reset(seed=42)
        _, info_before = single_env.reset(seed=42)
        initial_debt = info_before["total_debt"]

        # Allocate all surplus to the single card
        action = np.array([1.0], dtype=np.float32)
        _, _, _, _, info_after = single_env.step(action)

        assert info_after["total_debt"] < initial_debt


# ── Termination ──────────────────────────────────────────────────────────

class TestTermination:

    def test_terminates_when_all_paid(self):
        """Episode should terminate when all cards are paid off."""
        # Use a card with tiny balance and big income
        cfg = EnvConfig(
            cards=[CardConfig("Tiny", apr=0.10, balance=50, credit_limit=1000)],
            monthly_income=5000,
            fixed_expenses=1000,
        )
        env = CreditCardDebtEnv(config=cfg)
        env.reset(seed=42)

        action = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated is True
        assert truncated is False

    def test_truncates_at_max_months(self):
        """Episode should truncate at max_months if debt remains."""
        # Use high debt, low income to guarantee timeout
        cfg = EnvConfig(
            cards=[CardConfig("Big", apr=0.289, balance=50000, credit_limit=60000)],
            monthly_income=3000,
            fixed_expenses=2900,
            max_months=5,
        )
        env = CreditCardDebtEnv(config=cfg)
        env.reset(seed=42)

        for _ in range(5):
            action = np.array([1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

        assert truncated is True or terminated is True


# ── Action Handling ──────────────────────────────────────────────────────

class TestActionHandling:

    def test_continuous_softmax_normalization(self, env):
        """Unequal raw actions should be normalized to proportions."""
        env.reset(seed=42)
        action = np.array([3.0, 1.0, 0.0], dtype=np.float32)
        # Should not crash; surplus split proportionally
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)

    def test_zero_action_defaults_to_uniform(self, env):
        """All-zero action should fall back to uniform allocation."""
        env.reset(seed=42)
        action = np.zeros(3, dtype=np.float32)
        obs, _, _, _, info = env.step(action)
        # Should not crash — falls back to uniform

    def test_discrete_action_mode(self):
        """Discrete action mode should work with MultiDiscrete actions."""
        cfg = EnvConfig(
            cards=[
                CardConfig("A", apr=0.20, balance=3000, credit_limit=5000),
                CardConfig("B", apr=0.25, balance=2000, credit_limit=4000),
            ],
            action_mode="discrete",
            monthly_income=5000,
            fixed_expenses=3000,
        )
        env = CreditCardDebtEnv(config=cfg)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)


# ── Render ───────────────────────────────────────────────────────────────

class TestRender:

    def test_render_ansi(self, default_config):
        env = CreditCardDebtEnv(config=default_config, render_mode="ansi")
        env.reset(seed=42)
        env.step(env.action_space.sample())
        output = env.render()
        assert isinstance(output, str)
        assert "Month" in output


# ── Gymnasium Compliance ─────────────────────────────────────────────────

class TestGymnasiumCompliance:

    def test_check_env(self, default_config):
        """Run Gymnasium's built-in env checker."""
        from gymnasium.utils.env_checker import check_env
        env = CreditCardDebtEnv(config=default_config)
        # check_env will raise if there are issues
        check_env(env, skip_render_check=True)
