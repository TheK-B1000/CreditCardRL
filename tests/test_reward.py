"""Unit tests for the reward function."""

import pytest

from src.envs.financial_model import CardState
from src.envs.reward import RewardConfig, compute_step_reward, compute_terminal_reward


@pytest.fixture
def cfg() -> RewardConfig:
    return RewardConfig()


@pytest.fixture
def active_cards() -> list[CardState]:
    return [
        CardState("A", apr=0.24, balance=5000, credit_limit=10000, min_payment_floor=25),
        CardState("B", apr=0.18, balance=2000, credit_limit=5000, min_payment_floor=25),
    ]


class TestStepReward:

    def test_zero_interest_zero_penalty(self, cfg, active_cards):
        """No interest → no interest penalty component."""
        r = compute_step_reward(cfg, 0.0, 10000.0, active_cards, 0, 0)
        # Still has utilization penalty (cards are at 50% and 40%, both > 30%)
        assert isinstance(r, float)

    def test_interest_penalty_scales(self, cfg, active_cards):
        """Higher interest → lower (more negative) reward."""
        r1 = compute_step_reward(cfg, 50.0, 10000.0, active_cards, 0, 0)
        r2 = compute_step_reward(cfg, 200.0, 10000.0, active_cards, 0, 0)
        assert r2 < r1

    def test_missed_minimum_penalty(self, cfg, active_cards):
        """Missing minimums adds a large penalty."""
        r_ok = compute_step_reward(cfg, 100.0, 10000.0, active_cards, 0, 0)
        r_miss = compute_step_reward(cfg, 100.0, 10000.0, active_cards, 2, 0)
        assert r_miss < r_ok
        # Penalty should be γ × 2 = 5.0 × 2 = 10.0 less
        assert r_ok - r_miss == pytest.approx(cfg.gamma_ * 2, abs=0.01)

    def test_payoff_bonus(self, cfg, active_cards):
        """Paying off a card adds δ per card."""
        r_none = compute_step_reward(cfg, 100.0, 10000.0, active_cards, 0, 0)
        r_one = compute_step_reward(cfg, 100.0, 10000.0, active_cards, 0, 1)
        assert r_one > r_none
        assert r_one - r_none == pytest.approx(cfg.delta, abs=0.01)

    def test_utilization_penalty_zero_when_under_target(self, cfg):
        """No utilization penalty if all cards are under target."""
        low_util_cards = [
            CardState("A", apr=0.24, balance=1000, credit_limit=10000, min_payment_floor=25),
            CardState("B", apr=0.18, balance=500, credit_limit=10000, min_payment_floor=25),
        ]
        r = compute_step_reward(cfg, 0.0, 5000.0, low_util_cards, 0, 0)
        # With zero interest and under-target utilization, reward should be ~0
        assert r == pytest.approx(0.0, abs=0.01)


class TestTerminalReward:

    def test_all_paid_bonus(self, cfg):
        r = compute_terminal_reward(cfg, all_debt_paid=True, months_elapsed=24, max_months=60)
        assert r > 0  # ε - ζ × (24/60) = 10 - 0.4 = 9.6
        expected = cfg.epsilon - cfg.zeta * (24 / 60)
        assert r == pytest.approx(expected, abs=0.01)

    def test_not_paid_penalty(self, cfg):
        r = compute_terminal_reward(cfg, all_debt_paid=False, months_elapsed=60, max_months=60)
        assert r < 0  # 0 - ζ × 1.0 = -1.0
        assert r == pytest.approx(-cfg.zeta, abs=0.01)

    def test_faster_payoff_better(self, cfg):
        r_fast = compute_terminal_reward(cfg, True, months_elapsed=12, max_months=60)
        r_slow = compute_terminal_reward(cfg, True, months_elapsed=48, max_months=60)
        assert r_fast > r_slow
