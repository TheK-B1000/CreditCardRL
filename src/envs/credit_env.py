"""CreditCardDebtEnv — Gymnasium environment for credit card debt repayment.

The agent allocates monthly surplus budget across multiple credit cards
to minimize total interest paid, time-to-payoff, and maintain healthy utilization.

Action space:
  - continuous: Box(num_cards,) → softmax → proportional allocation of surplus
  - discrete: MultiDiscrete → $50 chunks allocated across cards

Observation space:
  - Box(5 * num_cards + 4): per-card features + global features, normalized to [0,1]
"""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.financial_model import (
    CardState,
    compute_interest,
    compute_late_fee,
    compute_min_payment,
    compute_overall_utilization,
    compute_weighted_avg_apr,
    update_balance,
)
from src.envs.reward import RewardConfig, compute_step_reward, compute_terminal_reward
from src.utils.config import CardConfig, EnvConfig


class CreditCardDebtEnv(gym.Env):
    """Gymnasium environment simulating monthly credit card debt repayment.

    Each step represents one month. The agent decides how to allocate surplus
    budget (income − fixed_expenses − sum_of_minimums) across cards.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ):
        """Initialize environment from a typed EnvConfig.

        Args:
            config: Environment configuration. Uses defaults if None.
            render_mode: "human" for printed output, "ansi" for string return.
        """
        super().__init__()

        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self.num_cards = self.config.num_cards
        self.max_months = self.config.max_months
        self.monthly_income = self.config.monthly_income
        self.fixed_expenses = self.config.fixed_expenses
        self.action_mode = self.config.action_mode
        self.reward_cfg = self.config.reward
        # Ensure utilization_target is in sync
        self.reward_cfg.utilization_target = self.config.utilization_target

        # Store initial card configs for reset
        self._card_configs = self.config.cards

        # ── Observation space ──────────────────────────────────────────────
        # Per card (5 features × num_cards):
        #   balance_norm, credit_limit_norm, apr_norm, min_payment_norm, utilization
        # Global (4 features):
        #   surplus_budget_norm, month_norm, total_debt_norm, weighted_avg_apr_norm
        obs_dim = 5 * self.num_cards + 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space ──────────────────────────────────────────────────
        if self.action_mode == "continuous":
            # Raw proportions — we softmax-normalize in step()
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.num_cards,), dtype=np.float32
            )
        elif self.action_mode == "discrete":
            # Each card gets 0..max_chunks of $50. We clip to surplus in step().
            self.discrete_chunk_size = 50.0
            max_possible_surplus = self.monthly_income  # generous upper bound
            max_chunks = int(max_possible_surplus / self.discrete_chunk_size) + 1
            self.action_space = spaces.MultiDiscrete(
                [max_chunks + 1] * self.num_cards
            )
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode!r}")

        # ── State variables (set in reset) ────────────────────────────────
        self.cards: list[CardState] = []
        self.month: int = 0
        self.initial_total_debt: float = 0.0
        self._last_step_info: dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to a fresh episode.

        Args:
            seed: RNG seed for reproducibility.
            options: Optional dict; can contain 'config' to override EnvConfig.

        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)

        # Allow runtime config override
        if options and "config" in options:
            cfg: EnvConfig = options["config"]
            self._card_configs = cfg.cards
            self.num_cards = cfg.num_cards
            self.monthly_income = cfg.monthly_income
            self.fixed_expenses = cfg.fixed_expenses
            self.max_months = cfg.max_months
            self.reward_cfg = cfg.reward
            self.reward_cfg.utilization_target = cfg.utilization_target

        # Initialize card states from config
        self.cards = [
            CardState(
                name=cc.name,
                apr=cc.apr,
                balance=cc.balance,
                credit_limit=cc.credit_limit,
                min_payment_floor=cc.min_payment_floor,
            )
            for cc in self._card_configs
        ]

        self.month = 0
        self.initial_total_debt = sum(c.balance for c in self.cards)

        info = self._build_info()
        return self._get_obs(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one month of debt repayment.

        Pipeline:
            1. Compute interest & minimum payments for each card
            2. Decode agent action → dollar payments per card
            3. Check for missed minimums → late fees
            4. Update balances
            5. Compute reward
            6. Check termination

        Args:
            action: Payment allocation (interpretation depends on action_mode).

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        self.month += 1

        # ── 1. Interest & minimums ────────────────────────────────────────
        interests = []
        min_payments = []
        for card in self.cards:
            interest = compute_interest(card)
            min_pay = compute_min_payment(card, interest)
            interests.append(interest)
            min_payments.append(min_pay)

        total_interest = sum(interests)
        total_min_payments = sum(min_payments)

        # ── 2. Decode action → payments ───────────────────────────────────
        surplus = max(0.0, self.monthly_income - self.fixed_expenses - total_min_payments)
        payments = self._decode_action(action, min_payments, surplus)

        # ── 3. Late fees ──────────────────────────────────────────────────
        missed_count = 0
        fees = []
        for i, card in enumerate(self.cards):
            fee = compute_late_fee(card, payments[i], min_payments[i])
            fees.append(fee)
            if payments[i] < min_payments[i] - 1e-6 and not card.is_paid_off:
                missed_count += 1

        # ── 4. Update balances ────────────────────────────────────────────
        cards_paid_off_before = sum(1 for c in self.cards if c.is_paid_off)
        for i, card in enumerate(self.cards):
            update_balance(card, payments[i], interests[i], fees[i])
        cards_paid_off_after = sum(1 for c in self.cards if c.is_paid_off)
        newly_paid_off = cards_paid_off_after - cards_paid_off_before

        # ── 5. Reward ─────────────────────────────────────────────────────
        all_paid = all(c.is_paid_off for c in self.cards)
        terminated = all_paid
        truncated = self.month >= self.max_months and not terminated

        reward = compute_step_reward(
            cfg=self.reward_cfg,
            interest_accrued=total_interest,
            initial_total_debt=self.initial_total_debt,
            cards=self.cards,
            missed_minimum_count=missed_count,
            cards_paid_off_this_step=newly_paid_off,
        )

        if terminated or truncated:
            reward += compute_terminal_reward(
                cfg=self.reward_cfg,
                all_debt_paid=all_paid,
                months_elapsed=self.month,
                max_months=self.max_months,
            )

        # ── 6. Info ───────────────────────────────────────────────────────
        info = self._build_info(
            payments=payments,
            interests=interests,
            min_payments=min_payments,
            fees=fees,
            missed_count=missed_count,
            newly_paid_off=newly_paid_off,
            all_paid=all_paid,
        )
        self._last_step_info = info

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> str | None:
        """Print or return a human-readable monthly statement."""
        info = self._last_step_info
        lines = [
            f"\n{'='*60}",
            f"  Month {self.month} / {self.max_months}",
            f"{'='*60}",
        ]
        for i, card in enumerate(self.cards):
            status = "✓ PAID OFF" if card.is_paid_off else f"${card.balance:,.2f}"
            interest = info.get("interests", [0] * self.num_cards)[i]
            payment = info.get("payments", [0] * self.num_cards)[i]
            fee = info.get("fees", [0] * self.num_cards)[i]
            lines.append(
                f"  {card.name:.<25s} Balance: {status:>12s}  "
                f"Interest: ${interest:>8,.2f}  "
                f"Payment: ${payment:>8,.2f}  "
                f"Fee: ${fee:>6,.2f}  "
                f"Util: {card.utilization:>5.1%}"
            )
        total_debt = sum(c.balance for c in self.cards)
        avg_util = compute_overall_utilization(self.cards)
        lines.append(f"  {'─'*56}")
        lines.append(
            f"  Total debt: ${total_debt:>10,.2f}  "
            f"Overall util: {avg_util:.1%}  "
            f"Missed mins: {info.get('missed_count', 0)}"
        )
        output = "\n".join(lines)

        if self.render_mode == "human":
            print(output)
            return None
        return output

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _decode_action(
        self,
        action: np.ndarray,
        min_payments: list[float],
        surplus: float,
    ) -> list[float]:
        """Convert raw action into dollar payments per card.

        Every card gets at least its minimum payment. The surplus is then
        allocated according to the agent's action.

        Args:
            action: Raw action from agent.
            min_payments: Minimum due per card.
            surplus: Budget available after minimums and fixed expenses.

        Returns:
            List of dollar amounts to pay to each card.
        """
        payments = list(min_payments)  # Start with minimums

        if surplus <= 0:
            return payments

        if self.action_mode == "continuous":
            # Softmax-normalize the action to get proportions
            raw = np.array(action, dtype=np.float64).flatten()[:self.num_cards]
            # Mask paid-off cards
            mask = np.array([0.0 if c.is_paid_off else 1.0 for c in self.cards])
            raw = raw * mask
            total = raw.sum()
            if total > 1e-8:
                proportions = raw / total
            else:
                # Uniform fallback across active cards
                active = mask.sum()
                proportions = mask / active if active > 0 else np.zeros(self.num_cards)
            surplus_alloc = proportions * surplus
        else:
            # Discrete: each element is number of $50 chunks
            chunks = np.array(action, dtype=np.float64).flatten()[:self.num_cards]
            # Mask paid-off cards
            for i, c in enumerate(self.cards):
                if c.is_paid_off:
                    chunks[i] = 0
            dollar_amounts = chunks * self.discrete_chunk_size
            total_requested = dollar_amounts.sum()
            if total_requested > surplus:
                # Scale down proportionally
                scale = surplus / total_requested if total_requested > 0 else 0
                surplus_alloc = dollar_amounts * scale
            else:
                surplus_alloc = dollar_amounts

        # Add surplus to minimum payments; cap at remaining balance + interest
        for i, card in enumerate(self.cards):
            if not card.is_paid_off:
                max_useful = card.balance + card.balance * card.monthly_rate  # balance + interest
                payments[i] = min(payments[i] + surplus_alloc[i], max_useful)

        return payments

    def _get_obs(self) -> np.ndarray:
        """Build normalized observation vector.

        Per card (5 features):
            balance / initial_per_card_debt  (or 0 if initial was 0)
            credit_limit / max_credit_limit
            APR / 0.30
            min_payment / budget
            utilization

        Global (4 features):
            surplus_budget / income
            month / max_months
            total_debt / initial_total_debt
            weighted_avg_APR / 0.30
        """
        obs = []
        max_limit = max((c.credit_limit for c in self.cards), default=1.0)
        budget = self.monthly_income - self.fixed_expenses

        for i, card in enumerate(self.cards):
            initial_bal = self._card_configs[i].balance if i < len(self._card_configs) else 1.0
            obs.append(card.balance / max(initial_bal, 1.0))            # balance norm
            obs.append(card.credit_limit / max(max_limit, 1.0))         # limit norm
            obs.append(card.apr / 0.30)                                 # APR norm
            interest = compute_interest(card)
            min_pay = compute_min_payment(card, interest)
            obs.append(min_pay / max(budget, 1.0))                      # min payment norm
            obs.append(card.utilization)                                 # utilization

        # Global features
        total_min = sum(
            compute_min_payment(c, compute_interest(c)) for c in self.cards
        )
        surplus = max(0.0, budget - total_min)
        obs.append(surplus / max(self.monthly_income, 1.0))             # surplus norm
        obs.append(self.month / max(self.max_months, 1))                # time norm
        total_debt = sum(c.balance for c in self.cards)
        obs.append(total_debt / max(self.initial_total_debt, 1.0))      # debt norm
        obs.append(compute_weighted_avg_apr(self.cards) / 0.30)         # avg APR norm

        obs_array = np.array(obs, dtype=np.float32)
        # Clip to [0, 1] for safety (some values can slightly exceed 1 due to fees)
        return np.clip(obs_array, 0.0, 1.0)

    def _build_info(self, **kwargs) -> dict[str, Any]:
        """Build info dict for step/reset."""
        info: dict[str, Any] = {
            "month": self.month,
            "balances": [c.balance for c in self.cards],
            "utilizations": [c.utilization for c in self.cards],
            "overall_utilization": compute_overall_utilization(self.cards),
            "total_debt": sum(c.balance for c in self.cards),
            "cards_paid_off": sum(1 for c in self.cards if c.is_paid_off),
        }
        info.update(kwargs)
        return info
