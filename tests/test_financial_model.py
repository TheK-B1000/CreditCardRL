"""Unit tests for the financial model functions.

Tests verify interest accrual, minimum payments, late fees, and balance updates
against hand-calculated expected values.
"""

import pytest

from src.envs.financial_model import (
    CardState,
    compute_interest,
    compute_late_fee,
    compute_min_payment,
    compute_overall_utilization,
    compute_weighted_avg_apr,
    update_balance,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_card() -> CardState:
    """Standard test card: $5000 balance, 24% APR, $10k limit."""
    return CardState(
        name="Test Card",
        apr=0.24,
        balance=5000.0,
        credit_limit=10000.0,
        min_payment_floor=25.0,
    )


@pytest.fixture
def small_balance_card() -> CardState:
    """Card with tiny balance for edge-case testing."""
    return CardState(
        name="Small Card",
        apr=0.199,
        balance=15.0,
        credit_limit=5000.0,
        min_payment_floor=25.0,
    )


@pytest.fixture
def paid_off_card() -> CardState:
    """Card already paid off."""
    return CardState(
        name="Done Card",
        apr=0.199,
        balance=0.0,
        credit_limit=5000.0,
        min_payment_floor=25.0,
        is_paid_off=True,
    )


# ── Interest Accrual ──────────────────────────────────────────────────────

class TestInterestAccrual:

    def test_monthly_interest(self, sample_card):
        """Interest = balance × (APR / 12)."""
        interest = compute_interest(sample_card)
        expected = 5000.0 * (0.24 / 12)  # = $100
        assert interest == pytest.approx(expected, abs=0.01)

    def test_interest_twelve_months_minimum_only(self, sample_card):
        """12 months of minimum-only payments: compare to compound formula.

        Each month: interest accrues, minimum is paid, balance grows net of payment.
        We verify the final balance against a manual step-by-step calculation.
        """
        card = sample_card
        monthly_rate = 0.24 / 12  # 0.02

        # Simulate 12 months
        for _ in range(12):
            interest = compute_interest(card)
            min_pay = compute_min_payment(card, interest)
            update_balance(card, min_pay, interest, 0.0)

        # Manual calculation: each month balance = balance + interest - min_pay
        # min_pay = max(25, 0.01 * B + interest)
        # Let's compute step by step
        manual_balance = 5000.0
        for _ in range(12):
            i = manual_balance * monthly_rate
            mp = max(25.0, 0.01 * manual_balance + i)
            manual_balance = manual_balance + i - mp
            manual_balance = max(0.0, manual_balance)

        assert card.balance == pytest.approx(manual_balance, abs=0.01)

    def test_interest_zero_for_paid_off(self, paid_off_card):
        """No interest on a paid-off card."""
        assert compute_interest(paid_off_card) == 0.0

    def test_interest_zero_for_zero_balance(self):
        """No interest if balance is zero (even if not marked paid off)."""
        card = CardState("X", apr=0.24, balance=0.0, credit_limit=5000, min_payment_floor=25)
        assert compute_interest(card) == 0.0


# ── Minimum Payment ──────────────────────────────────────────────────────

class TestMinPayment:

    def test_formula_higher_than_floor(self, sample_card):
        """When 1% of balance + interest > $25, use the formula."""
        interest = compute_interest(sample_card)  # $100
        min_pay = compute_min_payment(sample_card, interest)
        expected = 0.01 * 5000 + 100  # $150
        assert min_pay == pytest.approx(expected, abs=0.01)

    def test_floor_used_for_low_balance(self):
        """When 1% + interest < $25, use the $25 floor."""
        card = CardState("Low", apr=0.12, balance=500, credit_limit=2000, min_payment_floor=25)
        interest = compute_interest(card)  # 500 * 0.01 = $5
        min_pay = compute_min_payment(card, interest)
        formula_val = 0.01 * 500 + 5  # $10
        assert formula_val < 25.0  # Confirm floor kicks in
        assert min_pay == pytest.approx(25.0, abs=0.01)

    def test_cap_at_total_owed(self, small_balance_card):
        """Min payment can't exceed balance + interest."""
        interest = compute_interest(small_balance_card)
        min_pay = compute_min_payment(small_balance_card, interest)
        total_owed = small_balance_card.balance + interest
        assert min_pay <= total_owed + 0.01

    def test_zero_for_paid_off(self, paid_off_card):
        assert compute_min_payment(paid_off_card, 0.0) == 0.0


# ── Late Fees ────────────────────────────────────────────────────────────

class TestLateFees:

    def test_no_fee_when_minimum_met(self, sample_card):
        """No late fee if payment ≥ minimum."""
        interest = compute_interest(sample_card)
        min_pay = compute_min_payment(sample_card, interest)
        fee = compute_late_fee(sample_card, min_pay, min_pay)
        assert fee == 0.0
        assert sample_card.consecutive_late == 0

    def test_first_offense_30(self, sample_card):
        """First missed minimum → $30 fee."""
        interest = compute_interest(sample_card)
        min_pay = compute_min_payment(sample_card, interest)
        fee = compute_late_fee(sample_card, 0.0, min_pay)  # Paid nothing
        assert fee == pytest.approx(30.0)
        assert sample_card.consecutive_late == 1

    def test_repeat_offense_41(self, sample_card):
        """Second consecutive miss → $41 fee."""
        interest = compute_interest(sample_card)
        min_pay = compute_min_payment(sample_card, interest)

        # First miss
        compute_late_fee(sample_card, 0.0, min_pay)
        # Second miss
        fee = compute_late_fee(sample_card, 0.0, min_pay)
        assert fee == pytest.approx(41.0)
        assert sample_card.consecutive_late == 2

    def test_consecutive_resets_on_payment(self, sample_card):
        """Counter resets when minimum is met after a miss."""
        interest = compute_interest(sample_card)
        min_pay = compute_min_payment(sample_card, interest)

        compute_late_fee(sample_card, 0.0, min_pay)  # Miss
        assert sample_card.consecutive_late == 1

        compute_late_fee(sample_card, min_pay, min_pay)  # Pay
        assert sample_card.consecutive_late == 0


# ── Balance Update ────────────────────────────────────────────────────────

class TestBalanceUpdate:

    def test_basic_update(self, sample_card):
        """B_{t+1} = B_t + I + F - P."""
        interest = 100.0
        payment = 250.0
        fee = 0.0
        new_bal = update_balance(sample_card, payment, interest, fee)
        expected = 5000 + 100 - 250  # = 4850
        assert new_bal == pytest.approx(expected, abs=0.01)
        assert sample_card.balance == pytest.approx(expected, abs=0.01)

    def test_balance_floors_at_zero(self, sample_card):
        """Overpayment should not create negative balance."""
        new_bal = update_balance(sample_card, 10000.0, 100.0, 0.0)
        assert new_bal == 0.0
        assert sample_card.balance == 0.0
        assert sample_card.is_paid_off is True

    def test_pay_to_zero_marks_paid_off(self):
        """Exact payoff → is_paid_off = True."""
        card = CardState("X", apr=0.24, balance=100, credit_limit=5000, min_payment_floor=25)
        interest = compute_interest(card)
        update_balance(card, 100 + interest, interest, 0.0)
        assert card.balance == 0.0
        assert card.is_paid_off is True

    def test_fees_added_to_balance(self, sample_card):
        """Late fees increase the balance."""
        interest = 100.0
        fee = 30.0
        payment = 100.0  # Just covers interest
        new_bal = update_balance(sample_card, payment, interest, fee)
        expected = 5000 + 100 + 30 - 100  # = 5030
        assert new_bal == pytest.approx(expected, abs=0.01)


# ── Utilization ───────────────────────────────────────────────────────────

class TestUtilization:

    def test_single_card_utilization(self, sample_card):
        assert sample_card.utilization == pytest.approx(0.5)  # 5000/10000

    def test_overall_utilization(self):
        cards = [
            CardState("A", apr=0.2, balance=3000, credit_limit=10000, min_payment_floor=25),
            CardState("B", apr=0.2, balance=2000, credit_limit=5000, min_payment_floor=25),
        ]
        util = compute_overall_utilization(cards)
        assert util == pytest.approx(5000 / 15000, abs=0.001)

    def test_weighted_avg_apr(self):
        cards = [
            CardState("A", apr=0.24, balance=3000, credit_limit=10000, min_payment_floor=25),
            CardState("B", apr=0.12, balance=1000, credit_limit=5000, min_payment_floor=25),
        ]
        avg = compute_weighted_avg_apr(cards)
        expected = (0.24 * 3000 + 0.12 * 1000) / 4000
        assert avg == pytest.approx(expected, abs=0.001)
