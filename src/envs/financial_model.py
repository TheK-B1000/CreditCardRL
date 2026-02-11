"""Financial model for credit card debt simulation.

Implements the core financial math:
- APR → monthly periodic rate conversion
- Interest accrual on revolving balances
- Minimum payment calculation
- Late fee logic
- Utilization computation
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CardState:
    """Mutable state for a single credit card during simulation."""

    name: str
    apr: float                  # Annual percentage rate (e.g., 0.219 for 21.9%)
    balance: float              # Current outstanding balance
    credit_limit: float         # Total credit limit
    min_payment_floor: float    # Minimum dollar floor for min payment (e.g., $25)
    consecutive_late: int = 0   # Count of consecutive months with missed minimum
    is_paid_off: bool = False   # Whether this card has been fully paid off

    @property
    def monthly_rate(self) -> float:
        """APR ÷ 12 — simple periodic rate used by most issuers."""
        return self.apr / 12.0

    @property
    def utilization(self) -> float:
        """Current balance / credit limit. 0 if limit is 0."""
        if self.credit_limit <= 0:
            return 0.0
        return self.balance / self.credit_limit


def compute_interest(card: CardState) -> float:
    """Compute monthly interest accrued on current balance.

    Formula: I_t = B_t × (APR / 12)

    Returns:
        Interest amount (≥ 0). Zero if card is paid off.
    """
    if card.is_paid_off or card.balance <= 0:
        return 0.0
    return card.balance * card.monthly_rate


def compute_min_payment(card: CardState, interest: float) -> float:
    """Compute minimum payment due for the current billing cycle.

    Rule: min_payment = max(floor, 1% of balance + interest)
    Capped at the total owed (balance + interest) so we never demand more than is owed.

    Args:
        card: Current card state.
        interest: Interest accrued this cycle.

    Returns:
        Minimum payment amount (≥ 0).
    """
    if card.is_paid_off or card.balance <= 0:
        return 0.0

    total_owed = card.balance + interest
    min_pay = max(card.min_payment_floor, 0.01 * card.balance + interest)
    # Never require more than total owed
    return min(min_pay, total_owed)


def compute_late_fee(card: CardState, payment_made: float, min_payment_due: float) -> float:
    """Determine late fee if payment is less than the minimum due.

    Schedule:
        - First offense (or first after being current): $30
        - Repeat offense (consecutive): $41

    Also updates `card.consecutive_late` counter.

    Args:
        card: Current card state (mutated: consecutive_late updated).
        payment_made: Amount the consumer actually paid this month.
        min_payment_due: Minimum payment that was required.

    Returns:
        Late fee charged (0 if payment meets minimum).
    """
    LATE_FEE_FIRST = 30.0
    LATE_FEE_REPEAT = 41.0

    if card.is_paid_off or min_payment_due <= 0:
        card.consecutive_late = 0
        return 0.0

    if payment_made < min_payment_due - 1e-6:  # Small tolerance for float comparison
        card.consecutive_late += 1
        if card.consecutive_late <= 1:
            return LATE_FEE_FIRST
        else:
            return LATE_FEE_REPEAT
    else:
        card.consecutive_late = 0
        return 0.0


def update_balance(card: CardState, payment: float, interest: float, fees: float) -> float:
    """Advance one month: apply interest, fees, and subtract payment.

    Formula: B_{t+1} = B_t + I_t + F_t − P_t  (floored at 0)

    Args:
        card: Current card state (mutated: balance, is_paid_off updated).
        payment: Total payment applied to this card.
        interest: Interest accrued this cycle.
        fees: Total fees (e.g., late fees) this cycle.

    Returns:
        The new balance after update.
    """
    new_balance = card.balance + interest + fees - payment
    card.balance = max(0.0, new_balance)

    if card.balance < 0.01:  # Treat sub-penny balances as paid off
        card.balance = 0.0
        card.is_paid_off = True

    return card.balance


def compute_overall_utilization(cards: list[CardState]) -> float:
    """Overall utilization = sum(balances) / sum(credit_limits).

    Returns 0 if total credit limit is 0.
    """
    total_balance = sum(c.balance for c in cards)
    total_limit = sum(c.credit_limit for c in cards)
    if total_limit <= 0:
        return 0.0
    return total_balance / total_limit


def compute_weighted_avg_apr(cards: list[CardState]) -> float:
    """Balance-weighted average APR across all cards.

    Returns 0 if total balance is 0 (all paid off).
    """
    total_balance = sum(c.balance for c in cards)
    if total_balance <= 0:
        return 0.0
    return sum(c.apr * c.balance for c in cards) / total_balance
