"""Credit health metrics for baseline evaluation.

Provides a simplified credit-proxy score inspired by FICO scoring factors.
"""

from __future__ import annotations


def compute_credit_proxy_score(
    avg_utilization: float,
    missed_min_ratio: float,
    debt_reduction_ratio: float,
) -> float:
    """Compute a simplified credit health proxy score in [300, 850].

    Inspired by FICO scoring weights:
        - Payment history (35%): penalizes missed minimum payments
        - Utilization (30%): penalizes high credit utilization
        - Debt reduction (35%): rewards paying down principal

    Args:
        avg_utilization: Average utilization across episode (0–1+).
        missed_min_ratio: Fraction of (card × month) events where
            the minimum was missed (0–1). 0 = always on time.
        debt_reduction_ratio: (initial_debt - final_debt) / initial_debt.
            1.0 = fully paid off, 0.0 = no progress, <0 = debt grew.

    Returns:
        Score between 300 (worst) and 850 (best).
    """
    FLOOR, CEILING = 300.0, 850.0
    score_range = CEILING - FLOOR  # 550 points

    # Payment history (35% weight) — perfect=1, all missed=0
    payment_score = max(0.0, 1.0 - missed_min_ratio)

    # Utilization (30% weight) — below 30% is ideal
    # Linear penalty: 0% util → 1.0, 100% util → 0.0
    util_score = max(0.0, 1.0 - avg_utilization)

    # Debt reduction (35% weight) — fully paid=1.0, no progress=0.0
    reduction_score = max(0.0, min(1.0, debt_reduction_ratio))

    weighted = (
        0.35 * payment_score
        + 0.30 * util_score
        + 0.35 * reduction_score
    )

    return FLOOR + weighted * score_range
