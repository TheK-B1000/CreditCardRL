"""Abstract base class for heuristic baseline policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.envs.credit_env import CreditCardDebtEnv


class BaselinePolicy(ABC):
    """Interface for scripted payment allocation strategies.

    Subclasses implement `allocate()` which returns an action suitable
    for the environment's current action_mode.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this strategy."""
        ...

    @abstractmethod
    def allocate(self, env: CreditCardDebtEnv) -> np.ndarray:
        """Decide how to allocate surplus budget across cards.

        The environment guarantees minimum payments. This method decides
        how to distribute the *surplus* (income − expenses − minimums).

        For continuous action mode: return proportions (will be softmax-normalized).
        For discrete action mode: return chunk counts per card.

        Args:
            env: The environment instance (read card states, budget, etc.).

        Returns:
            Action array compatible with env.action_space.
        """
        ...

    def run_episode(
        self,
        env: CreditCardDebtEnv,
        seed: int | None = None,
    ) -> dict:
        """Run a full episode using this policy.

        Args:
            env: Environment instance.
            seed: Reset seed.

        Returns:
            Dict with episode metrics: total_interest, months, avg_utilization,
            final_debt, all_paid, utilization_history, interest_history.
        """
        obs, info = env.reset(seed=seed)

        total_interest = 0.0
        utilization_history = []
        interest_history = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = self.allocate(env)
            obs, reward, terminated, truncated, info = env.step(action)

            step_interest = sum(info.get("interests", []))
            total_interest += step_interest
            interest_history.append(step_interest)
            utilization_history.append(info.get("overall_utilization", 0.0))

        avg_util = np.mean(utilization_history) if utilization_history else 0.0

        return {
            "strategy": self.name,
            "total_interest": total_interest,
            "months": info.get("month", env.month),
            "avg_utilization": float(avg_util),
            "final_debt": info.get("total_debt", 0.0),
            "all_paid": all(c.is_paid_off for c in env.cards),
            "utilization_history": utilization_history,
            "interest_history": interest_history,
        }
