"""Custom Stable-Baselines3 callbacks for credit-card RL training."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.evaluation.metrics import compute_credit_proxy_score


class CreditMetricsCallback(BaseCallback):
    """Log domain-specific episode metrics to TensorBoard.

    Captures per-episode:
      - episode/total_interest
      - episode/months_to_payoff
      - episode/avg_utilization
      - episode/credit_proxy_score
      - episode/all_paid
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Accumulators for the current episode (per env in VecEnv)
        self._ep_interest: list[list[float]] = []
        self._ep_utilization: list[list[float]] = []
        self._ep_months: list[int] = []

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs  # type: ignore[union-attr]
        self._ep_interest = [[] for _ in range(n_envs)]
        self._ep_utilization = [[] for _ in range(n_envs)]
        self._ep_months = [0] * n_envs

    def _on_step(self) -> bool:
        # infos is a list[dict], one per env in the VecEnv
        infos = self.locals.get("infos", [])

        for i, info in enumerate(infos):
            # Accumulate step-level data
            if "interest_this_month" in info:
                self._ep_interest[i].append(info["interest_this_month"])
            if "utilization" in info:
                self._ep_utilization[i].append(info["utilization"])
            self._ep_months[i] = info.get("month", self._ep_months[i])

            # SB3 VecEnv wraps terminal info in "terminal_info" or
            # places episode stats in info["episode"] via Monitor wrapper.
            # We also check for our custom terminal signals.
            episode_info = info.get("episode")
            is_done = episode_info is not None

            if is_done:
                total_interest = sum(self._ep_interest[i])
                months = self._ep_months[i]
                utils = self._ep_utilization[i]
                avg_util = float(np.mean(utils)) if utils else 0.0
                all_paid = info.get("all_paid", False)

                # If terminal_observation available, check terminal info
                terminal_info = info.get("terminal_info", info)
                if "all_paid" in terminal_info:
                    all_paid = terminal_info["all_paid"]

                # Compute credit proxy score
                missed_ratio = 0.0  # Not easily available from SB3 VecEnv; default 0
                debt_reduction = 1.0 if all_paid else 0.5
                credit_score = compute_credit_proxy_score(
                    avg_utilization=avg_util,
                    missed_min_ratio=missed_ratio,
                    debt_reduction_ratio=debt_reduction,
                )

                # Log to TensorBoard via SB3's logger
                self.logger.record("episode/total_interest", total_interest)
                self.logger.record("episode/months_to_payoff", months)
                self.logger.record("episode/avg_utilization", avg_util)
                self.logger.record("episode/credit_proxy_score", credit_score)
                self.logger.record("episode/all_paid", float(all_paid))

                # Reset accumulators for this env
                self._ep_interest[i] = []
                self._ep_utilization[i] = []
                self._ep_months[i] = 0

        return True
