"""Custom Stable-Baselines3 callbacks for credit-card RL training."""

from __future__ import annotations

import sys
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


class EpisodeLoggerCallback(BaseCallback):
    """Print a one-line summary to stdout after every completed episode.

    Output format (one line per episode):
        [PPO|3card] ep=142 result=PAID_OFF months=23/60 interest=$1,247.33
            cards=3/3 util=34.2% score=681 reward=12.45 | PAID=89 TIMEOUT=53

    Args:
        scenario_tag: Short label for the env scenario (e.g. "3card", "stress5").
        algo_tag:     Short label for the algorithm (default "PPO").
        log_freq:     Print every N episodes (1 = every episode).
        flush:        Flush stdout after each print (useful for piped output).
        verbose:      SB3 verbose level.
    """

    def __init__(
        self,
        scenario_tag: str = "CreditCard",
        algo_tag: str = "PPO",
        log_freq: int = 1,
        flush: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.scenario_tag = scenario_tag
        self.algo_tag = algo_tag
        self.log_freq = max(1, log_freq)
        self.flush = flush

        # Running counters
        self._ep_count: int = 0
        self._paid_off_count: int = 0
        self._timeout_count: int = 0

        # Per-env accumulators
        self._ep_interest: list[list[float]] = []
        self._ep_utilization: list[list[float]] = []
        self._ep_months: list[int] = []
        self._ep_reward: list[float] = []

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs  # type: ignore[union-attr]
        self._ep_interest = [[] for _ in range(n_envs)]
        self._ep_utilization = [[] for _ in range(n_envs)]
        self._ep_months = [0] * n_envs
        self._ep_reward = [0.0] * n_envs

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        for i, info in enumerate(infos):
            # Accumulate step-level data
            interests = info.get("interests", [])
            step_interest = sum(interests) if interests else 0.0
            self._ep_interest[i].append(step_interest)

            overall_util = info.get("overall_utilization", 0.0)
            self._ep_utilization[i].append(overall_util)

            self._ep_months[i] = info.get("month", self._ep_months[i])

            if i < len(rewards):
                self._ep_reward[i] += float(rewards[i])

            # Check for episode end (SB3 Monitor wrapper adds "episode" key)
            episode_info = info.get("episode")
            if episode_info is not None:
                self._ep_count += 1

                # Determine outcome
                terminal_info = info.get("terminal_info", info)
                all_paid = terminal_info.get("all_paid", info.get("all_paid", False))

                if all_paid:
                    result = "PAID_OFF"
                    self._paid_off_count += 1
                else:
                    result = "TIMEOUT"
                    self._timeout_count += 1

                # Gather metrics
                months = self._ep_months[i]
                total_interest = sum(self._ep_interest[i])
                utils = self._ep_utilization[i]
                avg_util = float(np.mean(utils)) if utils else 0.0
                cards_paid = info.get("cards_paid_off", 0)
                total_cards = len(info.get("balances", []))

                # Credit proxy score
                debt_reduction = 1.0 if all_paid else 0.5
                credit_score = compute_credit_proxy_score(
                    avg_utilization=avg_util,
                    missed_min_ratio=0.0,
                    debt_reduction_ratio=debt_reduction,
                )

                ep_reward = self._ep_reward[i]

                # Max months from env (via info or training_env)
                max_months = 60  # fallback
                try:
                    env = self.training_env.envs[i]  # type: ignore
                    max_months = getattr(env, "max_months", 60)
                except Exception:
                    pass

                # Print log line
                if self._ep_count % self.log_freq == 0:
                    line = (
                        f"[{self.algo_tag}|{self.scenario_tag}] "
                        f"ep={self._ep_count} "
                        f"result={result} "
                        f"months={months}/{max_months} "
                        f"interest=${total_interest:,.2f} "
                        f"cards={cards_paid}/{total_cards} "
                        f"util={avg_util:.1%} "
                        f"score={credit_score:.0f} "
                        f"reward={ep_reward:.2f} "
                        f"| PAID={self._paid_off_count} "
                        f"TIMEOUT={self._timeout_count}"
                    )
                    print(line, flush=self.flush)

                # Reset per-env accumulators
                self._ep_interest[i] = []
                self._ep_utilization[i] = []
                self._ep_months[i] = 0
                self._ep_reward[i] = 0.0

        return True
