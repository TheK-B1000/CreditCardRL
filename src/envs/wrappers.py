"""Environment wrappers for training and evaluation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from src.envs.credit_env import CreditCardDebtEnv
from src.envs.reward import RewardConfig
from src.envs.scenario_sampler import ScenarioSampler
from src.utils.config import EnvConfig


class RandomScenarioWrapper(gym.Wrapper):
    """Wraps CreditCardDebtEnv to sample a new random scenario on each reset.

    Used for diverse-scenario training so the policy sees many 3-card scenarios
    instead of a single fixed one. Observation and action space match the inner env
    (e.g. 3-card continuous).
    """

    def __init__(
        self,
        env: CreditCardDebtEnv,
        sampler: ScenarioSampler,
        reward_cfg: RewardConfig | None = None,
    ):
        super().__init__(env)
        self.sampler = sampler
        self.reward_cfg = reward_cfg or env.reward_cfg

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        rng = np.random.default_rng(seed)
        scenario: EnvConfig = self.sampler.sample(rng)
        scenario.reward = self.reward_cfg
        return self.env.reset(seed=seed, options={"config": scenario})
