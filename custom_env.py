<<<<<<< HEAD
"""
Custom reward wrapper for CartPole.
Penalizes pole angle using angle_penalty_weight.
"""

from typing import Tuple

import gymnasium as gym
import numpy as np


class AnglePenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, angle_penalty_weight: float):
        super().__init__(env)
        self.weight = angle_penalty_weight

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        pole_angle = float(obs[2])

        # Modified reward
        reward = max(1.0 - self.weight * abs(pole_angle), 0.0)

        return obs, reward, terminated, truncated, info
=======
"""
Custom reward wrapper for CartPole.
Penalizes pole angle using angle_penalty_weight.
"""

from typing import Tuple

import gymnasium as gym
import numpy as np


class AnglePenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, angle_penalty_weight: float):
        super().__init__(env)
        self.weight = angle_penalty_weight

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        pole_angle = float(obs[2])

        # Modified reward
        reward = max(1.0 - self.weight * abs(pole_angle), 0.0)

        return obs, reward, terminated, truncated, info
>>>>>>> 00debab322f4b7ccc48e1f27f50b6ef94360d3dc
