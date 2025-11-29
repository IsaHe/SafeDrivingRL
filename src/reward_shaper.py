import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper para moldear la recompensa y suavizar las acciones.
    Ayuda a reducir el comportamiento de zigzag mediante penalizaciones y filtros.
    """

    def __init__(
        self,
        env,
        speed_weight=0.1,
        smoothness_weight=1.0,
        centering_weight=0.5,
        action_smoothing=0.0,
    ):
        super().__init__(env)
        self.speed_weight = speed_weight
        self.smoothness_weight = smoothness_weight
        self.centering_weight = centering_weight
        self.action_smoothing = action_smoothing

        self.last_steering = 0.0

    def reset(self, **kwargs):
        self.last_steering = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.action_smoothing > 0:
            smooth_steering = (1.0 - self.action_smoothing) * action[
                0
            ] + self.action_smoothing * self.last_steering
            actual_action = np.array([smooth_steering, action[1]])
        else:
            actual_action = action

        obs, reward, done, truncated, info = self.env.step(actual_action)

        current_steering = actual_action[0]

        velocity = info.get("velocity", 0.0)
        speed_reward = velocity * self.speed_weight

        steering_diff = abs(current_steering - self.last_steering)
        smoothness_penalty = steering_diff * self.smoothness_weight

        centering_penalty = (current_steering**2) * self.centering_weight

        shaped_reward = reward + speed_reward - smoothness_penalty - centering_penalty

        self.last_steering = current_steering

        info["shaped_reward"] = shaped_reward
        info["raw_reward"] = reward
        info["penalty_smoothness"] = smoothness_penalty
        info["penalty_centering"] = centering_penalty

        return obs, shaped_reward, done, truncated, info
