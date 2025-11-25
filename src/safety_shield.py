import gymnasium as gym
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv


class SafetyShieldWrapper(gym.Wrapper):
    """
    Implements a 'Progressive Repulsive Shield'.
    Fixes the 'Deadlock' issue where the agent stops completely to be safe.
    """

    def __init__(self, env, lidar_threshold=0.25, num_lasers=240):
        super().__init__(env)
        self.lidar_threshold = lidar_threshold
        self.num_lasers = num_lasers
        self.shield_activations = 0
        self.last_obs = None

    def reset(self, *, seed=None, options=None):
        try:
            obs, info = self.env.reset(seed=seed, options=options)
        except TypeError:
            obs, info = self.env.reset(seed=seed)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        if self.last_obs is not None:
            # Lidar processing
            lidar_readings = self.last_obs[-self.num_lasers :]
            sector_size = self.num_lasers // 3

            left_sector = lidar_readings[0:sector_size]
            front_sector = lidar_readings[sector_size : 2 * sector_size]
            right_sector = lidar_readings[2 * sector_size :]

            min_left = np.min(left_sector)
            min_front = np.min(front_sector)
            min_right = np.min(right_sector)

            raw_steering = action[0]
            raw_accel = action[1]

            safe_steering = raw_steering
            safe_accel = raw_accel
            shield_active = False

            if min_front < 0.10:
                safe_accel = -1.0
                shield_active = True
            elif min_front < self.lidar_threshold:
                if raw_accel > 0.0:
                    safe_accel = 0.0
                    shield_active = True

            if min_right < 0.15:
                if raw_steering < 0.05:
                    safe_steering = 0.25
                    shield_active = True

            elif min_left < 0.15:
                if raw_steering > -0.05:
                    safe_steering = -0.25
                    shield_active = True

            if shield_active:
                self.shield_activations += 1
                final_action = np.array([safe_steering, safe_accel])
            else:
                final_action = action
        else:
            final_action = action

        obs, reward, terminated, truncated, info = self.env.step(final_action)
        self.last_obs = obs

        if "shield_active" in locals() and shield_active:
            reward -= 0.1
            info["shield_activated"] = True
        else:
            info["shield_activated"] = False

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    pass
