import gymnasium as gym

import numpy as np

from metadrive.envs.metadrive_env import MetaDriveEnv


class SafetyShieldWrapper(gym.Wrapper):
    """

    Implements a 'Repulsive Action-Space Shield'.

    Unlike a blocking shield (which sets action to 0), this shield actively

    pushes the agent away from danger when the threshold is breached.

    """

    def __init__(self, env, front_threshold=0.15, side_threshold=0.04, num_lasers=240):
        super().__init__(env)

        self.front_threshold = front_threshold

        self.side_threshold = side_threshold

        self.num_lasers = num_lasers

        self.shield_activations = 0

        self.last_obs = None

        self.k_steer = 2.5

        self.k_brake = 8.0

    def reset(self, *, seed=None, options=None):
        try:
            obs, info = self.env.reset(seed=seed, options=options)

        except TypeError:
            obs, info = self.env.reset(seed=seed)

        self.last_obs = obs

        return obs, info

    def step(self, action):
        final_action = action.copy()

        shield_active = False

        if self.last_obs is not None:
            lidar_readings = self.last_obs[-self.num_lasers :]

            r_side = lidar_readings[40:80]

            front = np.concatenate((lidar_readings[-15:], lidar_readings[:15]))

            l_side = lidar_readings[160:200]

            min_front = np.min(front)

            min_r_side = np.min(r_side)

            min_l_side = np.min(l_side)

            if min_front < self.front_threshold:
                danger = (self.front_threshold - min_front) / self.front_threshold

                brake_force = -1.0 * (danger * self.k_brake)

                brake_force = np.clip(brake_force, -1.0, 0.0)

                if final_action[1] > brake_force:
                    final_action[1] = brake_force

                    shield_active = True

            steering_correction = 0.0

            if min_r_side < self.side_threshold:
                push = (self.side_threshold - min_r_side) * self.k_steer

                steering_correction += push

            if min_l_side < self.side_threshold:
                push = (self.side_threshold - min_l_side) * self.k_steer

                steering_correction -= push

            if abs(steering_correction) > 0.05:
                new_steer = final_action[0] + steering_correction

                final_action[0] = np.clip(new_steer, -1.0, 1.0)

                shield_active = True

        obs, env_reward, terminated, truncated, info = self.env.step(final_action)

        self.last_obs = obs

        if shield_active:
            self.shield_activations += 1

            info["shield_activated"] = True

            reward = -0.5

        else:
            info["shield_activated"] = False

            reward = env_reward

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    pass
