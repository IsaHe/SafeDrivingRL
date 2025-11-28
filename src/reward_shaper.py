import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, speed_weight=0.05, smoothness_weight=0.5):
        super().__init__(env)
        self.speed_weight = speed_weight
        self.smoothness_weight = smoothness_weight
        self.last_steering = 0.0

    def reset(self, **kwargs):
        self.last_steering = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        current_steering = action[0]

        obs, reward, done, truncated, info = self.env.step(action)

        velocity = info.get("velocity", 0.0)
        speed_reward = velocity * self.speed_weight

        steering_diff = abs(current_steering - self.last_steering)
        smoothness_penalty = steering_diff * self.smoothness_weight

        shaped_reward = reward + speed_reward - smoothness_penalty

        self.last_steering = current_steering

        info["shaped_reward"] = shaped_reward
        info["raw_reward"] = reward

        return obs, shaped_reward, done, truncated, info
