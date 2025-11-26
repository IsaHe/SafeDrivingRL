import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.safety_shield import SafetyShieldWrapper
from src.ppo_agent import ActorCritic


class AgentDashboard:
    """
    Real-time visualization of the agent's observation (Lidar) and actions.
    """

    def __init__(self, num_lasers):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 6))

        self.ax_lidar = self.fig.add_subplot(121, projection="polar")

        self.num_lasers = num_lasers
        self.angles = np.linspace(0, 2 * np.pi, num_lasers, endpoint=False)

        (self.lidar_line,) = self.ax_lidar.plot([], [], color="blue", linewidth=2)
        self.ax_lidar.set_title("Agent Vision (Lidar)", pad=20)
        self.ax_lidar.set_ylim(0, 1)
        self.ax_lidar.set_yticklabels([])

    def update(self, obs, action):
        if len(obs) < self.num_lasers:
            return

        lidar_data = obs[-self.num_lasers :]

        self.lidar_line.set_data(self.angles, lidar_data)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def evaluate():
    """
    Main evaluation loop with real-time dashboard.
    """
    # Standard or shielded
    MODEL_NAME = "ppo_shielded.pth"
    ENABLE_SHIELD = True
    NUM_EPISODES = 5
    RENDER = True
    SHOW_DASHBOARD = True

    model_path = os.path.join("./data/models", MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    num_lasers = 240
    env_config = {
        "use_render": RENDER,
        "manual_control": False,
        "traffic_density": 0.15,
        "num_scenarios": 1,
        "start_seed": 42,
        "vehicle_config": {
            "lidar": {"num_lasers": num_lasers, "distance": 50, "num_others": 0}
        },
    }

    env_raw = MetaDriveEnv(env_config)

    if ENABLE_SHIELD:
        print("Evaluation Mode: Shield Active")
        env = SafetyShieldWrapper(env_raw, lidar_threshold=0.25, num_lasers=num_lasers)
    else:
        print("Evaluation Mode: Standard (No Shield)")
        env = env_raw

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = ActorCritic(state_dim, action_dim)
    policy.load_state_dict(torch.load(model_path, map_location=torch.device("mps")))
    policy.eval()

    if SHOW_DASHBOARD:
        dashboard = AgentDashboard(num_lasers)
        print(
            "Dashboard initialized. You should see two windows: Simulation and Dashboard."
        )

    print(f"Loaded model: {MODEL_NAME}")
    print("Starting evaluation...")

    try:
        total_rewards = []
        crashes = 0
        interventions = 0

        for ep in range(NUM_EPISODES):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            step = 0

            while not (done or truncated):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    actor_features = policy.actor(obs_tensor)
                    action_mean = policy.actor_mean(actor_features)
                    action = action_mean.cpu().numpy().flatten()

                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step += 1

                if RENDER:
                    env.render()

                if SHOW_DASHBOARD:
                    dashboard.update(obs, action)

                if info.get("shield_activated", False):
                    interventions += 1

            outcome = "Unknown"
            if info.get("crash_vehicle", False):
                crashes += 1
                outcome = "Crash (Vehicle)"
            elif info.get("crash_object", False):
                crashes += 1
                outcome = "Crash (Object)"
            elif info.get("arrive_dest", False):
                outcome = "Success"
            else:
                outcome = "Timeout"

            total_rewards.append(episode_reward)
            print(f"Episode {ep + 1}: Reward: {episode_reward:.2f} | Result: {outcome}")

        avg_reward = sum(total_rewards) / len(total_rewards)
        print("\n--- EVALUATION SUMMARY ---")
        print(f"Model: {MODEL_NAME}")
        print(f"Shield Active: {ENABLE_SHIELD}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Total Crashes: {crashes}/{NUM_EPISODES}")
        if ENABLE_SHIELD:
            print(f"Total Shield Interventions: {interventions}")

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
    finally:
        env.close()
        plt.close("all")
        print("Evaluation finished.")


if __name__ == "__main__":
    evaluate()
