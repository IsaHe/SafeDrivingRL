import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.safety_shield import SafetyShieldWrapper
from src.ppo_agent import PPOAgent
from src.reward_shaper import RewardShapingWrapper


class AgentDashboard:
    def __init__(self, num_lasers, lidar_threshold=None):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 6))

        self.ax_lidar = self.fig.add_subplot(121, projection="polar")

        self.num_lasers = num_lasers
        self.angles = np.linspace(0, 2 * np.pi, num_lasers, endpoint=False)

        (self.lidar_line,) = self.ax_lidar.plot(
            [], [], color="blue", linewidth=2, label="Lidar Scan"
        )

        if lidar_threshold is not None:
            theta_circle = np.linspace(0, 2 * np.pi, 200)
            r_circle = np.full_like(theta_circle, lidar_threshold)

            self.ax_lidar.plot(
                theta_circle,
                r_circle,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold ({lidar_threshold})",
            )

            self.ax_lidar.fill_between(
                theta_circle, 0, r_circle, color="red", alpha=0.1
            )

        self.ax_lidar.set_title("Agent Vision (Lidar)", pad=20)
        self.ax_lidar.set_ylim(0, 1)
        self.ax_lidar.set_yticklabels([])

        self.ax_lidar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2))

    def update(self, obs, action):
        if len(obs) < self.num_lasers:
            return

        lidar_data = obs[-self.num_lasers :]

        self.lidar_line.set_data(self.angles, lidar_data)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def get_args():
    parser = argparse.ArgumentParser(description="PPO Evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Full name of the model file (e.g. ppo_shielded.pth)",
    )
    parser.add_argument(
        "--lidar_threshold",
        type=float,
        default=0.10,
        help="Lidar threshold for shield activation",
    )
    parser.add_argument(
        "--no_shield", action="store_true", help="Disable the safety shield"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--traffic_density",
        type=float,
        default=0.1,
        help="Density of traffic in MetaDrive",
    )
    parser.add_argument(
        "--action_smoothing",
        type=float,
        default=0.0,
        help="Must match the value used in training for consistent behavior.",
    )
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    return parser.parse_args()


def evaluate():
    args = get_args()

    MODEL_NAME = args.model_name
    ENABLE_SHIELD = not args.no_shield
    LIDAR_THRESHOLD = args.lidar_threshold
    NUM_EPISODES = args.episodes
    RENDER = not args.no_render
    SHOW_DASHBOARD = True
    TRAFFIC_DENSITY = args.traffic_density
    ACTION_SMOOTHING = args.action_smoothing

    model_path = os.path.join("./data/models", MODEL_NAME)
    if not os.path.exists(model_path):
        if os.path.exists(MODEL_NAME):
            model_path = MODEL_NAME
        else:
            print(f"Error: Model not found at {model_path}")
            return

    num_lasers = 240
    env_config = {
        "use_render": RENDER,
        "manual_control": False,
        "traffic_density": 0.10,
        "num_scenarios": 1,
        "start_seed": 100,
        "map": "SSSSSS",
        "vehicle_config": {
            "lidar": {"num_lasers": num_lasers, "distance": 50, "num_others": 0}
        },
    }

    env = RewardShapingWrapper(
        env=MetaDriveEnv(env_config),
        speed_weight=0.0,
        smoothness_weight=0.0,
        centering_weight=0.0,
        action_smoothing=ACTION_SMOOTHING,
    )

    if ENABLE_SHIELD:
        print(f"Evaluation Mode: Shield Active (Threshold: {LIDAR_THRESHOLD})")
        env = SafetyShieldWrapper(
            env, num_lasers=num_lasers, lidar_threshold=LIDAR_THRESHOLD
        )
    else:
        print("Evaluation Mode: Standard (No Shield)")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)

    try:
        agent.policy.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
    except AttributeError:
        if hasattr(agent, "load"):
            agent.load(model_path)
        else:
            print(
                "Error: No se pudo cargar el modelo en PPOAgent. Verifica la estructura de tu agente."
            )
            return

    agent.policy.eval()

    if SHOW_DASHBOARD:
        dashboard = AgentDashboard(num_lasers, lidar_threshold=LIDAR_THRESHOLD)
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
                try:
                    action, _, _ = agent.select_action(obs, deterministic=True)
                except TypeError:
                    action, _, _ = agent.select_action(obs)

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
        print(f"Success Rate: {(NUM_EPISODES - crashes) / NUM_EPISODES * 100:.1f}%")
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
