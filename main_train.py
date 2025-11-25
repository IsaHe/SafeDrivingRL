import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.safety_shield import SafetyShieldWrapper
from src.ppo_agent import PPOAgent


def train():
    """
    Main training loop using TensorBoard for logging.
    Includes timestamps to separate different execution runs.
    """
    # --- CONFIGURATION ---
    MAX_EPISODES = 2000
    MAX_STEPS = 1000
    UPDATE_TIMESTEP = 2000
    LR = 0.0003

    # TOGGLE THIS for comparison
    USE_SHIELD = True

    RENDER = False

    # Create a unique run name with timestamp to prevent overwriting logs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode_name = "shielded" if USE_SHIELD else "standard"
    run_name = f"{mode_name}_{timestamp}"

    # TensorBoard Writer
    log_dir = f"./runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📁 Logging training data to: {log_dir}")

    # Directories
    os.makedirs("./data/models", exist_ok=True)
    model_filename = f"./data/models/ppo_{mode_name}.pth"

    # Environment Setup
    num_lasers = 240
    env_config = {
        "use_render": RENDER,
        "manual_control": False,
        "traffic_density": 0.10,
        "num_scenarios": 1,
        "start_seed": 42,
        "out_of_road_penalty": 10.0,
        "crash_vehicle_penalty": 10.0,
        "crash_object_penalty": 10.0,
        "vehicle_config": {
            "lidar": {"num_lasers": num_lasers, "distance": 50, "num_others": 0}
        },
    }

    env_raw = MetaDriveEnv(env_config)

    if USE_SHIELD:
        print("🛡️  Training WITH Repulsive Safety Shield")
        env = SafetyShieldWrapper(env_raw, lidar_threshold=0.25, num_lasers=num_lasers)
    else:
        print("⚠️  Training WITHOUT Safety Shield (Standard PPO)")
        env = env_raw

    # Agent Setup
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, lr=LR)

    memory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}
    timestep = 0

    print(f"Starting training for '{run_name}'...")

    try:
        for episode in range(1, MAX_EPISODES + 1):
            obs, _ = env.reset()
            episode_reward = 0
            ep_shield_activations = 0

            for step in range(MAX_STEPS):
                timestep += 1

                # 1. Action Selection
                action, log_prob, _ = agent.select_action(obs)

                # 2. Environment Step
                next_obs, reward, done, truncated, info = env.step(action)

                # Track Shield usage (will be 0 if shield is disabled)
                if info.get("shield_activated", False):
                    ep_shield_activations += 1

                # Store in memory
                memory["states"].append(obs)
                memory["actions"].append(action)
                memory["log_probs"].append(log_prob)
                memory["rewards"].append(reward)
                memory["dones"].append(done or truncated)

                obs = next_obs
                episode_reward += reward

                # 3. PPO Update
                if timestep % UPDATE_TIMESTEP == 0:
                    agent.update(memory)
                    for key in memory:
                        memory[key] = []

                # 4. Episode End & Logging
                if done or truncated:
                    outcome = 0
                    if info.get("crash_vehicle", False):
                        outcome = 1
                    elif info.get("crash_object", False):
                        outcome = 2
                    elif info.get("out_of_road", False):
                        outcome = 3
                    elif info.get("arrive_dest", False):
                        outcome = 4

                    # Write to TensorBoard
                    writer.add_scalar("Reward/Episode", episode_reward, episode)
                    writer.add_scalar(
                        "Safety/Shield_Activations", ep_shield_activations, episode
                    )
                    writer.add_scalar("Training/Episode_Length", step, episode)
                    writer.add_scalar("Outcome/Type", outcome, episode)
                    writer.flush()  # Force write to disk
                    break

            # Console Logging
            if episode % 10 == 0:
                print(
                    f"Ep {episode}/{MAX_EPISODES} | R: {episode_reward:.2f} | Shield: {ep_shield_activations}"
                )

            # Save Model
            if episode % 100 == 0:
                agent.save(model_filename)

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()
        writer.close()
        print("Training finished.")


if __name__ == "__main__":
    train()
