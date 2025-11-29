import argparse
from collections import deque
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from metadrive.envs.metadrive_env import MetaDriveEnv
from src.safety_shield import SafetyShieldWrapper
from src.ppo_agent import PPOAgent
from src.reward_shaper import RewardShapingWrapper


def get_args():
    parser = argparse.ArgumentParser(description="PPO Training with Safety Shield")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ppo_model",
        help="Base name for the model file",
    )
    parser.add_argument(
        "--lidar_threshold",
        type=float,
        default=0.10,
        help="Lidar threshold for shield activation",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate for PPO"
    )
    parser.add_argument(
        "--no_shield",
        action="store_true",
        help="Disable the safety shield (Standard PPO)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=2000,
        help="Maximum number of training episodes",
    )
    parser.add_argument(
        "--smoothness_weight",
        type=float,
        default=0.5,
        help="Penalty factor for steering changes (anti-zigzag)",
    )
    parser.add_argument(
        "--speed_weight",
        type=float,
        default=0.05,
        help="Reward bonus factor for velocity",
    )
    parser.add_argument(
        "--centering_weight",
        type=float,
        default=0.5,
        help="Penalty for keeping steering wheel turned (centering)",
    )
    parser.add_argument(
        "--action_smoothing",
        type=float,
        default=0.0,
        help="Exponential moving average factor for steering (0.0 to 1.0). 0.0 disables it.",
    )
    parser.add_argument(
        "--traffic_density",
        type=float,
        default=0.1,
        help="Density of traffic in MetaDrive",
    )
    return parser.parse_args()


def train():
    args = get_args()

    MAX_EPISODES = args.max_episodes
    MAX_STEPS = 1000
    UPDATE_TIMESTEP = 2000
    LR = args.lr
    USE_SHIELD = not args.no_shield
    LIDAR_THRESHOLD = args.lidar_threshold
    TRAFFIC_DENSITY = args.traffic_density
    RENDER = False

    SPEED_WEIGHT = args.speed_weight
    SMOOTHNESS_WEIGHT = args.smoothness_weight
    ACTION_SMOOTHING = args.action_smoothing
    CENTERING_WEIGHT = args.centering_weight

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    shield_status = "shielded" if USE_SHIELD else "standard"
    run_name = f"{args.model_name}_{shield_status}_{timestamp}"

    log_dir = f"./runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging training data to: {log_dir}")
    print(f"Params: LR={LR}, Shield={USE_SHIELD}, Threshold={LIDAR_THRESHOLD}")

    os.makedirs("./data/models", exist_ok=True)
    final_model_name = f"{args.model_name}_{shield_status}.pth"
    model_filename = f"./data/models/{final_model_name}"

    reward_window = deque(maxlen=100)
    success_window = deque(maxlen=100)

    num_lasers = 240
    env_config = {
        "use_render": RENDER,
        "manual_control": False,
        "traffic_density": TRAFFIC_DENSITY,
        "num_scenarios": 1,
        "map": "SSSSSS",
        "start_seed": 42,
        "out_of_road_penalty": 10.0,
        "crash_vehicle_penalty": 10.0,
        "crash_object_penalty": 10.0,
        "success_reward": 30.0,
        "vehicle_config": {
            "lidar": {"num_lasers": num_lasers, "distance": 50, "num_others": 0}
        },
    }

    env = RewardShapingWrapper(
        env=MetaDriveEnv(env_config),
        speed_weight=SPEED_WEIGHT,
        smoothness_weight=SMOOTHNESS_WEIGHT,
        centering_weight=CENTERING_WEIGHT,
        action_smoothing=ACTION_SMOOTHING,
    )

    if USE_SHIELD:
        print(f"Training WITH Safety Shield (Threshold: {LIDAR_THRESHOLD})")
        env = SafetyShieldWrapper(
            env, num_lasers=num_lasers, lidar_threshold=LIDAR_THRESHOLD
        )
    else:
        print("Training WITHOUT Safety Shield")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, lr=LR)

    memory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}
    timestep = 0

    print(f"Starting training for '{run_name}'...")

    try:
        for episode in range(1, MAX_EPISODES + 1):
            progress = (episode - 1) / MAX_EPISODES
            new_lr = LR * (1.0 - progress)
            new_lr = max(new_lr, 1e-6)
            agent.set_lr(new_lr)

            obs, _ = env.reset()
            episode_reward = 0
            ep_shield_activations = 0

            for step in range(MAX_STEPS):
                timestep += 1

                action, log_prob, _ = agent.select_action(obs)

                next_obs, reward, done, truncated, info = env.step(action)

                if info.get("shield_activated", False):
                    ep_shield_activations += 1

                memory["states"].append(obs)
                memory["actions"].append(action)
                memory["log_probs"].append(log_prob)
                memory["rewards"].append(reward)
                memory["dones"].append(done or truncated)

                obs = next_obs
                episode_reward += reward

                if timestep % UPDATE_TIMESTEP == 0:
                    train_metrics = agent.update(memory)

                    writer.add_scalar(
                        "Loss/Policy_Loss", train_metrics["policy_loss"], episode
                    )
                    writer.add_scalar(
                        "Loss/Value_Loss", train_metrics["value_loss"], episode
                    )
                    writer.add_scalar(
                        "Training/Entropy", train_metrics["entropy"], episode
                    )
                    writer.add_scalar(
                        "Training/Approx_KL", train_metrics["approx_kl"], episode
                    )

                    for key in memory:
                        memory[key] = []

                if done or truncated:
                    outcome = 0
                    is_success = 0

                    if info.get("crash_vehicle", False):
                        outcome = 1
                    elif info.get("crash_object", False):
                        outcome = 2
                    elif info.get("out_of_road", False):
                        outcome = 3
                    elif info.get("arrive_dest", False):
                        outcome = 4
                        is_success = 1

                    reward_window.append(episode_reward)
                    success_window.append(is_success)

                    avg_reward_100 = np.mean(reward_window)
                    success_rate = np.mean(success_window)

                    writer.add_scalar("Reward/Raw_Episode", episode_reward, episode)
                    writer.add_scalar(
                        "Reward/Average_100_Episodes", avg_reward_100, episode
                    )
                    writer.add_scalar("Training/Success_Rate", success_rate, episode)
                    writer.add_scalar(
                        "Safety/Shield_Activations", ep_shield_activations, episode
                    )
                    writer.add_scalar("Training/Episode_Length", step, episode)
                    writer.add_scalar("Outcome/Type", outcome, episode)
                    writer.add_scalar("Training/Learning_Rate", new_lr, episode)
                    writer.flush()
                    break

            if episode % 10 == 0:
                print(
                    f"Ep {episode} | R: {episode_reward:.1f} | Avg100: {avg_reward_100:.1f} | SuccessRate: {success_rate:.2f} | Out: {outcome}"
                )

            if episode % 100 == 0:
                agent.save(model_filename)

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()
        writer.close()
        print("\n" + "=" * 50)
        print("TRAINING FINISHED")
        print("To evaluate this model, run the following command:")

        eval_cmd = f"python main_eval.py --model_name {final_model_name} --lidar_threshold {LIDAR_THRESHOLD} --action_smoothing {ACTION_SMOOTHING}"
        if not USE_SHIELD:
            eval_cmd += " --no_shield"

        print(f"\n👉 {eval_cmd}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    train()
