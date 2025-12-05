<<<<<<< HEAD
"""
Main script for training DQN on CartPole with custom reward,
full-training video recording, and all evaluation plots.
"""

from __future__ import annotations

import os
from typing import List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from custom_env import AnglePenaltyWrapper
from dqn_agent import DQNAgent
from video import VideoRecorder

# ============================================================
# Output directory (auto-created)
# ============================================================

OUTPUT_DIR = r"C:\Users\PC\Desktop\AbrarAqeel\RLProject\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Configuration (all in one place)
# ============================================================

ENV_ID = "CartPole-v1"
ANGLE_PENALTY_WEIGHT = 0.10
SEED = 42

HIDDEN_SIZE = 128
LEARNING_RATE = 3e-4
GAMMA = 0.98
BUFFER_SIZE = 50000
BATCH_SIZE = 64

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 10000

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 500
TARGET_UPDATE_INTERVAL = 500
PRINT_EVERY = 10

VIDEO_PATH = os.path.join(OUTPUT_DIR, "cartpole_training.mp4")
VIDEO_FPS = 30

WATERMARK = "Abrar Aqeel"


# ============================================================
# Plotting Functions
# ============================================================

def plot_rewards(rewards: List[float]):
    episodes = np.arange(1, len(rewards) + 1)
    rewards_np = np.array(rewards, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards_np, label="Reward per Episode")

    if len(rewards_np) >= 20:
        rolling = np.convolve(rewards_np, np.ones(20) / 20, mode="valid")
        plt.plot(np.arange(20, len(rewards_np) + 1), rolling, label="Rolling Mean (20)")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN CartPole Training Rewards")
    plt.legend()

    # Watermark
    plt.text(
        0.5, 0.02, WATERMARK, fontsize=10, color="gray",
        transform=plt.gca().transAxes, alpha=0.6, ha="center"
    )

    out = os.path.join(OUTPUT_DIR, "reward_curve.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[PLOT] Saved reward_curve.png to {out}")


def plot_success_rate(rewards: List[float], threshold: float = 200.0):
    successes = [1 if r >= threshold else 0 for r in rewards]
    successes = np.array(successes)

    if len(successes) >= 20:
        rate = np.convolve(successes, np.ones(20) / 20, mode="valid")
    else:
        rate = successes.astype(float)

    plt.figure(figsize=(10, 5))
    plt.plot(rate, label="Success Rate (window=20)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.title("Success Rate Over Training")
    plt.legend()

    plt.text(
        0.5, 0.02, WATERMARK, fontsize=10, color="gray",
        transform=plt.gca().transAxes, alpha=0.6, ha="center"
    )

    out = os.path.join(OUTPUT_DIR, "success_rate.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[PLOT] Saved success_rate.png to {out}")


def plot_learning_curve(rewards: List[float]):
    rewards_np = np.array(rewards, dtype=float)

    if len(rewards_np) >= 20:
        rolling = np.convolve(rewards_np, np.ones(20) / 20, mode="valid")
    else:
        rolling = rewards_np

    plt.figure(figsize=(10, 5))
    plt.plot(rolling, label="Rolling Mean Reward")

    plt.xlabel("Episode")
    plt.ylabel("Rolling Mean Reward")
    plt.title("Learning Curve")
    plt.legend()

    plt.text(
        0.5, 0.02, WATERMARK, fontsize=10, color="gray",
        transform=plt.gca().transAxes, alpha=0.6, ha="center"
    )

    out = os.path.join(OUTPUT_DIR, "learning_curve.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[PLOT] Saved learning_curve.png to {out}")


# ============================================================
# Training Function
# ============================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Gym env with RGB frames for video
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = AnglePenaltyWrapper(env, angle_penalty_weight=ANGLE_PENALTY_WEIGHT)

    obs, _ = env.reset(seed=SEED)
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        device=device
    )

    recorder = VideoRecorder(VIDEO_PATH, VIDEO_FPS)

    rewards = []
    global_step = 0

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            agent.total_steps = global_step
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            ep_reward += reward
            global_step += 1

            if global_step % TARGET_UPDATE_INTERVAL == 0:
                agent.update_target()

            frame = env.render()
            recorder.write(frame)

            if done:
                break

        rewards.append(ep_reward)

        if ep % PRINT_EVERY == 0:
            avg = np.mean(rewards[-PRINT_EVERY:])
            print(f"[TRAIN] Episode {ep}/{NUM_EPISODES}  Reward={ep_reward:.1f}  Avg={avg:.1f}")

    env.close()
    recorder.close()

    # Save plots
    plot_rewards(rewards)
    plot_success_rate(rewards)
    plot_learning_curve(rewards)

    # Save model weights
    torch.save(agent.policy.state_dict(), os.path.join(OUTPUT_DIR, "dqn_cartpole.pth"))

    print("[INFO] Saved model to outputs directory.")


if __name__ == "__main__":
    train()
=======
"""
Main script for training DQN on CartPole with custom reward,
full-training video recording, and all evaluation plots.
"""

from __future__ import annotations

import os
from typing import List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from custom_env import AnglePenaltyWrapper
from dqn_agent import DQNAgent
from video import VideoRecorder

# ============================================================
# Output directory (auto-created)
# ============================================================

OUTPUT_DIR = r"C:\Users\PC\Desktop\AbrarAqeel\RLProject\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Configuration (all in one place)
# ============================================================

ENV_ID = "CartPole-v1"
ANGLE_PENALTY_WEIGHT = 0.10
SEED = 42

HIDDEN_SIZE = 128
LEARNING_RATE = 3e-4
GAMMA = 0.98
BUFFER_SIZE = 50000
BATCH_SIZE = 64

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 10000

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 500
TARGET_UPDATE_INTERVAL = 500
PRINT_EVERY = 10

VIDEO_PATH = os.path.join(OUTPUT_DIR, "cartpole_training.mp4")
VIDEO_FPS = 30

WATERMARK = "Abrar Aqeel (22108103) | Abubakar Tariq (22108104)"


# ============================================================
# Plotting Functions
# ============================================================

def plot_rewards(rewards: List[float]):
    episodes = np.arange(1, len(rewards) + 1)
    rewards_np = np.array(rewards, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards_np, label="Reward per Episode")

    if len(rewards_np) >= 20:
        rolling = np.convolve(rewards_np, np.ones(20) / 20, mode="valid")
        plt.plot(np.arange(20, len(rewards_np) + 1), rolling, label="Rolling Mean (20)")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN CartPole Training Rewards")
    plt.legend()

    # Watermark
    plt.text(
        0.5, 0.02, WATERMARK, fontsize=10, color="gray",
        transform=plt.gca().transAxes, alpha=0.6, ha="center"
    )

    out = os.path.join(OUTPUT_DIR, "reward_curve.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[PLOT] Saved reward_curve.png to {out}")


def plot_success_rate(rewards: List[float], threshold: float = 200.0):
    successes = [1 if r >= threshold else 0 for r in rewards]
    successes = np.array(successes)

    if len(successes) >= 20:
        rate = np.convolve(successes, np.ones(20) / 20, mode="valid")
    else:
        rate = successes.astype(float)

    plt.figure(figsize=(10, 5))
    plt.plot(rate, label="Success Rate (window=20)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.title("Success Rate Over Training")
    plt.legend()

    plt.text(
        0.5, 0.02, WATERMARK, fontsize=10, color="gray",
        transform=plt.gca().transAxes, alpha=0.6, ha="center"
    )

    out = os.path.join(OUTPUT_DIR, "success_rate.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[PLOT] Saved success_rate.png to {out}")


def plot_learning_curve(rewards: List[float]):
    rewards_np = np.array(rewards, dtype=float)

    if len(rewards_np) >= 20:
        rolling = np.convolve(rewards_np, np.ones(20) / 20, mode="valid")
    else:
        rolling = rewards_np

    plt.figure(figsize=(10, 5))
    plt.plot(rolling, label="Rolling Mean Reward")

    plt.xlabel("Episode")
    plt.ylabel("Rolling Mean Reward")
    plt.title("Learning Curve")
    plt.legend()

    plt.text(
        0.5, 0.02, WATERMARK, fontsize=10, color="gray",
        transform=plt.gca().transAxes, alpha=0.6, ha="center"
    )

    out = os.path.join(OUTPUT_DIR, "learning_curve.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[PLOT] Saved learning_curve.png to {out}")


# ============================================================
# Training Function
# ============================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Gym env with RGB frames for video
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = AnglePenaltyWrapper(env, angle_penalty_weight=ANGLE_PENALTY_WEIGHT)

    obs, _ = env.reset(seed=SEED)
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        device=device
    )

    recorder = VideoRecorder(VIDEO_PATH, VIDEO_FPS)

    rewards = []
    global_step = 0

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            agent.total_steps = global_step
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            ep_reward += reward
            global_step += 1

            if global_step % TARGET_UPDATE_INTERVAL == 0:
                agent.update_target()

            frame = env.render()
            recorder.write(frame)

            if done:
                break

        rewards.append(ep_reward)

        if ep % PRINT_EVERY == 0:
            avg = np.mean(rewards[-PRINT_EVERY:])
            print(f"[TRAIN] Episode {ep}/{NUM_EPISODES}  Reward={ep_reward:.1f}  Avg={avg:.1f}")

    env.close()
    recorder.close()

    # Save plots
    plot_rewards(rewards)
    plot_success_rate(rewards)
    plot_learning_curve(rewards)

    # Save model weights
    torch.save(agent.policy.state_dict(), os.path.join(OUTPUT_DIR, "dqn_cartpole.pth"))

    print("[INFO] Saved model to outputs directory.")


if __name__ == "__main__":
    train()

>>>>>>> 00debab322f4b7ccc48e1f27f50b6ef94360d3dc
