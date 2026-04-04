"""
DQN (Deep Q-Network) for LunarLander-v2
========================================
Based on Mnih et al. (2015) — the same algorithm described in the Sutton & Barto
textbook excerpt. Key features:
  - Experience replay buffer
  - Target network (frozen for C steps)
  - Epsilon-greedy exploration with linear decay
  - TensorBoard logging
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import os

# ─────────────────────────────────────────────
# 1. Hyperparameters  (tune these!)
# ─────────────────────────────────────────────
HYPERPARAMS = {
    "env_name": "LunarLander-v3",
    "seed":              42,
    "total_steps":       500_000,       # training budget
    "batch_size":        64,
    "buffer_size":       100_000,       # replay memory capacity
    "gamma":             0.99,          # discount factor
    "lr":                5e-4,          # Adam learning rate
    "target_update_freq":1_000,         # C: steps between target-net syncs
    "eps_start":         1.0,           # ε at step 0
    "eps_end":           0.01,          # minimum ε
    "eps_decay_steps":   200_000,       # steps over which ε decays linearly
    "learning_starts":   10_000,        # fill buffer before training
    "train_freq":        4,             # update weights every N steps
    "hidden_size":       256,           # neurons per hidden layer
    "grad_clip":         10.0,          # max gradient norm
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 2. Q-Network  (two hidden layers + ReLU)
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 3. Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)).to(DEVICE),
            torch.LongTensor(act).to(DEVICE),
            torch.FloatTensor(rew).to(DEVICE),
            torch.FloatTensor(np.array(nobs)).to(DEVICE),
            torch.FloatTensor(done).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────
# 4. Epsilon-greedy policy
# ─────────────────────────────────────────────
def get_epsilon(step: int, hp: dict) -> float:
    """Linear decay from eps_start → eps_end over eps_decay_steps."""
    fraction = min(1.0, step / hp["eps_decay_steps"])
    return hp["eps_start"] + fraction * (hp["eps_end"] - hp["eps_start"])


def select_action(obs, q_net, epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        return int(q_net(obs_t).argmax(dim=1).item())


# ─────────────────────────────────────────────
# 5. Single Q-learning update step
#    (equation 16.3 from the textbook, with
#     frozen target network ≈ q̃)
# ─────────────────────────────────────────────
def update(q_net, target_net, optimizer, buffer, hp):
    obs, actions, rewards, next_obs, dones = buffer.sample(hp["batch_size"])

    # Current Q-values: Q(s, a)
    q_values = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target: r + γ · max_a' Q̃(s', a')   (0 if terminal)
    with torch.no_grad():
        max_next_q = target_net(next_obs).max(dim=1).values
        targets = rewards + hp["gamma"] * max_next_q * (1.0 - dones)

    loss = nn.SmoothL1Loss()(q_values, targets)   # Huber loss (clips large errors)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), hp["grad_clip"])
    optimizer.step()

    return loss.item()


# ─────────────────────────────────────────────
# 6. Training loop
# ─────────────────────────────────────────────
def train(hp: dict = HYPERPARAMS, render: bool = False):
    render_mode = "human" if render else None
    env = gym.make(hp["env_name"], render_mode=render_mode)

    # Reproducibility
    random.seed(hp["seed"])
    np.random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])
    env.action_space.seed(hp["seed"])

    obs_dim  = env.observation_space.shape[0]   # 8 for LunarLander
    n_actions = env.action_space.n              # 4

    q_net      = QNetwork(obs_dim, n_actions, hp["hidden_size"]).to(DEVICE)
    target_net = QNetwork(obs_dim, n_actions, hp["hidden_size"]).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=hp["lr"])
    buffer    = ReplayBuffer(hp["buffer_size"])
    writer    = SummaryWriter(log_dir="runs/DQN_LunarLander")

    obs, _ = env.reset(seed=hp["seed"])
    episode_reward = 0.0
    episode_num    = 0
    episode_losses = []
    best_avg_reward = -np.inf

    for step in range(1, hp["total_steps"] + 1):
        epsilon = get_epsilon(step, hp)
        action  = select_action(obs, q_net, epsilon, n_actions)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, float(terminated))

        obs = next_obs
        episode_reward += reward

        # ── Learn ──────────────────────────────
        if len(buffer) >= hp["learning_starts"] and step % hp["train_freq"] == 0:
            loss = update(q_net, target_net, optimizer, buffer, hp)
            episode_losses.append(loss)

        # ── Sync target network ─────────────────
        if step % hp["target_update_freq"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ── Episode bookkeeping ─────────────────
        if done:
            episode_num += 1
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            writer.add_scalar("train/episode_reward", episode_reward, step)
            writer.add_scalar("train/epsilon",        epsilon,        step)
            writer.add_scalar("train/loss",           avg_loss,       step)

            if episode_num % 10 == 0:
                print(f"Step {step:>7d} | Ep {episode_num:>4d} | "
                      f"Reward {episode_reward:>8.2f} | ε={epsilon:.3f} | "
                      f"Loss={avg_loss:.4f}")

            obs, _ = env.reset()
            episode_reward = 0.0
            episode_losses = []

    env.close()
    writer.close()

    # Save final weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(q_net.state_dict(), "checkpoints/dqn_lunar.pth")
    print("Training complete. Weights saved to checkpoints/dqn_lunar.pth")
    return q_net


# ─────────────────────────────────────────────
# 7. Evaluation (average over N test episodes)
# ─────────────────────────────────────────────
def evaluate(q_net, n_episodes: int = 10, render: bool = True):
    render_mode = "human" if render else None
    env = gym.make(HYPERPARAMS["env_name"], render_mode=render_mode)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total  = 0.0
        done   = False
        while not done:
            action = select_action(obs, q_net, epsilon=0.0,
                                   n_actions=env.action_space.n)
            obs, r, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            total += r
        rewards.append(total)
        print(f"  Test ep {ep+1}: {total:.2f}")
    env.close()
    print(f"\nAverage reward over {n_episodes} episodes: {np.mean(rewards):.2f}")
    return rewards


# ─────────────────────────────────────────────
# 8. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    trained_net = train()
    print("\n--- Evaluation ---")
    evaluate(trained_net, n_episodes=10, render=False)