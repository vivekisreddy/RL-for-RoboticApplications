"""
DDPG (Deep Deterministic Policy Gradient) for LunarLanderContinuous-v2
========================================================================
Silver et al. (2014) + Lillicrap et al. (2015).
Key features:
  - Actor  μ(s|θ^μ)  — deterministic policy
  - Critic Q(s,a|θ^Q) — action-value function
  - Target networks for both (soft update: τ)
  - Ornstein-Uhlenbeck noise for exploration
  - Experience replay buffer (same as DQN)
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
import copy

# ─────────────────────────────────────────────
# 1. Hyperparameters  (tune these!)
# ─────────────────────────────────────────────
HYPERPARAMS = {
    "env_name": "LunarLanderContinuous-v3",
    "seed":            42,
    "total_steps":     500_000,
    "batch_size":      128,
    "buffer_size":     200_000,
    "gamma":           0.99,          # discount factor
    "tau":             0.005,         # soft target-update rate
    "actor_lr":        1e-4,          # actor (policy) learning rate
    "critic_lr":       3e-4,          # critic (value) learning rate — often higher
    "hidden_size":     400,           # neurons per hidden layer
    "learning_starts": 10_000,        # warm-up steps (random actions)
    "train_freq":      1,             # update every step
    "grad_clip":       1.0,
    # OU noise parameters
    "ou_mu":           0.0,
    "ou_theta":        0.15,          # mean-reversion speed
    "ou_sigma":        0.2,           # noise magnitude
    "ou_sigma_end":    0.05,          # noise decays to this
    "ou_decay_steps":  200_000,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 2. Actor Network  μ(s) → a  ∈ [-1, 1]^n
# ─────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, act_limit: float):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),   # output in (-1,1)
        )
        # Weight initialisation: small final layer → stable initial actions
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-2].bias,   -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.act_limit


# ─────────────────────────────────────────────
# 3. Critic Network  Q(s, a) → scalar
# ─────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        # Concatenate (s, a) at the first layer
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),             nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias,   -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


# ─────────────────────────────────────────────
# 4. Ornstein-Uhlenbeck Noise
#    (temporally correlated — better than white
#     noise for continuous action exploration)
# ─────────────────────────────────────────────
class OUNoise:
    def __init__(self, size: int, mu=0.0, theta=0.15, sigma=0.2):
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size  = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()


# ─────────────────────────────────────────────
# 5. Replay Buffer  (identical to DQN version)
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
            torch.FloatTensor(np.array(act)).to(DEVICE),
            torch.FloatTensor(rew).to(DEVICE).unsqueeze(1),
            torch.FloatTensor(np.array(nobs)).to(DEVICE),
            torch.FloatTensor(done).to(DEVICE).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────
# 6. Soft target-network update
#    θ_target ← τ·θ + (1-τ)·θ_target
# ─────────────────────────────────────────────
def soft_update(net, target_net, tau: float):
    for p, tp in zip(net.parameters(), target_net.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ─────────────────────────────────────────────
# 7. DDPG update step
# ─────────────────────────────────────────────
def update(actor, actor_target,
           critic, critic_target,
           actor_opt, critic_opt,
           buffer, hp):

    obs, acts, rews, nobs, dones = buffer.sample(hp["batch_size"])

    # ── Critic update ──────────────────────────────────────────────────
    with torch.no_grad():
        next_acts   = actor_target(nobs)
        target_q    = rews + hp["gamma"] * critic_target(nobs, next_acts) * (1.0 - dones)

    current_q = critic(obs, acts)
    critic_loss = nn.MSELoss()(current_q, target_q)

    critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), hp["grad_clip"])
    critic_opt.step()

    # ── Actor update  (maximise Q by gradient ascent on μ) ────────────
    actor_loss = -critic(obs, actor(obs)).mean()   # negative → gradient ascent

    actor_opt.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), hp["grad_clip"])
    actor_opt.step()

    # ── Soft target updates ────────────────────────────────────────────
    soft_update(actor,  actor_target,  hp["tau"])
    soft_update(critic, critic_target, hp["tau"])

    return actor_loss.item(), critic_loss.item()


# ─────────────────────────────────────────────
# 8. Training loop
# ─────────────────────────────────────────────
def get_ou_sigma(step: int, hp: dict) -> float:
    """Linearly decay OU sigma for less exploration over time."""
    fraction = min(1.0, step / hp["ou_decay_steps"])
    return hp["ou_sigma"] + fraction * (hp["ou_sigma_end"] - hp["ou_sigma"])


def train(hp: dict = HYPERPARAMS, render: bool = False):
    render_mode = "human" if render else None
    env = gym.make(hp["env_name"], render_mode=render_mode)

    random.seed(hp["seed"])
    np.random.seed(hp["seed"])
    torch.manual_seed(hp["seed"])
    env.action_space.seed(hp["seed"])

    obs_dim   = env.observation_space.shape[0]   # 8 for LunarLanderContinuous
    act_dim   = env.action_space.shape[0]         # 2
    act_limit = float(env.action_space.high[0])   # 1.0

    actor        = Actor(obs_dim, act_dim, hp["hidden_size"], act_limit).to(DEVICE)
    actor_target = copy.deepcopy(actor).to(DEVICE)
    actor_target.eval()

    critic        = Critic(obs_dim, act_dim, hp["hidden_size"]).to(DEVICE)
    critic_target = copy.deepcopy(critic).to(DEVICE)
    critic_target.eval()

    actor_opt  = optim.Adam(actor.parameters(),  lr=hp["actor_lr"])
    critic_opt = optim.Adam(critic.parameters(), lr=hp["critic_lr"])

    buffer = ReplayBuffer(hp["buffer_size"])
    noise  = OUNoise(act_dim, hp["ou_mu"], hp["ou_theta"], hp["ou_sigma"])
    writer = SummaryWriter(log_dir="runs/DDPG_LunarLanderContinuous")

    obs, _ = env.reset(seed=hp["seed"])
    noise.reset()
    episode_reward   = 0.0
    episode_num      = 0
    actor_losses_buf  = []
    critic_losses_buf = []

    for step in range(1, hp["total_steps"] + 1):
        sigma = get_ou_sigma(step, hp)
        noise.sigma = sigma

        if step < hp["learning_starts"]:
            # Random warm-up
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                action = actor(obs_t).cpu().numpy()[0]
            action = np.clip(action + noise.sample(),
                             env.action_space.low, env.action_space.high)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        episode_reward += reward

        # ── Learn ──────────────────────────────────────────────────────
        if len(buffer) >= hp["learning_starts"] and step % hp["train_freq"] == 0:
            al, cl = update(actor, actor_target,
                            critic, critic_target,
                            actor_opt, critic_opt,
                            buffer, hp)
            actor_losses_buf.append(al)
            critic_losses_buf.append(cl)

        # ── Episode bookkeeping ────────────────────────────────────────
        if done:
            episode_num += 1
            avg_al = np.mean(actor_losses_buf)  if actor_losses_buf  else 0.0
            avg_cl = np.mean(critic_losses_buf) if critic_losses_buf else 0.0

            writer.add_scalar("train/episode_reward", episode_reward, step)
            writer.add_scalar("train/actor_loss",     avg_al,         step)
            writer.add_scalar("train/critic_loss",    avg_cl,         step)
            writer.add_scalar("train/ou_sigma",       sigma,          step)

            if episode_num % 10 == 0:
                print(f"Step {step:>7d} | Ep {episode_num:>4d} | "
                      f"Reward {episode_reward:>8.2f} | σ={sigma:.3f} | "
                      f"A-loss={avg_al:.4f} | C-loss={avg_cl:.4f}")

            obs, _ = env.reset()
            noise.reset()
            episode_reward    = 0.0
            actor_losses_buf  = []
            critic_losses_buf = []

    env.close()
    writer.close()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(actor.state_dict(),  "checkpoints/ddpg_actor_lunar.pth")
    torch.save(critic.state_dict(), "checkpoints/ddpg_critic_lunar.pth")
    print("Training complete. Weights saved to checkpoints/")
    return actor


# ─────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────
def evaluate(actor, n_episodes: int = 10, render: bool = True):
    render_mode = "human" if render else None
    env = gym.make(HYPERPARAMS["env_name"], render_mode=render_mode)
    act_limit = float(env.action_space.high[0])
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                action = actor(obs_t).cpu().numpy()[0]
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, r, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            total += r
        rewards.append(total)
        print(f"  Test ep {ep+1}: {total:.2f}")
    env.close()
    print(f"\nAverage reward over {n_episodes} episodes: {np.mean(rewards):.2f}")
    return rewards


# ─────────────────────────────────────────────
# 10. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    trained_actor = train()
    print("\n--- Evaluation ---")
    evaluate(trained_actor, n_episodes=10, render=False)