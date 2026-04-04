import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
from collections import deque
from PIL import Image

# =========================
# 1. Custom Environment
# =========================

class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.state = None

    def reset(self):
        self.state = np.array([0.0], dtype=np.float32)
        return self.state

    def step(self, action):
        if action == 0:
            self.state += 1
        else:
            self.state -= 1

        reward = 1.0 if self.state[0] >= 5 else -0.1
        done = abs(self.state[0]) >= 10

        return self.state, reward, done, {}

    def render(self, mode="rgb_array"):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        pos = int(100 + self.state[0] * 5)
        pos = np.clip(pos, 0, 199)
        img[:, pos:pos+5] = [255, 0, 0]
        return img


# =========================
# 2. DQN Network
# =========================

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 3. Training Setup
# =========================

env = SimpleEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = deque(maxlen=10000)

gamma = 0.99
batch_size = 64
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
episodes = 200

episode_rewards = []

# =========================
# 4. Training Loop
# =========================

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            q_values = policy_net(states)
            next_q_values = target_net(next_states)

            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            next_q_value = torch.max(next_q_values, dim=1)[0]

            expected_q_value = rewards + gamma * next_q_value * (1 - dones)

            loss = nn.MSELoss()(q_value, expected_q_value.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    episode_rewards.append(total_reward)

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode} | Reward: {total_reward:.2f}")

# =========================
# 5. Save Reward Plot
# =========================

plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Rewards")
plt.savefig("training_rewards.png")
plt.close()

print("Saved training_rewards.png")

# =========================
# 6. Save Trained Model
# =========================

torch.save(policy_net.state_dict(), "dqn_model.pth")
print("Saved dqn_model.pth")

# =========================
# 7. Generate GIF
# =========================

frames = []
state = env.reset()
done = False

while not done:
    frame = env.render()
    frames.append(frame)

    state_tensor = torch.FloatTensor(state).to(device)
    action = torch.argmax(policy_net(state_tensor)).item()
    state, _, done, _ = env.step(action)

imageio.mimsave("agent_run.gif", frames, fps=10)
print("Saved agent_run.gif")