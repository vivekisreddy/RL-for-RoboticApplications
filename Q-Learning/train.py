import gymnasium as gym
import numpy as np
import pickle
from utils import epsilon_greedy, decay_epsilon
import config

def train(method="sarsa"):
    env = gym.make(config.ENV_NAME)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    rewards_per_episode = []
    epsilon = config.EPSILON_START

    for episode in range(config.NUM_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        if method == "sarsa":
            action = epsilon_greedy(q_table, state, epsilon)

        while not done:
            if method == "q_learning":
                action = epsilon_greedy(q_table, state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if method == "sarsa":
                next_action = epsilon_greedy(q_table, next_state, epsilon)
                target = reward + config.GAMMA * q_table[next_state][next_action]
            elif method == "q_learning":
                target = reward + config.GAMMA * np.max(q_table[next_state])

            q_table[state][action] += config.ALPHA * (target - q_table[state][action])

            state = next_state
            if method == "sarsa":
                action = next_action

            total_reward += reward

        rewards_per_episode.append(total_reward)

        if config.DECAY_EPSILON:
            epsilon = decay_epsilon(epsilon, config.EPSILON_MIN, config.EPSILON_DECAY)

        print(f"{method.upper()} | Episode {episode} | Reward {total_reward}")

    env.close()

    # Save Q-table
    with open(f"{method}_q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    # Save rewards
    with open(f"{method}_rewards.pkl", "wb") as f:
        pickle.dump(rewards_per_episode, f)

    print(f"Saved {method}_q_table.pkl and {method}_rewards.pkl")

    return q_table, rewards_per_episode

if __name__ == "__main__":
    train("sarsa")
    train("q_learning")