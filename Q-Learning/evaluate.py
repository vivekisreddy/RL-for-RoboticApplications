import gymnasium as gym
import pickle
import numpy as np
import time
import sys

method = sys.argv[1]  # sarsa or q_learning

env = gym.make("CliffWalking-v1", render_mode="human")

q_table = pickle.load(open(f"{method}_q_table.pkl", "rb"))

state, _ = env.reset()
done = False

print(f"Evaluating {method.upper()} policy")

while not done:
    time.sleep(0.3)

    action = np.argmax(q_table[state])

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    state = next_state

env.close()