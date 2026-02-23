# utils.py

import numpy as np

# Policy + epsilon scheduling 

def epsilon_greedy(q_table, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, 4)
    return np.argmax(q_table[state])


def decay_epsilon(epsilon, epsilon_min, decay_rate):
    return max(epsilon_min, epsilon * decay_rate)