# config.py

ENV_NAME = "CliffWalking-v1"

NUM_EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.99

EPSILON_START = 0.1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995  # used only if decay enabled

DECAY_EPSILON = False  # change to True for part (3)