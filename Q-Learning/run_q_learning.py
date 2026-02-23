# run_q_learning.py

from train import train

if __name__ == "__main__":
    q_table, rewards = train(method="q_learning")