# plot_results.py

import matplotlib.pyplot as plt
from train import train

q_sarsa, rewards_sarsa = train("sarsa")
q_q, rewards_q = train("q_learning")

plt.figure(figsize=(10, 5))

plt.plot(rewards_sarsa, label="SARSA", color="blue")
plt.plot(rewards_q, label="Q-Learning", color="red")

plt.xlabel("Episodes")
plt.ylabel("Sum of Rewards")
plt.title("Cliff Walking: SARSA vs Q-Learning")
plt.legend()

plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

plt.show()