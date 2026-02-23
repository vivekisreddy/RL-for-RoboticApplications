import pickle
import matplotlib.pyplot as plt

# Load rewards
with open("sarsa_rewards.pkl", "rb") as f:
    rewards_sarsa = pickle.load(f)
with open("q_learning_rewards.pkl", "rb") as f:
    rewards_q = pickle.load(f)

plt.figure(figsize=(10, 5))
plt.plot(rewards_sarsa, label="SARSA", color="blue")
plt.plot(rewards_q, label="Q-Learning", color="red")
plt.xlabel("Episodes")
plt.ylabel("Sum of Rewards")
plt.title("Cliff Walking: SARSA vs Q-Learning")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)

# Save the plot
plt.savefig("cliffwalking_rewards.png", dpi=300)
print("Saved plot as cliffwalking_rewards.png")

plt.show()