import numpy as np
import random
import matplotlib.pyplot as plt


## Environment for Cliff Walking Problem
class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state

        # Actions: 0=up,1=down,2=left,3=right
        if action == 0:
            r -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c -= 1
        elif action == 3:
            c += 1

        # Boundary conditions
        r = max(0, min(self.rows - 1, r))
        c = max(0, min(self.cols - 1, c))

        next_state = (r, c)

        reward = -1
        done = False

        # Check cliff
        if r == 3 and 1 <= c <= 10:
            reward = -100
            next_state = self.start

        if next_state == self.goal:
            done = True

        self.state = next_state
        return next_state, reward, done


# Epsilon-greedy action selection
def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(Q[state])


# Q-learning algorithm
def q_learning(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((env.rows, env.cols, 4))
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env.step(action)

            r, c = state
            nr, nc = next_state

            Q[r, c, action] += alpha * (
                reward + gamma * np.max(Q[nr, nc]) - Q[r, c, action]
            )

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode


# Sarsa algorithm
def sarsa(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    Q = np.zeros((env.rows, env.cols, 4))
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            r, c = state
            nr, nc = next_state

            Q[r, c, action] += alpha * (
                reward + gamma * Q[nr, nc, next_action] - Q[r, c, action]
            )

            state = next_state
            action = next_action
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode


def plot_cliff_grid_with_paths(env):
    rows, cols = env.rows, env.cols

    # Build grid: 0 = normal, 1 = cliff
    grid = np.zeros((rows, cols))
    grid[3, 1:11] = 1  # cliff cells

    fig, ax = plt.subplots(figsize=(8, 3))

    # Show grid
    # use origin='upper' so (0,0) is top-left like our row/col
    cmap = plt.cm.get_cmap("Greys", 2)
    im = ax.imshow(grid, cmap=cmap, origin="upper")

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

    # Label axes
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticklabels(np.arange(rows))

    # Mark start, goal, and cliff
    sr, sc = env.start
    gr, gc = env.goal

    ax.text(sc, sr, "S", ha="center", va="center", color="green", fontsize=12, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center", color="gold", fontsize=12, fontweight="bold")

    for c in range(1, 11):
        ax.text(c, 3, "C", ha="center", va="center", color="black", fontsize=10)

    # Define example paths as sequences of (row, col)
    # Optimal (greedy) path: straight along the cliff top row = 3, cols 0..11
    optimal_path = [(3, c) for c in range(0, 12)]

    # Safest path: go up one row, go across, then down to goal
    # S(3,0) -> (2,0) -> ... -> (2,11) -> (3,11)
    safest_path = [(3, 0)] + [(2, c) for c in range(0, 12)] + [(3, 11)]

    # Convert to x,y for plotting (x = col, y = row)
    opt_x = [c for r, c in optimal_path]
    opt_y = [r for r, c in optimal_path]

    safe_x = [c for r, c in safest_path]
    safe_y = [r for r, c in safest_path]

    # Plot paths
    ax.plot(opt_x, opt_y, color="red", linewidth=2, label="Optimal path")
    ax.plot(safe_x, safe_y, color="blue", linewidth=2, label="Safest path")

    # Put start/end markers on paths
    ax.scatter([sc], [sr], color="green", s=60)
    ax.scatter([gc], [gr], color="gold", s=60)

    ax.invert_yaxis()  # to keep row 0 at top visually
    ax.set_title("Cliff Walking Grid: Optimal (red) vs Safest (blue)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# Compare Q-learning and Sarsa
if __name__ == "__main__":
    env = CliffWalkingEnv()

    Q_q, rewards_q = q_learning(env)
    Q_s, rewards_s = sarsa(env)

    plt.figure()
    plt.plot(rewards_q, label="Q-learning")
    plt.plot(rewards_s, label="Sarsa")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot cliff grid with two example paths
    plot_cliff_grid_with_paths(env)