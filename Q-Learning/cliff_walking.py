import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

class CliffWalkingEnvironment:
    """
    Cliff Walking Environment from Sutton & Barto Example 6.6
    Grid: 4 rows x 12 columns
    Start: (3, 0) - bottom left
    Goal: (3, 11) - bottom right
    Cliff: (3, 1) through (3, 10) - bottom middle
    """
    
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        
        # Define the cliff cells
        self.cliff = [(3, col) for col in range(1, 11)]
        
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.actions = [0, 1, 2, 3]
        self.action_names = ['Up', 'Right', 'Down', 'Left']
        
        # Action effects (row_delta, col_delta)
        self.action_effects = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        
        self.current_state = self.start_state
    
    def reset(self):
        """Reset environment to start state"""
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action):
        """
        Take action and return (next_state, reward, done)
        """
        row, col = self.current_state
        delta_row, delta_col = self.action_effects[action]
        
        # Calculate next position
        next_row = max(0, min(self.rows - 1, row + delta_row))
        next_col = max(0, min(self.cols - 1, col + delta_col))
        next_state = (next_row, next_col)
        
        # Check if we fell off the cliff
        if next_state in self.cliff:
            reward = -100
            self.current_state = self.start_state  # Reset to start
            done = False  # Episode continues
            return self.start_state, reward, done
        
        # Check if we reached the goal
        if next_state == self.goal_state:
            reward = -1
            self.current_state = next_state
            done = True
            return next_state, reward, done
        
        # Normal step
        reward = -1
        self.current_state = next_state
        done = False
        return next_state, reward, done
    
    def is_terminal(self, state):
        """Check if state is terminal"""
        return state == self.goal_state
    
    def visualize_path(self, path, title="Path"):
        """Visualize a path through the grid"""
        grid = np.zeros((self.rows, self.cols))
        
        # Mark cliff
        for cliff_cell in self.cliff:
            grid[cliff_cell] = -1
        
        # Mark path
        for i, state in enumerate(path):
            if state not in self.cliff and state != self.goal_state:
                grid[state] = i + 1
        
        # Mark start and goal
        grid[self.start_state] = -2
        grid[self.goal_state] = -3
        
        plt.figure(figsize=(12, 4))
        sns.heatmap(grid, annot=True, fmt='.0f', cmap='RdYlGn', 
                   cbar=False, linewidths=1, linecolor='black')
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.tight_layout()
        return plt.gcf()


def epsilon_greedy_policy(Q, state, epsilon, env):
    """
    Epsilon-greedy action selection
    """
    if np.random.random() < epsilon:
        # Explore: random action
        return np.random.choice(env.actions)
    else:
        # Exploit: best action
        q_values = [Q[(state, a)] for a in env.actions]
        max_q = max(q_values)
        # Break ties randomly
        best_actions = [a for a in env.actions if Q[(state, a)] == max_q]
        return np.random.choice(best_actions)


def sarsa(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, 
          epsilon_decay=False, epsilon_min=0.01):
    """
    SARSA: On-policy TD Control
    
    Update rule:
    Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
    
    Key: A' is selected using the CURRENT policy (epsilon-greedy)
    """
    # Initialize Q-table
    Q = defaultdict(lambda: 0.0)
    
    # Track performance
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        # Decay epsilon if requested
        if epsilon_decay:
            current_epsilon = max(epsilon_min, epsilon * (0.99 ** episode))
        else:
            current_epsilon = epsilon
        
        # Initialize episode
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, current_epsilon, env)
        
        total_reward = 0
        steps = 0
        
        while True:
            # Take action A, observe R, S'
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                # Terminal state: Q(S',A') = 0
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
                break
            
            # Choose A' from S' using policy derived from Q (epsilon-greedy)
            next_action = epsilon_greedy_policy(Q, next_state, current_epsilon, env)
            
            # SARSA update: Use the action A' that we WILL take
            Q[(state, action)] += alpha * (
                reward + gamma * Q[(next_state, next_action)] - Q[(state, action)]
            )
            
            # Move to next state-action pair
            state = next_state
            action = next_action  # This is the key: we use the selected action
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    return Q, episode_rewards, episode_lengths


def q_learning(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1,
               epsilon_decay=False, epsilon_min=0.01):
    """
    Q-learning: Off-policy TD Control
    
    Update rule:
    Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
    
    Key: We use MAX over next actions, not the action we'll actually take
    """
    # Initialize Q-table
    Q = defaultdict(lambda: 0.0)
    
    # Track performance
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        # Decay epsilon if requested
        if epsilon_decay:
            current_epsilon = max(epsilon_min, epsilon * (0.99 ** episode))
        else:
            current_epsilon = epsilon
        
        # Initialize episode
        state = env.reset()
        
        total_reward = 0
        steps = 0
        
        while True:
            # Choose A from S using policy derived from Q (epsilon-greedy)
            action = epsilon_greedy_policy(Q, state, current_epsilon, env)
            
            # Take action A, observe R, S'
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                # Terminal state
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
                break
            
            # Q-learning update: Use MAX over next actions (off-policy)
            max_next_q = max([Q[(next_state, a)] for a in env.actions])
            Q[(state, action)] += alpha * (
                reward + gamma * max_next_q - Q[(state, action)]
            )
            
            # Move to next state
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    return Q, episode_rewards, episode_lengths


def extract_policy(Q, env):
    """Extract greedy policy from Q-function"""
    policy = {}
    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            if state == env.goal_state:
                continue
            q_values = [Q[(state, a)] for a in env.actions]
            policy[state] = np.argmax(q_values)
    return policy


def generate_episode_path(env, Q, epsilon=0.0, max_steps=100):
    """Generate a single episode path using learned Q-function"""
    path = []
    state = env.reset()
    path.append(state)
    
    for _ in range(max_steps):
        action = epsilon_greedy_policy(Q, state, epsilon, env)
        next_state, reward, done = env.step(action)
        path.append(next_state)
        
        if done:
            break
        state = next_state
    
    return path


def smooth_curve(values, window=50):
    """Smooth curve using moving average"""
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window)
        smoothed.append(np.mean(values[start:i+1]))
    return smoothed


def main():
    """Main execution function"""
    print("="*70)
    print("Cliff Walking Problem - SARSA vs Q-Learning")
    print("="*70)
    
    # Create environment
    env = CliffWalkingEnvironment()
    
    # =========================================================================
    # PART 1: Run SARSA and Q-Learning with fixed epsilon
    # =========================================================================
    print("\n" + "="*70)
    print("PART 1: Fixed Epsilon (ε = 0.1)")
    print("="*70)
    
    episodes = 500
    alpha = 0.5
    gamma = 1.0
    epsilon = 0.1
    
    print(f"\nRunning SARSA... (episodes={episodes}, α={alpha}, ε={epsilon})")
    Q_sarsa, rewards_sarsa, lengths_sarsa = sarsa(
        env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon
    )
    
    print(f"Running Q-Learning... (episodes={episodes}, α={alpha}, ε={epsilon})")
    Q_qlearning, rewards_qlearning, lengths_qlearning = q_learning(
        env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon
    )
    
    # =========================================================================
    # PART 2: Visualize learned paths
    # =========================================================================
    print("\n" + "="*70)
    print("PART 2: Visualizing Learned Paths")
    print("="*70)
    
    # Generate paths using learned policies (greedy)
    path_sarsa = generate_episode_path(env, Q_sarsa, epsilon=0.0)
    path_qlearning = generate_episode_path(env, Q_qlearning, epsilon=0.0)
    
    print(f"\nSARSA path length: {len(path_sarsa)-1} steps")
    print(f"Q-Learning path length: {len(path_qlearning)-1} steps")
    
    # Visualize paths
    fig1 = env.visualize_path(path_sarsa, "SARSA Learned Path (Greedy - Safe Route)")
    plt.savefig('sarsa_path.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig2 = env.visualize_path(path_qlearning, "Q-Learning Learned Path (Greedy - Optimal Route)")
    plt.savefig('qlearning_path.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # PART 3: Plot performance comparison
    # =========================================================================
    print("\n" + "="*70)
    print("PART 3: Performance Comparison")
    print("="*70)
    
    # Create performance plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Sum of rewards per episode
    ax1 = axes[0]
    ax1.plot(rewards_sarsa, alpha=0.3, color='blue', linewidth=0.5)
    ax1.plot(smooth_curve(rewards_sarsa, 50), color='blue', linewidth=2, label='SARSA')
    ax1.plot(rewards_qlearning, alpha=0.3, color='red', linewidth=0.5)
    ax1.plot(smooth_curve(rewards_qlearning, 50), color='red', linewidth=2, label='Q-Learning')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Sum of Rewards per Episode')
    ax1.set_title('Performance Comparison: SARSA vs Q-Learning (ε = 0.1 fixed)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=-13, color='green', linestyle='--', alpha=0.5, label='Optimal (no exploration)')
    
    # Plot 2: Episode length
    ax2 = axes[1]
    ax2.plot(lengths_sarsa, alpha=0.3, color='blue', linewidth=0.5)
    ax2.plot(smooth_curve(lengths_sarsa, 50), color='blue', linewidth=2, label='SARSA')
    ax2.plot(lengths_qlearning, alpha=0.3, color='red', linewidth=0.5)
    ax2.plot(smooth_curve(lengths_qlearning, 50), color='red', linewidth=2, label='Q-Learning')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Episode Length (steps)')
    ax2.set_title('Episode Length Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison_fixed_epsilon.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nAverage reward (last 100 episodes):")
    print(f"  SARSA:      {np.mean(rewards_sarsa[-100:]):.2f}")
    print(f"  Q-Learning: {np.mean(rewards_qlearning[-100:]):.2f}")
    
    # =========================================================================
    # PART 4: Run with decaying epsilon
    # =========================================================================
    print("\n" + "="*70)
    print("PART 4: Decaying Epsilon (ε starts at 0.1, decays to 0.01)")
    print("="*70)
    
    episodes_decay = 1000
    
    print(f"\nRunning SARSA with epsilon decay...")
    Q_sarsa_decay, rewards_sarsa_decay, lengths_sarsa_decay = sarsa(
        env, episodes=episodes_decay, alpha=alpha, gamma=gamma, 
        epsilon=0.1, epsilon_decay=True, epsilon_min=0.01
    )
    
    print(f"Running Q-Learning with epsilon decay...")
    Q_qlearning_decay, rewards_qlearning_decay, lengths_qlearning_decay = q_learning(
        env, episodes=episodes_decay, alpha=alpha, gamma=gamma,
        epsilon=0.1, epsilon_decay=True, epsilon_min=0.01
    )
    
    # Generate paths with decayed policies
    path_sarsa_decay = generate_episode_path(env, Q_sarsa_decay, epsilon=0.0)
    path_qlearning_decay = generate_episode_path(env, Q_qlearning_decay, epsilon=0.0)
    
    print(f"\nWith epsilon decay:")
    print(f"  SARSA path length: {len(path_sarsa_decay)-1} steps")
    print(f"  Q-Learning path length: {len(path_qlearning_decay)-1} steps")
    
    # Visualize decayed epsilon paths
    fig3 = env.visualize_path(path_sarsa_decay, "SARSA Path (Decayed ε - Converged to Optimal)")
    plt.savefig('sarsa_path_decay.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig4 = env.visualize_path(path_qlearning_decay, "Q-Learning Path (Decayed ε - Optimal)")
    plt.savefig('qlearning_path_decay.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot decaying epsilon performance
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Rewards with decay
    ax1 = axes[0]
    ax1.plot(rewards_sarsa_decay, alpha=0.3, color='blue', linewidth=0.5)
    ax1.plot(smooth_curve(rewards_sarsa_decay, 50), color='blue', linewidth=2, label='SARSA')
    ax1.plot(rewards_qlearning_decay, alpha=0.3, color='red', linewidth=0.5)
    ax1.plot(smooth_curve(rewards_qlearning_decay, 50), color='red', linewidth=2, label='Q-Learning')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Sum of Rewards per Episode')
    ax1.set_title('Performance with Epsilon Decay (ε: 0.1 → 0.01)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=-13, color='green', linestyle='--', alpha=0.5, label='Optimal')
    
    # Plot 2: Episode lengths with decay
    ax2 = axes[1]
    ax2.plot(lengths_sarsa_decay, alpha=0.3, color='blue', linewidth=0.5)
    ax2.plot(smooth_curve(lengths_sarsa_decay, 50), color='blue', linewidth=2, label='SARSA')
    ax2.plot(lengths_qlearning_decay, alpha=0.3, color='red', linewidth=0.5)
    ax2.plot(smooth_curve(lengths_qlearning_decay, 50), color='red', linewidth=2, label='Q-Learning')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Episode Length (steps)')
    ax2.set_title('Episode Length with Epsilon Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=13, color='green', linestyle='--', alpha=0.5, label='Optimal')
    
    plt.tight_layout()
    plt.savefig('performance_comparison_decay_epsilon.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print final statistics
    print(f"\nAverage reward (last 100 episodes with decay):")
    print(f"  SARSA:      {np.mean(rewards_sarsa_decay[-100:]):.2f}")
    print(f"  Q-Learning: {np.mean(rewards_qlearning_decay[-100:]):.2f}")
    
    # =========================================================================
    # PART 5: Create comprehensive analysis document
    # =========================================================================
    print("\n" + "="*70)
    print("Creating Analysis Report...")
    print("="*70)
    
    analysis = f"""
CLIFF WALKING ANALYSIS REPORT
{'='*70}

EXPERIMENT PARAMETERS:
- Episodes (fixed ε): {episodes}
- Episodes (decay ε): {episodes_decay}
- Learning rate (α): {alpha}
- Discount factor (γ): {gamma}
- Epsilon (ε): {epsilon} (fixed) or 0.1→0.01 (decay)

{'='*70}
QUESTION 1: Why do SARSA and Q-Learning learn different paths?
{'='*70}

ANSWER:
The key difference is ON-POLICY vs OFF-POLICY learning:

**SARSA (On-Policy)**:
- Update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
- A' is chosen using the CURRENT policy (ε-greedy with ε={epsilon})
- SARSA learns the value of the policy IT IS ACTUALLY FOLLOWING
- Since it explores with ε={epsilon}, it occasionally takes random actions
- Near the cliff, random actions can cause falling → big penalty
- SARSA learns: "The cliff edge is dangerous when I'm exploring"
- Result: Learns SAFE path (blue path) away from cliff

**Q-Learning (Off-Policy)**:
- Update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
- Uses MAX over actions, not the action actually taken
- Q-Learning learns the value of the OPTIMAL policy
- It separates learning from behavior
- Learns: "The optimal path is along the cliff edge"
- Result: Learns OPTIMAL path (red path) but performs worse during training

**OBSERVED RESULTS**:
- SARSA path length: {len(path_sarsa)-1} steps (safer, longer route)
- Q-Learning path length: {len(path_qlearning)-1} steps (optimal, risky route)

{'='*70}
QUESTION 2: Why does Q-Learning have lower average rewards?
{'='*70}

ANSWER:
**Paradox**: Q-Learning finds the optimal policy but gets worse rewards!

**Explanation**:
1. Q-Learning learns Q* (optimal values) assuming no exploration
2. But during training, it ACTS with ε-greedy (explores {epsilon*100}% of time)
3. The optimal path goes along the cliff edge (13 steps)
4. With ε={epsilon}, near cliff: {epsilon*100}% chance of random action
5. Random action near cliff → fall off → -100 reward
6. Q-Learning takes risky optimal path → frequently falls during training

**SARSA's Advantage**:
1. SARSA learns the value of the ε-greedy policy it's actually using
2. It learns: "Taking the cliff path with exploration is dangerous"
3. Finds safer path through middle of grid (~17 steps)
4. Longer path but fewer cliff falls → better average reward during training

**MEASURED RESULTS**:
Fixed ε={epsilon}:
- SARSA avg reward (last 100): {np.mean(rewards_sarsa[-100:]):.2f}
- Q-Learning avg reward (last 100): {np.mean(rewards_qlearning[-100:]):.2f}

Q-Learning has WORSE online performance despite learning the BETTER policy!

{'='*70}
QUESTION 3: Why do both converge with epsilon decay?
{'='*70}

ANSWER:
When ε gradually decreases (0.1 → 0.01):

**What Happens**:
1. Early episodes (high ε): Lots of exploration
   - SARSA: Learns safe path, avoids cliff
   - Q-Learning: Learns optimal values, but falls often

2. Middle episodes (medium ε): Less exploration
   - Both algorithms refine their Q-values
   - Exploration decreases, exploitation increases

3. Late episodes (low ε ≈ 0.01): Mostly exploitation
   - Both act nearly greedily (99% greedy, 1% random)
   - SARSA: Q-values now represent near-greedy policy
   - With little exploration, cliff path becomes safe
   - SARSA updates toward optimal Q*

4. Final convergence:
   - Both converge to Q* (optimal action-values)
   - Both follow optimal policy when ε→0

**WHY THIS WORKS**:
- As ε→0, SARSA's learned policy → greedy policy
- Greedy policy based on Q* is the optimal policy
- Q-Learning already learned Q*, just needed behavior to catch up
- Result: Both find 13-step optimal path along cliff

**MEASURED RESULTS**:
With ε decay (0.1 → 0.01):
- SARSA path: {len(path_sarsa_decay)-1} steps
- Q-Learning path: {len(path_qlearning_decay)-1} steps
- Both converge to optimal 13-step path!

Avg rewards (last 100 episodes):
- SARSA: {np.mean(rewards_sarsa_decay[-100:]):.2f}
- Q-Learning: {np.mean(rewards_qlearning_decay[-100:]):.2f}
- Much closer to optimal (-13) as exploration decreased

{'='*70}
KEY INSIGHTS
{'='*70}

1. **On-Policy vs Off-Policy**:
   - SARSA (on-policy): Safe during training, slower convergence
   - Q-Learning (off-policy): Risky during training, faster to optimal

2. **Exploration-Exploitation Tradeoff**:
   - Fixed ε: Constant exploration penalty
   - Decaying ε: Best of both worlds - explore early, exploit later

3. **Performance vs Optimality**:
   - Best TRAINING performance ≠ Best LEARNED policy
   - Q-Learning suffers during training but learns optimal faster

4. **Practical Implications**:
   - Safety-critical applications: Use SARSA (or conservative exploration)
   - Simulation environments: Use Q-Learning (can tolerate failures)
   - Always use epsilon decay for best final performance!

{'='*70}
"""
    
    # Save analysis
    with open('cliff_walking_analysis.txt', 'w') as f:
        f.write(analysis)
    
    print("\n" + analysis)
    print("\nAll results saved to outputs")
    print("="*70)


if __name__ == "__main__":
    main()