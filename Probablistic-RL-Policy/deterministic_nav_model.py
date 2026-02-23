import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import time

# Define the world
W = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Parameters
GOAL = (7, 10)  # W(8,11) in 0-indexed coordinates (row 7, col 10)
GAMMA = 0.95
OBSTACLE_REWARD = -50.0
MOVEMENT_REWARD = -1.0
GOAL_REWARD = 100.0
THETA = 0.01  # Convergence threshold

# 8 possible actions: N, NE, E, SE, S, SW, W, NW
ACTIONS = [
    (-1, 0),   # 0: North
    (-1, 1),   # 1: NorthEast
    (0, 1),    # 2: East
    (1, 1),    # 3: SouthEast
    (1, 0),    # 4: South
    (1, -1),   # 5: SouthWest
    (0, -1),   # 6: West
    (-1, -1),  # 7: NorthWest
]

ACTION_NAMES = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def is_valid_state(row, col):
    """Check if state is valid (within bounds and not an obstacle)"""
    if row < 0 or row >= W.shape[0] or col < 0 or col >= W.shape[1]:
        return False
    return W[row, col] == 0


def get_next_state(state, action):
    """Get next state given current state and action"""
    row, col = state
    dr, dc = ACTIONS[action]
    new_row, new_col = row + dr, col + dc
    return (new_row, new_col)


def get_reward(state, next_state):
    """Get reward for transitioning to next_state"""
    if next_state == GOAL:
        return GOAL_REWARD
    if not is_valid_state(next_state[0], next_state[1]):
        return OBSTACLE_REWARD
    return MOVEMENT_REWARD


def get_transition_prob_deterministic(state, action, next_state):
    """Get transition probability for deterministic model"""
    expected_next = get_next_state(state, action)
    if next_state == expected_next:
        return 1.0
    return 0.0


def get_transition_prob_stochastic(state, action, next_state):
    """Get transition probability for stochastic model (20% chance of ±45°)"""
    # Main action: 60% probability
    # +45°: 20% probability
    # -45°: 20% probability
    
    # Calculate where each action would take us
    expected_next = get_next_state(state, action)
    
    # Get adjacent actions (±45 degrees)
    left_action = (action - 1) % 8
    right_action = (action + 1) % 8
    
    left_next = get_next_state(state, left_action)
    right_next = get_next_state(state, right_action)
    
    # If we hit an obstacle, we stay in current state
    if not is_valid_state(expected_next[0], expected_next[1]):
        expected_next = state
    if not is_valid_state(left_next[0], left_next[1]):
        left_next = state
    if not is_valid_state(right_next[0], right_next[1]):
        right_next = state
    
    # Calculate probability of ending up in next_state
    prob = 0.0
    if next_state == expected_next:
        prob += 0.6
    if next_state == left_next:
        prob += 0.2
    if next_state == right_next:
        prob += 0.2
    
    return prob


def get_all_states():
    """Get all valid states"""
    states = []
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] == 0:
                states.append((i, j))
    return states


def policy_evaluation(policy, V, states, is_stochastic=False, theta=THETA):
    """Evaluate a policy"""
    while True:
        delta = 0
        for state in states:
            if state == GOAL:
                continue
            
            v = V[state]
            action = policy[state]
            
            # Calculate expected value
            new_v = 0
            
            # Get all possible next states considering the action
            if is_stochastic:
                # For stochastic, need to consider all possible outcomes
                expected_next = get_next_state(state, action)
                left_action = (action - 1) % 8
                right_action = (action + 1) % 8
                left_next = get_next_state(state, left_action)
                right_next = get_next_state(state, right_action)
                
                # If hit obstacle, stay in current state
                if not is_valid_state(expected_next[0], expected_next[1]):
                    expected_next = state
                if not is_valid_state(left_next[0], left_next[1]):
                    left_next = state
                if not is_valid_state(right_next[0], right_next[1]):
                    right_next = state
                
                # Collect all unique outcomes with their probabilities
                outcomes = {}
                outcomes[expected_next] = outcomes.get(expected_next, 0) + 0.6
                outcomes[left_next] = outcomes.get(left_next, 0) + 0.2
                outcomes[right_next] = outcomes.get(right_next, 0) + 0.2
                
                for next_state, prob in outcomes.items():
                    reward = get_reward(state, next_state)
                    if next_state == GOAL:
                        next_v = 0  # Terminal state
                    else:
                        next_v = V[next_state]
                    new_v += prob * (reward + GAMMA * next_v)
            else:
                # Deterministic case
                next_state = get_next_state(state, action)
                if not is_valid_state(next_state[0], next_state[1]):
                    next_state = state
                
                reward = get_reward(state, next_state)
                if next_state == GOAL:
                    next_v = 0
                else:
                    next_v = V[next_state]
                new_v = reward + GAMMA * next_v
            
            V[state] = new_v
            delta = max(delta, abs(v - new_v))
        
        if delta < theta:
            break
    
    return V


def policy_iteration(states, is_stochastic=False):
    """Policy Iteration algorithm"""
    print(f"\n{'='*60}")
    print(f"Running Policy Iteration ({'Stochastic' if is_stochastic else 'Deterministic'})")
    print(f"{'='*60}")
    
    # Initialize policy randomly
    policy = {}
    for state in states:
        policy[state] = np.random.randint(0, 8)
    
    # Initialize value function
    V = {}
    for state in states:
        V[state] = 0.0
    
    iteration = 0
    convergence_history = []
    start_time = time.time()
    
    while True:
        iteration += 1
        
        # Policy Evaluation
        V = policy_evaluation(policy, V, states, is_stochastic)
        
        # Policy Improvement
        policy_stable = True
        max_change = 0
        
        for state in states:
            if state == GOAL:
                continue
            
            old_action = policy[state]
            
            # Find best action
            action_values = np.zeros(8)
            for action in range(8):
                value = 0
                
                if is_stochastic:
                    # Get all possible outcomes
                    expected_next = get_next_state(state, action)
                    left_action = (action - 1) % 8
                    right_action = (action + 1) % 8
                    left_next = get_next_state(state, left_action)
                    right_next = get_next_state(state, right_action)
                    
                    # If hit obstacle, stay in current state
                    if not is_valid_state(expected_next[0], expected_next[1]):
                        expected_next = state
                    if not is_valid_state(left_next[0], left_next[1]):
                        left_next = state
                    if not is_valid_state(right_next[0], right_next[1]):
                        right_next = state
                    
                    # Collect all unique outcomes with their probabilities
                    outcomes = {}
                    outcomes[expected_next] = outcomes.get(expected_next, 0) + 0.6
                    outcomes[left_next] = outcomes.get(left_next, 0) + 0.2
                    outcomes[right_next] = outcomes.get(right_next, 0) + 0.2
                    
                    for next_state, prob in outcomes.items():
                        reward = get_reward(state, next_state)
                        if next_state == GOAL:
                            next_v = 0
                        else:
                            next_v = V[next_state]
                        value += prob * (reward + GAMMA * next_v)
                else:
                    # Deterministic case
                    next_state = get_next_state(state, action)
                    if not is_valid_state(next_state[0], next_state[1]):
                        next_state = state
                    
                    reward = get_reward(state, next_state)
                    if next_state == GOAL:
                        next_v = 0
                    else:
                        next_v = V[next_state]
                    value = reward + GAMMA * next_v
                
                action_values[action] = value
            
            best_action = np.argmax(action_values)
            policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
                max_change += 1
        
        max_v = max([abs(v) for v in V.values()])
        convergence_history.append(max_v)
        
        print(f"Iteration {iteration}: Policy changed at {max_change} states, Max |V|: {max_v:.4f}")
        
        if policy_stable:
            break
    
    elapsed_time = time.time() - start_time
    print(f"\nConverged in {iteration} iterations ({elapsed_time:.2f} seconds)")
    
    return policy, V, convergence_history


def value_iteration(states, is_stochastic=False):
    """Value Iteration algorithm"""
    print(f"\n{'='*60}")
    print(f"Running Value Iteration ({'Stochastic' if is_stochastic else 'Deterministic'})")
    print(f"{'='*60}")
    
    # Initialize value function
    V = {}
    for state in states:
        V[state] = 0.0
    
    iteration = 0
    convergence_history = []
    start_time = time.time()
    
    while True:
        iteration += 1
        delta = 0
        
        for state in states:
            if state == GOAL:
                continue
            
            v = V[state]
            
            # Find max action value
            action_values = np.zeros(8)
            for action in range(8):
                value = 0
                
                if is_stochastic:
                    # Get all possible outcomes
                    expected_next = get_next_state(state, action)
                    left_action = (action - 1) % 8
                    right_action = (action + 1) % 8
                    left_next = get_next_state(state, left_action)
                    right_next = get_next_state(state, right_action)
                    
                    # If hit obstacle, stay in current state
                    if not is_valid_state(expected_next[0], expected_next[1]):
                        expected_next = state
                    if not is_valid_state(left_next[0], left_next[1]):
                        left_next = state
                    if not is_valid_state(right_next[0], right_next[1]):
                        right_next = state
                    
                    # Collect all unique outcomes with their probabilities
                    outcomes = {}
                    outcomes[expected_next] = outcomes.get(expected_next, 0) + 0.6
                    outcomes[left_next] = outcomes.get(left_next, 0) + 0.2
                    outcomes[right_next] = outcomes.get(right_next, 0) + 0.2
                    
                    for next_state, prob in outcomes.items():
                        reward = get_reward(state, next_state)
                        if next_state == GOAL:
                            next_v = 0
                        else:
                            next_v = V[next_state]
                        value += prob * (reward + GAMMA * next_v)
                else:
                    # Deterministic case
                    next_state = get_next_state(state, action)
                    if not is_valid_state(next_state[0], next_state[1]):
                        next_state = state
                    
                    reward = get_reward(state, next_state)
                    if next_state == GOAL:
                        next_v = 0
                    else:
                        next_v = V[next_state]
                    value = reward + GAMMA * next_v
                
                action_values[action] = value
            
            V[state] = max(action_values)
            delta = max(delta, abs(v - V[state]))
        
        max_v = max([abs(v) for v in V.values()])
        convergence_history.append(max_v)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Delta: {delta:.6f}, Max |V|: {max_v:.4f}")
        
        if delta < THETA:
            break
    
    elapsed_time = time.time() - start_time
    print(f"\nConverged in {iteration} iterations ({elapsed_time:.2f} seconds)")
    
    # Extract policy from value function
    policy = {}
    for state in states:
        if state == GOAL:
            policy[state] = 0
            continue
        
        action_values = np.zeros(8)
        for action in range(8):
            value = 0
            
            if is_stochastic:
                # Get all possible outcomes
                expected_next = get_next_state(state, action)
                left_action = (action - 1) % 8
                right_action = (action + 1) % 8
                left_next = get_next_state(state, left_action)
                right_next = get_next_state(state, right_action)
                
                # If hit obstacle, stay in current state
                if not is_valid_state(expected_next[0], expected_next[1]):
                    expected_next = state
                if not is_valid_state(left_next[0], left_next[1]):
                    left_next = state
                if not is_valid_state(right_next[0], right_next[1]):
                    right_next = state
                
                # Collect all unique outcomes with their probabilities
                outcomes = {}
                outcomes[expected_next] = outcomes.get(expected_next, 0) + 0.6
                outcomes[left_next] = outcomes.get(left_next, 0) + 0.2
                outcomes[right_next] = outcomes.get(right_next, 0) + 0.2
                
                for next_state, prob in outcomes.items():
                    reward = get_reward(state, next_state)
                    if next_state == GOAL:
                        next_v = 0
                    else:
                        next_v = V[next_state]
                    value += prob * (reward + GAMMA * next_v)
            else:
                # Deterministic case
                next_state = get_next_state(state, action)
                if not is_valid_state(next_state[0], next_state[1]):
                    next_state = state
                
                reward = get_reward(state, next_state)
                if next_state == GOAL:
                    next_v = 0
                else:
                    next_v = V[next_state]
                value = reward + GAMMA * next_v
            
            action_values[action] = value
        
        policy[state] = np.argmax(action_values)
    
    return policy, V, convergence_history


def plot_policy(policy, title, filename, show_trajectory=False, start_state=None):
    """Plot the optimal policy as arrows on the grid"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the world
    ax.imshow(W, cmap='gray_r', origin='upper')
    
    # Plot goal
    ax.plot(GOAL[1], GOAL[0], 'r*', markersize=20, label='Goal', zorder=5)
    
    # Plot arrows for policy
    arrow_scale = 0.35
    for state, action in policy.items():
        if state == GOAL:
            continue
        
        row, col = state
        dr, dc = ACTIONS[action]
        
        # Scale arrows
        dr *= arrow_scale
        dc *= arrow_scale
        
        ax.arrow(col, row, dc, dr, 
                head_width=0.2, head_length=0.15,
                fc='blue', ec='blue', alpha=0.6, linewidth=0.5)
    
    # Plot sample trajectory if requested
    if show_trajectory and start_state is not None:
        trajectory = []
        current = start_state
        trajectory.append(current)
        max_steps = 200
        
        for step in range(max_steps):
            if current == GOAL:
                break
            if current not in policy:
                print(f"Warning: State {current} not in policy!")
                break
            
            action = policy[current]
            next_state = get_next_state(current, action)
            
            # Check if next state is valid
            if not is_valid_state(next_state[0], next_state[1]):
                print(f"Warning: Policy leads to obstacle at step {step}: {current} -> {next_state}")
                break
            
            trajectory.append(next_state)
            current = next_state
        
        # Plot trajectory
        if len(trajectory) > 1:
            traj_rows = [s[0] for s in trajectory]
            traj_cols = [s[1] for s in trajectory]
            ax.plot(traj_cols, traj_rows, 'g-', linewidth=3, alpha=0.7, 
                   label=f'Sample Path ({len(trajectory)} steps)', zorder=4)
            ax.plot(traj_cols[0], traj_rows[0], 'go', markersize=12, 
                   label='Start', zorder=5)
        else:
            print(f"Warning: Trajectory only has {len(trajectory)} states")
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved policy plot: {filename}")
    plt.close()


def plot_value_function(V, title, filename):
    """Plot the value function as a heatmap"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create value grid
    V_grid = np.full_like(W, np.nan, dtype=float)
    for state, value in V.items():
        V_grid[state] = value
    
    # Plot
    im = ax.imshow(V_grid, cmap='viridis', origin='upper')
    ax.plot(GOAL[1], GOAL[0], 'r*', markersize=20, label='Goal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=20)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved value function plot: {filename}")
    plt.close()


def plot_convergence(history_pi_det, history_vi_det, 
                     history_pi_sto, history_vi_sto, filename):
    """Plot convergence comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Deterministic
    ax1.plot(history_pi_det, 'b-', label='Policy Iteration', linewidth=2)
    ax1.plot(history_vi_det, 'r-', label='Value Iteration', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max |V|')
    ax1.set_title('Convergence: Deterministic Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Stochastic
    ax2.plot(history_pi_sto, 'b-', label='Policy Iteration', linewidth=2)
    ax2.plot(history_vi_sto, 'r-', label='Value Iteration', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max |V|')
    ax2.set_title('Convergence: Stochastic Model')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved convergence plot: {filename}")
    plt.close()


def main():
    """Main function"""
    print("\nRobot Navigation using Dynamic Programming")
    print("=" * 60)
    
    # Get all valid states
    states = get_all_states()
    print(f"\nTotal valid states: {len(states)}")
    print(f"Goal state: {GOAL}")
    
    # Run algorithms
    
    # 1. Policy Iteration - Deterministic
    policy_pi_det, V_pi_det, history_pi_det = policy_iteration(states, is_stochastic=False)
    
    # 2. Value Iteration - Deterministic
    policy_vi_det, V_vi_det, history_vi_det = value_iteration(states, is_stochastic=False)
    
    # 3. Policy Iteration - Stochastic
    policy_pi_sto, V_pi_sto, history_pi_sto = policy_iteration(states, is_stochastic=True)
    
    # 4. Value Iteration - Stochastic
    policy_vi_sto, V_vi_sto, history_vi_sto = value_iteration(states, is_stochastic=True)
    
    # Create plots
    print("\n" + "="*60)
    print("Creating plots...")
    print("="*60)
    
    # Policy plots
    plot_policy(policy_pi_det, 
                'Optimal Policy - Policy Iteration (Deterministic)', 
                'policy_pi_deterministic.png')
    
    plot_policy(policy_vi_det, 
                'Optimal Policy - Value Iteration (Deterministic)', 
                'policy_vi_deterministic.png')
    
    # For stochastic, add trajectory from bottom-left corner
    start_state = (12, 2)  # Bottom left area
    plot_policy(policy_pi_sto, 
                'Optimal Policy - Policy Iteration (Stochastic)', 
                'policy_pi_stochastic.png',
                show_trajectory=True, start_state=start_state)
    
    plot_policy(policy_vi_sto, 
                'Optimal Policy - Value Iteration (Stochastic)', 
                'policy_vi_stochastic.png',
                show_trajectory=True, start_state=start_state)
    
    # Value function plots
    plot_value_function(V_pi_det, 
                       'Value Function - Policy Iteration (Deterministic)', 
                       'value_pi_deterministic.png')
    
    plot_value_function(V_vi_det, 
                       'Value Function - Value Iteration (Deterministic)', 
                       'value_vi_deterministic.png')
    
    plot_value_function(V_pi_sto, 
                       'Value Function - Policy Iteration (Stochastic)', 
                       'value_pi_stochastic.png')
    
    plot_value_function(V_vi_sto, 
                       'Value Function - Value Iteration (Stochastic)', 
                       '/value_vi_stochastic.png')
    
    # Convergence comparison
    plot_convergence(history_pi_det, history_vi_det, 
                    history_pi_sto, history_vi_sto,
                    'convergence_comparison.png')
    
    print("\n" + "="*60)
    print("ANALYSIS AND COMMENTS")
    print("="*60)
    
    print("\n1. CONVERGENCE RATE COMPARISON:")
    print("-" * 60)
    print(f"Deterministic Model:")
    print(f"  - Policy Iteration: {len(history_pi_det)} iterations")
    print(f"  - Value Iteration:  {len(history_vi_det)} iterations")
    print(f"\nStochastic Model:")
    print(f"  - Policy Iteration: {len(history_pi_sto)} iterations")
    print(f"  - Value Iteration:  {len(history_vi_sto)} iterations")
    
    print("\n2. KEY OBSERVATIONS:")
    print("-" * 60)
    print("• Policy Iteration typically converges in fewer iterations than Value")
    print("  Iteration because it performs full policy evaluation at each step.")
    print("\n• Value Iteration updates values iteratively and may require more")
    print("  iterations but each iteration is computationally cheaper.")
    print("\n• The stochastic model generally requires more iterations to converge")
    print("  due to the uncertainty in action outcomes.")
    print("\n• In the stochastic model, the optimal policy avoids narrow passages")
    print("  and stays toward the center of corridors to minimize collision risk.")
    print("\n• The deterministic model can safely navigate through narrow passages")
    print("  as there is no uncertainty in the action outcomes.")
    
    print("\n" + "="*60)
    print("All plots have been saved to...")
    print("="*60)


if __name__ == "__main__":
    main()