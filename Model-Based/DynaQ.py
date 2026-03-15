"""
=============================================================================
Dyna-Q Algorithm - Dyna Maze (Example 8.1, Sutton & Barto, Chap. 8)
=============================================================================

This script implements the Tabular Dyna-Q algorithm and reproduces Figure 8.2
from "Reinforcement Learning: An Introduction" (Sutton & Barto, 2nd ed.).

The agent navigates a 6×9 grid maze from Start (S) to Goal (G), with four
grey-cell obstacles to avoid. We compare three planning variants:
  - n = 0  : Pure Q-learning (no planning)
  - n = 5  : Q-learning + 5 simulated planning steps per real step
  - n = 50 : Q-learning + 50 simulated planning steps per real step

Key hyperparameters (as specified in the textbook):
  - Alpha (α) = 0.1   — learning rate
  - Gamma (γ) = 0.95  — discount factor
  - Epsilon (ε) = 0.1 — exploration rate (ε-greedy policy)
  - Episodes = 50
  - Repetitions = 30  — averaged over 30 independent runs

Usage:
    python dyna_q_maze.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


# =============================================================================
# SECTION 1: MAZE ENVIRONMENT
# =============================================================================

class DynaMaze:
    """
    A 6-row × 9-column grid-world maze environment.

    Coordinate system:
        - Row 0 is the TOP row, row 5 is the BOTTOM row.
        - Col 0 is the LEFT column, col 8 is the RIGHT column.
        - States are represented as (row, col) tuples.

    Layout (matches the textbook Figure 8.2 inset and the provided image):
        - S (start) : row 2, col 0
        - G (goal)  : row 0, col 8
        - Obstacles : (1,2), (2,2), (3,2), (0,7), (1,7), (3,4)

    Actions: 0=Up, 1=Down, 2=Right, 3=Left
    Reward  : +1 on entering the goal, 0 everywhere else.
    Episode ends when the agent reaches G, then resets to S.
    """

    # Grid dimensions
    ROWS = 6
    COLS = 9

    # Special states
    START = (2, 0)
    GOAL  = (0, 8)

    # Obstacle (blocked) cells — agent cannot enter these
    OBSTACLES = {(1, 2), (2, 2), (3, 2), (0, 7), (1, 7), (3, 4)}

    # Action definitions: (row_delta, col_delta)
    ACTIONS = {
        0: (-1,  0),   # Up
        1: ( 1,  0),   # Down
        2: ( 0,  1),   # Right
        3: ( 0, -1),   # Left
    }
    N_ACTIONS = 4

    def reset(self):
        """Reset the agent to the start state and return it."""
        self.state = self.START
        return self.state

    def step(self, action):
        """
        Apply `action` from the current state.

        Returns
        -------
        next_state : (row, col)
        reward     : float  (+1 at goal, else 0)
        done       : bool   (True if goal reached)
        """
        row, col = self.state
        dr, dc   = self.ACTIONS[action]

        # Candidate next position
        next_row = row + dr
        next_col = col + dc

        # Stay in place if the move goes out of bounds or into an obstacle
        if (0 <= next_row < self.ROWS and
                0 <= next_col < self.COLS and
                (next_row, next_col) not in self.OBSTACLES):
            next_state = (next_row, next_col)
        else:
            next_state = self.state   # bounce back

        # Reward and terminal check
        if next_state == self.GOAL:
            reward, done = 1.0, True
        else:
            reward, done = 0.0, False

        self.state = next_state
        return next_state, reward, done


# =============================================================================
# SECTION 2: DYNA-Q AGENT
# =============================================================================

class DynaQAgent:
    """
    Tabular Dyna-Q agent (Algorithm from p. 164, Sutton & Barto).

    Algorithm overview per time step:
      (a) Observe current state S.
      (b) Choose action A via ε-greedy policy on Q(S, ·).
      (c) Execute A; observe reward R and next state S'.
      (d) Direct RL update:
            Q(S,A) ← Q(S,A) + α[R + γ·max_a Q(S',a) − Q(S,A)]
      (e) Model learning:
            Model(S,A) ← (R, S')      [deterministic model]
      (f) Planning — repeat n times:
            S̃  ← random previously-visited state
            Ã  ← random action previously taken in S̃
            R̃, S̃' ← Model(S̃, Ã)
            Q(S̃,Ã) ← Q(S̃,Ã) + α[R̃ + γ·max_a Q(S̃',a) − Q(S̃,Ã)]

    Parameters
    ----------
    n_planning : int   — number of simulated planning steps per real step
    alpha      : float — step-size / learning rate  (default 0.1)
    gamma      : float — discount factor            (default 0.95)
    epsilon    : float — exploration probability    (default 0.1)
    n_actions  : int   — number of available actions (default 4)
    """

    def __init__(self, n_planning=5, alpha=0.1, gamma=0.95,
                 epsilon=0.1, n_actions=4):
        self.n      = n_planning
        self.alpha  = alpha
        self.gamma  = gamma
        self.eps    = epsilon
        self.n_act  = n_actions

        # Q-table: maps (state, action) → value (default 0.0)
        self.Q = defaultdict(float)

        # Deterministic model: maps (state, action) → (reward, next_state)
        self.model = {}

        # Track which actions have been tried in each state (for planning step)
        # Keys: state → set of actions taken at least once
        self.observed_state_actions = defaultdict(set)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def choose_action(self, state):
        """
        ε-greedy action selection.

        With probability ε, choose a random action (exploration).
        Otherwise, choose the action with the highest Q-value (exploitation).
        Ties among greedy actions are broken uniformly at random.
        """
        if np.random.random() < self.eps:
            return np.random.randint(self.n_act)   # explore

        # Greedy: find all actions tied at the max Q-value
        q_vals = [self.Q[(state, a)] for a in range(self.n_act)]
        max_q  = max(q_vals)
        greedy_actions = [a for a in range(self.n_act) if q_vals[a] == max_q]
        return np.random.choice(greedy_actions)   # break ties randomly

    # ------------------------------------------------------------------
    # Q-value update (shared by direct RL and planning)
    # ------------------------------------------------------------------

    def _q_update(self, state, action, reward, next_state):
        """
        One-step Q-learning update (Bellman update):

            Q(S,A) ← Q(S,A) + α · [R + γ · max_a Q(S',a) - Q(S,A)]

        This identical update is used for both:
          - Direct RL  (step d in the algorithm, real experience)
          - Planning   (step f in the algorithm, simulated experience)
        """
        best_next = max(self.Q[(next_state, a)] for a in range(self.n_act))
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td_error

    # ------------------------------------------------------------------
    # Single environment step (acting + direct RL + model update + planning)
    # ------------------------------------------------------------------

    def step(self, state, env):
        """
        Execute one full Dyna-Q step:
          1. Choose action (ε-greedy).
          2. Interact with the real environment.
          3. Direct RL Q-update.
          4. Update the deterministic model.
          5. Perform n planning (simulated) Q-updates.

        Parameters
        ----------
        state : (row, col) — current state of the agent
        env   : DynaMaze   — the environment (used only for the real step)

        Returns
        -------
        next_state : (row, col)
        done       : bool
        """
        # --- (b) Choose action ---
        action = self.choose_action(state)

        # --- (c) Take real action in the environment ---
        next_state, reward, done = env.step(action)

        # --- (d) Direct RL update using real experience ---
        self._q_update(state, action, reward, next_state)

        # --- (e) Model learning: record (S,A) → (R, S') ---
        self.model[(state, action)] = (reward, next_state)
        self.observed_state_actions[state].add(action)

        # --- (f) Planning: n simulated updates ---
        self._planning()

        return next_state, done

    # ------------------------------------------------------------------
    # Planning sub-routine
    # ------------------------------------------------------------------

    def _planning(self):
        """
        Perform n steps of random-sample one-step Q-planning.

        For each simulated step:
          1. Sample a previously observed state at random.
          2. Sample a previously taken action from that state at random.
          3. Retrieve the modelled (reward, next_state).
          4. Apply the same Q-update as in direct RL.

        Because we only sample from (state, action) pairs already stored
        in the model, we never query the model with unknown inputs.
        """
        if not self.model:
            return   # Nothing to plan with yet

        # Collect all state-action pairs stored in the model
        model_keys = list(self.model.keys())

        for _ in range(self.n):
            # Random previously-observed (state, action)
            idx   = np.random.randint(len(model_keys))
            s, a  = model_keys[idx]

            # Retrieve model prediction
            r, s_next = self.model[(s, a)]

            # Simulated Q-update (identical formula to direct RL)
            self._q_update(s, a, r, s_next)


# =============================================================================
# SECTION 3: TRAINING LOOP
# =============================================================================

def run_experiment(n_planning, n_episodes=50, n_runs=30,
                   alpha=0.1, gamma=0.95, epsilon=0.1):
    """
    Run Dyna-Q for `n_runs` independent repetitions and collect the number
    of steps taken to reach the goal in each episode.

    Parameters
    ----------
    n_planning : int  — planning steps per real environment step
    n_episodes : int  — number of episodes per run (default 50)
    n_runs     : int  — number of independent repetitions (default 30)
    alpha      : float — learning rate
    gamma      : float — discount factor
    epsilon    : float — exploration parameter

    Returns
    -------
    steps_per_episode : np.ndarray, shape (n_episodes,)
        Mean number of steps to reach the goal across all runs,
        for each episode (episode 1 is excluded per the textbook).
    """
    # Accumulate step counts: shape (n_runs, n_episodes)
    all_steps = np.zeros((n_runs, n_episodes))

    for run in range(n_runs):
        # Fresh environment and agent for each independent run
        env   = DynaMaze()
        agent = DynaQAgent(n_planning=n_planning, alpha=alpha,
                            gamma=gamma, epsilon=epsilon)

        for ep in range(n_episodes):
            state     = env.reset()
            steps     = 0
            done      = False

            # Run one episode until the goal is reached
            while not done:
                state, done = agent.step(state, env)
                steps += 1

                # Safety cap: prevent extremely long first episodes from
                # skewing the plot (first episode can take ~1700 steps)
                if steps > 3000:
                    break

            all_steps[run, ep] = steps

    # Average over all runs; episode 0 (the very first) is excluded in
    # Figure 8.2 because it was ~1700 steps identical across all n values.
    # We return episodes 1..n_episodes-1 (0-indexed: [1:])
    mean_steps = all_steps.mean(axis=0)
    return mean_steps   # shape: (n_episodes,)


# =============================================================================
# SECTION 4: REPRODUCE FIGURE 8.2
# =============================================================================

def plot_figure_8_2(results, planning_steps, n_episodes=50):
    """
    Reproduce Figure 8.2 from Sutton & Barto:
      - X-axis: Episodes (2 through n_episodes)
      - Y-axis: Steps per episode (average over 30 runs)
      - One curve per planning step count n ∈ {0, 5, 50}
    Also draw the maze inset.

    Parameters
    ----------
    results       : dict  {n_value: mean_steps_array}
    planning_steps: list  planning step values used
    n_episodes    : int   total episodes run
    """
    colors = {0: '#e74c3c', 5: '#2ecc71', 50: '#3498db'}
    labels = {0: 'n = 0  (direct RL only)',
              5: 'n = 5  planning steps',
              50: 'n = 50 planning steps'}

    # ---- Plot 1: Learning Curves only ----
    fig1, ax_main = plt.subplots(figsize=(10, 6))
    fig1.patch.set_facecolor('#f8f8f8')

    # Episode 1 (index 0) is omitted, matching the textbook
    episodes_x = np.arange(2, n_episodes + 1)

    for n in planning_steps:
        # results[n] has shape (n_episodes,); skip index 0 (episode 1)
        y = results[n][1:]
        ax_main.plot(episodes_x, y,
                     color=colors[n], linewidth=2.0,
                     label=labels[n])

    ax_main.set_xlabel('Episodes', fontsize=13)
    ax_main.set_ylabel('Steps per episode', fontsize=13)
    ax_main.set_title('Figure 8.2 — Dyna-Q on the Maze Task\n'
                      '(α=0.1, γ=0.95, ε=0.1, averaged over 30 runs)',
                      fontsize=13)
    ax_main.set_xlim(2, n_episodes)
    ax_main.set_ylim(0, 800)
    ax_main.legend(fontsize=11)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_facecolor('#fdfdfd')

    fig1.tight_layout()
    fig1.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Learning curves saved to: learning_curves.png")

    # ---- Plot 2: Maze Grid only ----
    fig2, ax_maze = plt.subplots(figsize=(6, 4))
    fig2.patch.set_facecolor('#f8f8f8')
    _draw_maze(ax_maze)

    fig2.tight_layout()
    fig2.savefig('maze_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Maze grid saved to: maze_grid.png")


def _draw_maze(ax):
    """
    Draw the 6×9 Dyna Maze as a clean grid with coloured cells.

    Cell colour legend:
      - White  : free cell
      - Grey   : obstacle (blocked)
      - Green  : goal (G)
      - Blue   : start (S)
    """
    rows, cols = DynaMaze.ROWS, DynaMaze.COLS

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Maze layout', fontsize=11)

    for r in range(rows):
        for c in range(cols):
            # In the plot, row 0 should appear at the TOP.
            # matplotlib y=0 is at the bottom, so we flip: plot_row = rows-1-r
            pr = rows - 1 - r

            if (r, c) in DynaMaze.OBSTACLES:
                colour = '#999999'   # grey obstacle
            elif (r, c) == DynaMaze.GOAL:
                colour = '#2ecc71'   # green goal
            elif (r, c) == DynaMaze.START:
                colour = '#3498db'   # blue start
            else:
                colour = 'white'

            rect = mpatches.FancyBboxPatch(
                (c, pr), 1, 1,
                boxstyle="square,pad=0",
                linewidth=0.8, edgecolor='#555555',
                facecolor=colour
            )
            ax.add_patch(rect)

            # Label S and G
            if (r, c) == DynaMaze.START:
                ax.text(c + 0.5, pr + 0.5, 'S',
                        ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')
            elif (r, c) == DynaMaze.GOAL:
                ax.text(c + 0.5, pr + 0.5, 'G',
                        ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')


# =============================================================================
# SECTION 5: MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main driver:
      1. Run Dyna-Q with n = 0, 5, 50 planning steps (30 runs × 50 episodes).
      2. Plot and save Figure 8.2.
      3. Print a short commentary on the results.
    """
    print("=" * 60)
    print("  Dyna-Q Maze Experiment (Sutton & Barto, Example 8.1)")
    print("=" * 60)

    PLANNING_STEPS = [0, 5, 50]
    N_EPISODES     = 50
    N_RUNS         = 30

    results = {}
    for n in PLANNING_STEPS:
        print(f"\nRunning Dyna-Q with n={n} planning steps "
              f"({N_RUNS} runs × {N_EPISODES} episodes)...")
        results[n] = run_experiment(
            n_planning=n,
            n_episodes=N_EPISODES,
            n_runs=N_RUNS
        )
        # Report average steps on episode 2 (first reported episode)
        print(f"  Avg steps on episode 2: {results[n][1]:.1f}")
        print(f"  Avg steps on episode 10: {results[n][9]:.1f}")

    # Plot Figure 8.2
    plot_figure_8_2(results, PLANNING_STEPS, N_EPISODES)

    # ------------------------------------------------------------------
    # Commentary on Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  COMMENTARY ON RESULTS")
    print("=" * 60)
    print("""
n = 0  (Direct RL / Q-learning only):
  No planning occurs. The agent updates only one Q-value per real step,
  so knowledge propagates very slowly backward from the goal.
  It typically takes ~25 episodes before performance plateaus.

n = 5  (5 planning steps per real step):
  After each real transition, 5 additional simulated updates are performed
  using the learned model. This multiplies the effective number of Q-updates
  by 6×, allowing the agent to converge in roughly 5 episodes — 5× faster
  than the n=0 case.

n = 50 (50 planning steps per real step):
  50 simulated updates per real step provide very rapid Q-value propagation.
  The agent typically finds near-optimal performance in just 3 episodes.

Key insight (Dyna-Q advantage):
  Planning lets the agent 're-use' past experience cheaply via the model.
  The same real trajectory that updates one (S,A) pair in direct RL spawns
  50 additional virtual updates in the n=50 case — no extra environment
  interactions required. This is the core benefit of model-based RL.
""")


if __name__ == "__main__":
    main()