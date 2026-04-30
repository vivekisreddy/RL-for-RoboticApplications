# RL for Robotic Applications

A modular reinforcement learning framework implementing custom policies for robotic control tasks, built as part of RBE 595 at WPI. The repo is structured to support rapid experimentation across both model-free and model-based RL paradigms.

## Repository Structure

| Folder | Description |
|---|---|
| `DRL-Policy/` | Deep RL policies (DQN and variants) for continuous control |
| `Model-Based/` | Model-based RL (Dyna-Q) with learned environment dynamics |
| `Probablistic-RL-Policy/` | Probabilistic policy implementations for uncertainty-aware control |
| `Q-Learning/` | Tabular and approximate Q-learning baselines |

## Algorithms Implemented

- **DQN** — Deep Q-Network with experience replay and target network
- **Dyna-Q** — Model-based planning with simulated rollouts
- **Probabilistic Policy** — Stochastic policy for environments with noisy observations
- **Q-Learning** — Tabular baseline for discrete action spaces

## Design Goals

- **Modular** — Each policy is self-contained and interchangeable across environments
- **Transferable** — Environment abstractions designed to support sim-to-real transfer to robotics platforms
- **Configurable** — Hyperparameters decoupled from policy logic for clean experimentation

## Stack

- Python, PyTorch, OpenAI Gym
- C/C++ for performance-critical environment components

## Usage

```bash
# Clone the repo
git clone https://github.com/vivekisreddy/RL-for-RoboticApplications.git
cd RL-for-RoboticApplications

# Run a policy (example: DQN)
cd DRL-Policy
python train.py
```

## Author

**Vivek Reddy Kasireddy** — [LinkedIn](https://linkedin.com) | [Website](https://vivek.com)  
WPI Computer Science & Robotics Engineering, Class of 2026
