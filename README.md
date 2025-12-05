# CartPole DQN Reinforcement Learning Project

## Overview

This project implements a Deep Q-Network (DQN) agent to solve the CartPole-v1 environment using a custom reward modification. The focus of the project is to demonstrate reinforcement learning principles through a clean, simple, and readable implementation while incorporating a unique environment change required by the assignment.

The final outputs include:

* Reward Curve
* Success Rate Plot
* Learning Curve
* Full Training Video Recording
* Trained Model File

All outputs are saved in the `outputs/` folder.

---

## Features

* Custom reward function penalizing pole angle
* DQN agent implemented from scratch

  * Replay buffer
  * Epsilon-greedy exploration
  * Target network
  * CUDA acceleration
* Full training video recording (500 episodes)
* Reward, Success Rate, and Learning Curve plots
* Clean and minimal 4-file implementation
* All outputs automatically saved to `/outputs`

---

## Directory Structure

```
RLProject/
│
├── run.py
├── dqn_agent.py
├── custom_env.py
├── video.py
├── outputs/
│   ├── reward_curve.png
│   ├── success_rate.png
│   ├── learning_curve.png
│   ├── cartpole_training.mp4
│   └── dqn_cartpole.pth
└── README.md
```

---

## Custom Reward Function

The uniqueness requirement for this project is met by modifying the CartPole environment’s reward function.

Default reward:

```
reward = 1
```

Modified reward:

```
reward = max(1.0 − (angle_penalty_weight × |pole_angle|), 0.0)
```

Where:

* `pole_angle` is the angle of the pole at each step
* `angle_penalty_weight` controls penalty severity (set to **0.10**)

This encourages the agent not only to survive longer but to **keep the pole as vertical as possible**, producing more stable behavior.

---

## Installation

### 1. Clone the repository

```
git clone <your-repo-url>
cd RLProject
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## How to Run

### Train the agent

```
python run.py
```

This will:

* Train for 500 episodes
* Record the entire training to `outputs/cartpole_training.mp4`
* Save all performance plots
* Save the trained model

---

## Outputs

### Reward Curve

Shows reward per episode and rolling mean.

### Success Rate

Displays how often the agent achieves at least 200 reward.

### Learning Curve

Shows smoothed reward trend over time.

### Training Video

An MP4 visualizing the entire training process frame-by-frame.

### Model File

The trained weights saved as:

```
outputs/dqn_cartpole.pth
```

---

## Requirements

* Python 3.8+
* PyTorch
* Gymnasium
* NumPy
* Matplotlib
* OpenCV

---

## License

This project is for educational and academic use.

---

## Acknowledgements

Thanks to the instructors for the assignment framework and evaluation criteria.
This implementation closely follows required stages while emphasizing readability and proper RL procedure.
