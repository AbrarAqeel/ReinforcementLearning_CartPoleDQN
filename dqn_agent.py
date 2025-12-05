<<<<<<< HEAD
"""
DQN Agent: Q-network, replay buffer, training logic.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(self, s, a, r, ns, d) -> None:
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards),
            np.stack(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_size,
            learning_rate,
            gamma,
            buffer_size,
            batch_size,
            epsilon_start,
            epsilon_end,
            epsilon_decay_steps,
            device
    ):
        self.device = device
        self.gamma = gamma
        self.batch = batch_size

        self.e_start = epsilon_start
        self.e_end = epsilon_end
        self.e_decay = epsilon_decay_steps
        self.total_steps = 0

        self.policy = QNetwork(state_dim, action_dim, hidden_size).to(device)
        self.target = QNetwork(state_dim, action_dim, hidden_size).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optim = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_size)
        self.action_dim = action_dim

    def epsilon(self):
        f = min(self.total_steps / self.e_decay, 1.0)
        return self.e_start + f * (self.e_end - self.e_start)

    def select_action(self, state):
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.action_dim)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy(state_t)
        return int(torch.argmax(q, dim=1).item())

    def store_transition(self, s, a, r, ns, d):
        self.memory.push(s, a, r, ns, d)

    def train_step(self):
        if len(self.memory) < self.batch:
            return

        s, a, r, ns, d = self.memory.sample(self.batch)

        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        q = self.policy(s).gather(1, a).squeeze()

        with torch.no_grad():
            next_q = self.target(ns).max(1).values * (1 - d)

        target = r + self.gamma * next_q

        loss = self.loss_fn(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())
=======
"""
DQN Agent: Q-network, replay buffer, training logic.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(self, s, a, r, ns, d) -> None:
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards),
            np.stack(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_size,
            learning_rate,
            gamma,
            buffer_size,
            batch_size,
            epsilon_start,
            epsilon_end,
            epsilon_decay_steps,
            device
    ):
        self.device = device
        self.gamma = gamma
        self.batch = batch_size

        self.e_start = epsilon_start
        self.e_end = epsilon_end
        self.e_decay = epsilon_decay_steps
        self.total_steps = 0

        self.policy = QNetwork(state_dim, action_dim, hidden_size).to(device)
        self.target = QNetwork(state_dim, action_dim, hidden_size).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optim = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_size)
        self.action_dim = action_dim

    def epsilon(self):
        f = min(self.total_steps / self.e_decay, 1.0)
        return self.e_start + f * (self.e_end - self.e_start)

    def select_action(self, state):
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.action_dim)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy(state_t)
        return int(torch.argmax(q, dim=1).item())

    def store_transition(self, s, a, r, ns, d):
        self.memory.push(s, a, r, ns, d)

    def train_step(self):
        if len(self.memory) < self.batch:
            return

        s, a, r, ns, d = self.memory.sample(self.batch)

        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        q = self.policy(s).gather(1, a).squeeze()

        with torch.no_grad():
            next_q = self.target(ns).max(1).values * (1 - d)

        target = r + self.gamma * next_q

        loss = self.loss_fn(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())
>>>>>>> 00debab322f4b7ccc48e1f27f50b6ef94360d3dc
