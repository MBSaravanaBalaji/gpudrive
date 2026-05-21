"""
DQN agent for the crash NPC.

Architecture:
  - 3-layer MLP Q-network (input → 128 → 128 → n_actions)
  - Experience replay buffer (circular)
  - Target network synced every `target_update_freq` steps
  - Epsilon-greedy exploration with linear decay
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Q-Network ─────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

@dataclass
class DQNConfig:
    obs_dim: int
    n_actions: int
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_capacity: int = 50_000
    batch_size: int = 256
    target_update_freq: int = 500   # steps between target network syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000  # linear decay over this many env steps
    hidden: int = 128
    device: str = "cpu"


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.q_net = QNetwork(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.target_net = QNetwork(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_capacity)

        self.total_steps: int = 0
        self.epsilon: float = cfg.eps_start

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.cfg.n_actions)
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return int(self.q_net(t).argmax(dim=1).item())

    def _update_epsilon(self):
        frac = min(1.0, self.total_steps / self.cfg.eps_decay_steps)
        self.epsilon = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    # ── Learning step ─────────────────────────────────────────────────────────

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)
        self.total_steps += 1
        self._update_epsilon()

        if self.total_steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self) -> float:
        """Sample a batch and do one gradient update. Returns loss value."""
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.cfg.batch_size)

        obs_t      = torch.FloatTensor(obs).to(self.device)
        actions_t  = torch.LongTensor(actions).to(self.device)
        rewards_t  = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t    = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Bellman)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(1)[0]
            target = rewards_t + self.cfg.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "cfg": self.cfg,
        }, path)
        print(f"Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> DQNAgent:
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["cfg"]
        cfg.device = device
        agent = cls(cfg)
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.target_net.load_state_dict(ckpt["q_net"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.total_steps = ckpt["total_steps"]
        agent.epsilon = ckpt["epsilon"]
        return agent
