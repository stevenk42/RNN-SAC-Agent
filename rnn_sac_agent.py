from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class BeliefEncoder(nn.Module):
    """Encodes observation sequences into belief states using a GRU."""

    def __init__(self, obs_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self, obs_seq: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # obs_seq: [batch, seq_len, obs_dim]
        batch_size, _, _ = obs_seq.shape
        projected = F.relu(self.fc(obs_seq))

        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=obs_seq.device)

        output, hidden = self.gru(projected, hidden)
        return output, hidden


class DiscreteActor(nn.Module):
    """Categorical policy over discrete actions given the belief state."""

    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, belief: torch.Tensor) -> Categorical:
        logits = self.net(belief)
        return Categorical(logits=logits)


class DiscreteCritic(nn.Module):
    """Q-value estimator for discrete actions."""

    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        return self.net(belief)


class SequenceReplayBuffer:
    """Stores entire episodes and samples fixed-length sequences for BPTT."""

    def __init__(self, capacity: int, seq_len: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.episodes: List[Dict[str, List]] = []
        self.ptr = 0

    def push(self, episode: Dict[str, List]) -> None:
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
        else:
            self.episodes[self.ptr] = episode
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        if not self.episodes:
            raise ValueError("Cannot sample from an empty buffer.")

        indices = np.random.choice(len(self.episodes), batch_size)

        obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        act_batch = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        rew_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        done_batch = np.ones((batch_size, self.seq_len), dtype=np.float32)

        for i, idx in enumerate(indices):
            ep = self.episodes[idx]
            ep_len = len(ep["rew"])
            if ep_len == 0:
                continue

            start_max = max(ep_len - self.seq_len, 0)
            start = np.random.randint(0, start_max + 1)
            end = min(start + self.seq_len, ep_len)
            length = end - start

            obs_seq = ep["obs"][start:end]
            act_seq = ep["act"][start:end]
            rew_seq = ep["rew"][start:end]
            done_seq = ep["done"][start:end]

            obs_batch[i, :length] = np.asarray(obs_seq, dtype=np.float32)
            act_batch[i, :length] = np.asarray(act_seq, dtype=np.int64)
            rew_batch[i, :length] = np.asarray(rew_seq, dtype=np.float32)
            done_batch[i, :length] = np.asarray(done_seq, dtype=np.float32)

            if length < self.seq_len:
                done_batch[i, length:] = 1.0

        return (
            torch.as_tensor(obs_batch, device=device),
            torch.as_tensor(act_batch, device=device),
            torch.as_tensor(rew_batch, device=device),
            torch.as_tensor(done_batch, device=device),
        )

    def __len__(self) -> int:
        return len(self.episodes)


class RNNSACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha: float = 0.2,
        seq_len: int = 16,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.alpha = alpha
        self.seq_len = seq_len

        self.encoder = BeliefEncoder(obs_dim, hidden_dim).to(self.device)
        self.actor = DiscreteActor(hidden_dim, action_dim).to(self.device)
        self.q1 = DiscreteCritic(hidden_dim, action_dim).to(self.device)
        self.q2 = DiscreteCritic(hidden_dim, action_dim).to(self.device)

        self.q1_target = DiscreteCritic(hidden_dim, action_dim).to(self.device)
        self.q2_target = DiscreteCritic(hidden_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        self.buffer = SequenceReplayBuffer(capacity=1000, seq_len=seq_len, obs_dim=obs_dim)
        self.hidden: Optional[torch.Tensor] = None

    def reset_hidden(self) -> None:
        self.hidden = None

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)

        with torch.no_grad():
            belief, self.hidden = self.encoder(obs_tensor, self.hidden)
            dist = self.actor(belief.squeeze(1))

            if evaluate:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()

        return int(action.item())

    def train(self, batch_size: int = 32) -> None:
        if len(self.buffer) < batch_size:
            return

        obs_seq, act_seq, rew_seq, done_seq = self.buffer.sample(batch_size, self.device)

        belief_seq, _ = self.encoder(obs_seq)

        b = belief_seq.reshape(-1, belief_seq.size(-1))
        a = act_seq.reshape(-1)
        r = rew_seq.reshape(-1)
        d = done_seq.reshape(-1)

        with torch.no_grad():
            next_belief_seq = torch.cat(
                [belief_seq[:, 1:], torch.zeros_like(belief_seq[:, -1:])], dim=1
            )
            next_b = next_belief_seq.reshape(-1, next_belief_seq.size(-1))

            next_dist = self.actor(next_b)
            next_probs = next_dist.probs
            next_log_probs = F.log_softmax(next_dist.logits, dim=-1)

            target_q1 = self.q1_target(next_b)
            target_q2 = self.q2_target(next_b)
            min_target_q = torch.min(target_q1, target_q2)

            target_v = (next_probs * (min_target_q - self.alpha * next_log_probs)).sum(dim=1)
            q_target = r + self.gamma * target_v * (1.0 - d)

        q1_vals = self.q1(b).gather(1, a.unsqueeze(-1)).squeeze(-1)
        q2_vals = self.q2(b).gather(1, a.unsqueeze(-1)).squeeze(-1)

        q1_loss = F.mse_loss(q1_vals, q_target)
        q2_loss = F.mse_loss(q2_vals, q_target)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self.encoder_optimizer.step()

        current_belief = b.detach()
        dist = self.actor(current_belief)
        probs = dist.probs
        log_probs = F.log_softmax(dist.logits, dim=-1)

        q1_current = self.q1(current_belief)
        q2_current = self.q2(current_belief)
        min_q_current = torch.min(q1_current, q2_current)

        actor_loss = (probs * (self.alpha * log_probs - min_q_current)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.q1, self.q1_target, tau=0.005)
        self._soft_update(self.q2, self.q2_target, tau=0.005)

    def _soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
