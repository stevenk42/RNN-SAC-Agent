from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class BlindKeyDoorEnvConfig:
    """Configuration for the BlindKeyDoor environment."""

    size: int = 5
    max_steps: int = 100
    slip_prob: float = 0.0
    render_mode: Optional[str] = None


class BlindKeyDoorEnv(gym.Env):
    """A partially observable grid world with a key and a locked door.

    The agent must pick up the key before reaching the locked door.  The
    observation is extremely limited: ``[door_open, has_key, last_action]``.

    * ``door_open`` (0 or 1): indicates whether the door has been opened.
    * ``has_key`` (0 or 1): indicates whether the agent currently holds the key.
    * ``last_action`` (0-4): index of the action executed on the previous step.
      ``4`` indicates that no action has been taken yet (i.e. immediately after
      reset).

    The action space is ``{0: up, 1: down, 2: left, 3: right, 4: noop}``.
    Movement actions can randomly ``slip`` and execute a different action,
    depending on ``slip_prob``.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    ACTIONS = {
        0: np.array([-1, 0], dtype=np.int8),  # Up
        1: np.array([1, 0], dtype=np.int8),   # Down
        2: np.array([0, -1], dtype=np.int8),  # Left
        3: np.array([0, 1], dtype=np.int8),   # Right
        4: np.array([0, 0], dtype=np.int8),   # No-op
    }

    SYMBOLS = {
        "empty": ".",
        "agent": "A",
        "key": "K",
        "door": "D",
        "door_open": "O",
    }

    def __init__(
        self,
        size: int = 5,
        max_steps: int = 100,
        slip_prob: float = 0.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if size < 3:
            raise ValueError("Grid size must be at least 3x3 to fit start, key, and door.")
        if not (0.0 <= slip_prob <= 1.0):
            raise ValueError("slip_prob must be within [0, 1].")

        self.config = BlindKeyDoorEnvConfig(size=size, max_steps=max_steps, slip_prob=slip_prob, render_mode=render_mode)
        self.size = size
        self.max_steps = max_steps
        self.slip_prob = slip_prob
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, float(len(self.ACTIONS) - 1)], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self._agent_pos = np.zeros(2, dtype=np.int8)
        self._key_pos = np.zeros(2, dtype=np.int8)
        self._door_pos = np.zeros(2, dtype=np.int8)
        self._step_count = 0
        self._has_key = False
        self._door_open = False
        self._last_action = len(self.ACTIONS) - 1  # 4 -> "no previous action"

    def seed(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)

        self._step_count = 0
        self._has_key = False
        self._door_open = False
        self._last_action = len(self.ACTIONS) - 1

        # Layout: agent starts bottom-left, door bottom-right, key top-right.
        self._agent_pos = np.array([self.size - 1, 0], dtype=np.int8)
        self._door_pos = np.array([self.size - 1, self.size - 1], dtype=np.int8)
        self._key_pos = np.array([0, self.size - 1], dtype=np.int8)

        # Randomly add one vertical wall to enforce memory usage.
        self._wall_column = self.rng.integers(1, self.size - 1)

        return self._get_obs(), {}

    def step(self, action: int):  # type: ignore[override]
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        executed_action = self._apply_slip(action)
        movement = self.ACTIONS[int(executed_action)]

        if executed_action != 4:
            new_pos = self._agent_pos + movement
            if self._is_valid_position(new_pos):
                self._agent_pos = new_pos

        reward = 0.0
        terminated = False

        if not self._has_key and np.array_equal(self._agent_pos, self._key_pos):
            self._has_key = True
            reward += 1.0

        if self._has_key and not self._door_open and np.array_equal(self._agent_pos, self._door_pos):
            self._door_open = True
            reward += 10.0
            terminated = True

        reward -= 0.01  # small step penalty

        self._last_action = int(executed_action)
        self._step_count += 1
        truncated = self._step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return np.array([
            float(self._door_open),
            float(self._has_key),
            float(self._last_action),
        ], dtype=np.float32)

    def _apply_slip(self, action: int) -> int:
        if self.rng.random() < self.slip_prob:
            return int(self.rng.integers(0, len(self.ACTIONS)))
        return int(action)

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        if np.any(pos < 0) or np.any(pos >= self.size):
            return False
        # Impassable wall column except for a single doorway on row 2.
        if pos[1] == self._wall_column and pos[0] != 1:
            return False
        return True

    # ------------------------------------------------------------------
    # Rendering utilities
    # ------------------------------------------------------------------
    def render(self):  # type: ignore[override]
        if self.render_mode != "ansi":
            raise NotImplementedError("Only ansi render mode is supported.")
        return self._render_text()

    def _render_text(self) -> str:
        grid = [[self.SYMBOLS["empty"] for _ in range(self.size)] for _ in range(self.size)]

        if not self._door_open:
            grid[self._door_pos[0]][self._door_pos[1]] = self.SYMBOLS["door"]
        else:
            grid[self._door_pos[0]][self._door_pos[1]] = self.SYMBOLS["door_open"]

        if not self._has_key:
            grid[self._key_pos[0]][self._key_pos[1]] = self.SYMBOLS["key"]

        grid[self._agent_pos[0]][self._agent_pos[1]] = self.SYMBOLS["agent"]

        lines = [" ".join(row) for row in grid]
        lines.append(f"door_open={self._door_open}, has_key={self._has_key}, last_action={self._last_action}")
        return "\n".join(lines)

    def close(self):  # type: ignore[override]
        pass
