"""Minimal EnvProtocol-conforming synthetic env for the Part B async pipeline
benchmark. No simulator -- fixed-size random-ish transitions, matching this
repo's other benchmarks' convention of synthetic rollout tensors (see
resip_ppo_e2e_measurement.py's module docstring for the same scoping choice).

Must be picklable by reference for multiprocessing spawn (env_contract.py's
EnvFactory contract) -- SyntheticEnv and synthetic_env_factory are both
top-level, no closures/lambdas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _EnvState:
    obs: dict
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict
    final_observation: dict | None


class _Space:
    def __init__(self, shape):
        self.shape = shape


class SyntheticEnv:
    """Fixed obs_dim/action_dim, num_envs parallel envs, fixed episode length
    (truncation only, no early termination) -- exercises the truncation path
    with final_observation=None (collector falls back to next_obs, per
    resolve_terminal_observation_contract's documented fallback)."""

    def __init__(self, num_envs: int, obs_dim: int = 16, action_dim: int = 10, episode_len: int = 24):
        self._num_envs = num_envs
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._episode_len = episode_len
        self._steps = np.zeros(num_envs, dtype=np.int32)
        self._state: _EnvState | None = None
        self._rng = np.random.default_rng(0)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._obs_dim}

    @property
    def observation_space(self) -> Any:
        return _Space((self._obs_dim,))

    @property
    def action_space(self) -> Any:
        return _Space((self._action_dim,))

    @property
    def state(self) -> _EnvState | None:
        return self._state

    @property
    def cfg(self) -> Any:
        return {"max_episode_seconds": None, "ctrl_dt": 0.02}

    @property
    def play_capabilities(self) -> Any:
        return type("Caps", (), {"supports_physics_state_playback": False})()

    def init_state(self) -> _EnvState:
        obs, _ = self.reset(np.arange(self._num_envs, dtype=np.int32))
        self._state = _EnvState(
            obs=obs, reward=np.zeros(self._num_envs, dtype=np.float32),
            terminated=np.zeros(self._num_envs, dtype=bool),
            truncated=np.zeros(self._num_envs, dtype=bool), info={}, final_observation=None,
        )
        return self._state

    def reset(self, env_indices: np.ndarray | None = None) -> tuple[dict, dict]:
        if env_indices is None:
            env_indices = np.arange(self._num_envs, dtype=np.int32)
        self._steps[env_indices] = 0
        obs = {"obs": self._rng.standard_normal((self._num_envs, self._obs_dim)).astype(np.float32)}
        return obs, {}

    def step(self, actions: np.ndarray) -> _EnvState:
        self._steps += 1
        obs = {"obs": self._rng.standard_normal((self._num_envs, self._obs_dim)).astype(np.float32)}
        reward = self._rng.standard_normal(self._num_envs).astype(np.float32)
        truncated = self._steps >= self._episode_len
        terminated = np.zeros(self._num_envs, dtype=bool)
        if np.any(truncated):
            self._steps[truncated] = 0
        state = _EnvState(
            obs=obs, reward=reward, terminated=terminated, truncated=truncated,
            info={}, final_observation=None,
        )
        self._state = state
        return state

    def set_nan_guard(self, guard: Any) -> None:
        pass

    def close(self) -> None:
        pass


def synthetic_env_factory(num_envs: int, env_cfg_override: dict | None = None) -> SyntheticEnv:
    """Top-level, picklable-by-reference EnvFactory (env_contract.py contract)."""
    obs_dim = (env_cfg_override or {}).get("obs_dim", 16)
    action_dim = (env_cfg_override or {}).get("action_dim", 10)
    episode_len = (env_cfg_override or {}).get("episode_len", 24)
    return SyntheticEnv(num_envs, obs_dim=obs_dim, action_dim=action_dim, episode_len=episode_len)
