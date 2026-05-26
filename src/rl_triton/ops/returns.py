import torch

from rl_triton.ops._scan import _run_scan


def compute_discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute discounted returns (reward-to-go) via a backward associative scan.

    Recurrence: G[t] = r[t] + gamma * (1 - done[t]) * G[t+1], G[T] = bootstrap.

    Maps to the shared linear recurrence A[t] = u[t] + v[t] * A[t+1] with:
      u[t] = rewards[t]
      v[t] = gamma * (1 - dones[t])

    Dispatches to the flat single-block kernel for seq_len <= 131072 and falls
    back to the chunked kernel for longer sequences.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        dones:            Episode termination flags (1.0=done), same shape, float32.
        gamma:            Discount factor.
        bootstrap_values: Per-environment boundary value G[T], shape [num_envs].
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Defaults to zeros.

    Returns:
        returns: Discounted returns G[t], shape [num_envs, seq_len], float32.
    """
    assert rewards.is_cuda and dones.is_cuda, "rewards and dones must be on CUDA"
    assert rewards.dtype == torch.float32, f"rewards: expected float32, got {rewards.dtype}"
    assert dones.dtype == torch.float32,   f"dones: expected float32, got {dones.dtype}"
    assert rewards.shape == dones.shape,   "rewards and dones must have the same shape"

    v = gamma * (1.0 - dones)
    return _run_scan(rewards, v, bootstrap_values)
