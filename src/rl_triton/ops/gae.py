import torch

from rl_triton.ops._scan import _run_scan


def compute_gae_triton(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation via a backward associative scan.

    Recurrence:
      A[t] = delta[t] + gamma * lambda * (1 - done[t]) * A[t+1],  A[T] = bootstrap

    where delta[t] = r[t] + gamma * (1 - done[t]) * V(s_{t+1}) - V(s_t).

    Maps to the shared linear recurrence A[t] = u[t] + v[t] * A[t+1] with:
      u[t] = delta[t]
      v[t] = gamma * lambda * (1 - done[t])

    Dispatches to the flat single-block kernel for seq_len <= 131072 and falls
    back to the chunked kernel for longer sequences.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        values:           V(s_t), same shape, float32, CUDA.
        next_values:      V(s_{t+1}), same shape, float32, CUDA.
        dones:            Episode termination flags (1.0=done), same shape, float32.
        gamma:            Discount factor.
        lambda_:          GAE trace parameter in [0, 1].
        bootstrap_values: Per-environment boundary value A[T], shape [num_envs].
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Defaults to zeros.

    Returns:
        advantages: A[t], shape [num_envs, seq_len], float32.
    """
    for name, t in [("rewards", rewards), ("values", values), ("next_values", next_values), ("dones", dones)]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"

    not_done = 1.0 - dones
    deltas = rewards + gamma * not_done * next_values - values
    decays = gamma * lambda_ * not_done
    return _run_scan(deltas, decays, bootstrap_values)
