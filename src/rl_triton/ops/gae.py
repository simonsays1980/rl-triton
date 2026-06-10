import torch
import triton

from rl_triton.kernels.gae import gae_fused_kernel
from rl_triton.ops._scan import _run_scan, _FLAT_MAX_SEQ_LEN

_WARPS = {512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32}


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
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

    V(s_{t+1}) is read directly from values[:, t+1]; the caller does NOT need to
    pass a separate next_values tensor.  For the last timestep, V(s_T) is taken
    from bootstrap_values (which is also A[T] = 0 for terminated episodes, or
    V(s_T) for truncated ones).

    Maps to the shared linear recurrence A[t] = u[t] + v[t] * A[t+1] with:
      u[t] = delta[t]
      v[t] = gamma * lambda * (1 - done[t])

    Dispatches to the fully-fused single-block kernel for seq_len <= 131072
    (delta and decay computed inside the kernel, no intermediate tensors),
    and falls back to the chunked scan + PyTorch elementwise ops for longer
    sequences.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        values:           V(s_t), same shape, float32, CUDA.
                          V(s_{t+1}) is read as values[:, t+1] inside the kernel.
        dones:            Episode termination flags (1.0=done), same shape, float32.
        gamma:            Discount factor.
        lambda_:          GAE trace parameter in [0, 1].
        bootstrap_values: Per-environment V(s_T) / boundary value A[T], shape [num_envs].
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Also used as V(s_{t+1}) at t=T-1 inside the kernel.
                          Defaults to zeros.

    Returns:
        advantages: A[t], shape [num_envs, seq_len], float32.
    """
    for name, t in [("rewards", rewards), ("values", values), ("dones", dones)]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"

    num_envs, seq_len = rewards.shape

    rewards = rewards.contiguous()
    values  = values.contiguous()
    dones   = dones.contiguous()

    if bootstrap_values is None:
        bootstrap_values = torch.zeros(num_envs, device=rewards.device, dtype=torch.float32)
    else:
        assert bootstrap_values.shape == (num_envs,), \
            f"bootstrap_values must have shape [{num_envs}], got {bootstrap_values.shape}"
        assert bootstrap_values.is_cuda, "bootstrap_values must be on CUDA"
        bootstrap_values = bootstrap_values.contiguous()

    out = torch.empty_like(rewards)

    if seq_len <= _FLAT_MAX_SEQ_LEN:
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        num_warps  = _WARPS.get(BLOCK_SIZE, 16)
        num_stages = 2 if BLOCK_SIZE >= 2048 else 1

        gae_fused_kernel[(num_envs,)](
            rewards, values, dones,
            out,
            bootstrap_values,
            seq_len,
            rewards.stride(0),
            gamma=gamma,
            lambda_=lambda_,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    # Chunked path for seq_len > 131072: PyTorch pre-computation + chunked scan.
    not_done    = 1.0 - dones
    next_values = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = bootstrap_values
    deltas = rewards + gamma * not_done * next_values - values
    decays = gamma * lambda_ * not_done
    return _run_scan(deltas, decays, bootstrap_values)
