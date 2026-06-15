import torch
import triton

from rl_triton.kernels.gae import gae_fused_kernel
from rl_triton.ops._scan import _run_scan, _FLAT_MAX_SEQ_LEN

_WARPS = {512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32}


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float,
    lambda_: float,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation via a backward associative scan.

    Recurrence:

    - A[t] = delta[t] + gamma * lambda * (1 - terminated[t]) * A[t+1],  A[T] = bootstrap
    - delta[t] = r[t] + gamma * (1 - terminated[t]) * V(s_{t+1}) - V(s_t)
    
    V(s_{t+1}) is read from values[:, t+1]; no separate next_values tensor needed.

    Pass only true terminations in `terminateds`.  For a terminated step,
    terminated[t]=1 zeros the carry so no value bleeds across the episode
    boundary.  For a truncated step, set terminated[t]=0 and supply V(s_T)
    via `bootstrap_values` so the kernel bootstraps correctly from the
    continuation state.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        values:           V(s_t), [num_envs, seq_len], float32, CUDA.
        terminateds:      True termination flags only (1.0=terminated),
                          [num_envs, seq_len], float32, CUDA.
                          Do not pass (terminated | truncated) here — truncated steps
                          must have terminated[t]=0 with V(s_T) in bootstrap_values.
        gamma:            Discount factor.
        lambda_:          GAE trace parameter in [0, 1].
        bootstrap_values: V(s_T) per environment, shape [num_envs].
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Defaults to zeros.

    Returns:
        advantages: A[t], shape [num_envs, seq_len], float32.
    """
    for name, t in [("rewards", rewards), ("values", values), ("terminateds", terminateds)]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"

    num_envs, seq_len = rewards.shape

    rewards     = rewards.contiguous()
    values      = values.contiguous()
    terminateds = terminateds.contiguous()

    if bootstrap_values is None:
        bootstrap_values = torch.zeros(num_envs, device=rewards.device, dtype=torch.float32)
    else:
        assert bootstrap_values.shape == (num_envs,), \
            f"bootstrap_values must have shape [{num_envs}], got {bootstrap_values.shape}"
        assert bootstrap_values.is_cuda, "bootstrap_values must be on CUDA"
        bootstrap_values = bootstrap_values.contiguous()

    out = torch.empty_like(rewards)

    # Linear recurrence: u[t] = delta[t], v[t] = gamma * lambda * (1 - terminated[t]).
    # Fused kernel for seq_len <= 131072 (computes delta and decay inside the
    # kernel, no intermediate tensors). Chunked scan + PyTorch ops for longer.
    if seq_len <= _FLAT_MAX_SEQ_LEN:
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        num_warps  = _WARPS.get(BLOCK_SIZE, 16)
        num_stages = 2 if BLOCK_SIZE >= 2048 else 1

        gae_fused_kernel[(num_envs,)](
            rewards, values, terminateds,
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
    not_terminated = 1.0 - terminateds
    next_values    = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = bootstrap_values
    deltas = rewards + gamma * not_terminated * next_values - values
    decays = gamma * lambda_ * not_terminated
    return _run_scan(deltas, decays, bootstrap_values)
