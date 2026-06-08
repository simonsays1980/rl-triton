import torch
import triton

from rl_triton.kernels.discounted_returns_fused import discounted_returns_fused_kernel
from rl_triton.kernels.eligibility_traces_fused import eligibility_traces_fused_kernel
from rl_triton.kernels.lambda_returns_fused import lambda_returns_fused_kernel
from rl_triton.ops._scan import _run_scan, _run_scan_forward, _FLAT_MAX_SEQ_LEN

# Lambda returns carries more registers (next_values, u computation) — same budget as GAE.
_WARPS_LAMBDA = {512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32}

# Discounted returns and eligibility traces carry very few registers (2 loads, 1 scan).
# More warps fit without spilling, giving the scan tree more parallelism.
_WARPS_LIGHT = {512: 8, 1024: 16, 2048: 16, 4096: 16, 8192: 32, 16384: 32}


def compute_lambda_returns(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute TD(λ) targets (λ-returns) via a backward associative scan.

    Recurrence:
      G[t] = r[t] + γ(1-d[t]) * [(1-λ)*V(s_{t+1}) + λ*G[t+1]],  G[T] = bootstrap

    Maps to the shared linear recurrence A[t] = u[t] + v[t] * A[t+1] with:
      u[t] = r[t] + γ * (1-λ) * (1-d[t]) * V(s_{t+1})
      v[t] = γ * λ * (1-d[t])

    V(s_{t+1}) is read directly from next_values[:, t] — the caller supplies this
    as a pre-shifted values tensor (next_values[:, t] = values[:, t+1]).  For
    seq_len <= 131072 the fused kernel instead reads values[t+1] directly from
    the values tensor, so next_values is only used as the source for V(s_{t+1})
    — it does not need to be a shifted copy; passing the raw values tensor works
    and avoids an extra allocation when the caller has values available.

    Special cases:
      λ=0: reduces to one-step TD targets  G[t] = r[t] + γ(1-d[t])*V(s_{t+1})
      λ=1: reduces to discounted returns   G[t] = r[t] + γ(1-d[t])*G[t+1]

    Dispatches to the fully-fused single-block kernel for seq_len <= 131072 and
    falls back to the chunked scan for longer sequences.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        next_values:      V(s_{t+1}), same shape, float32, CUDA.
                          For the fused path next_values[:, t] = values[:, t+1]
                          is reconstructed inside the kernel; bootstrap_values
                          provides V(s_T) at the boundary.
        dones:            Episode termination flags (1.0=done), same shape, float32.
        gamma:            Discount factor.
        lambda_:          Trace parameter in [0, 1].
        bootstrap_values: Per-environment boundary value G[T] / V(s_T),
                          shape [num_envs]. Defaults to zeros (terminated episodes).

    Returns:
        lambda_returns: G[t], shape [num_envs, seq_len], float32.
    """
    for name, t in [("rewards", rewards), ("next_values", next_values), ("dones", dones)]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"

    num_envs, seq_len = rewards.shape

    rewards     = rewards.contiguous()
    next_values = next_values.contiguous()
    dones       = dones.contiguous()

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
        num_warps  = _WARPS_LAMBDA.get(BLOCK_SIZE, 16)
        num_stages = 2 if BLOCK_SIZE >= 2048 else 1
        lambda_returns_fused_kernel[(num_envs,)](
            rewards, next_values, dones,
            out, bootstrap_values,
            seq_len, rewards.stride(0),
            gamma=gamma, lambda_=lambda_,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    # Chunked fallback for seq_len > 131072.
    not_done = 1.0 - dones
    u = rewards + gamma * (1.0 - lambda_) * not_done * next_values
    v = gamma * lambda_ * not_done
    return _run_scan(u, v, bootstrap_values)


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

    Dispatches to the fully-fused single-block kernel for seq_len <= 131072 and
    falls back to the chunked scan for longer sequences.

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

    num_envs, seq_len = rewards.shape

    rewards = rewards.contiguous()
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
        num_warps  = _WARPS_LIGHT.get(BLOCK_SIZE, 16)
        num_stages = 2 if BLOCK_SIZE >= 1024 else 1
        discounted_returns_fused_kernel[(num_envs,)](
            rewards, dones,
            out, bootstrap_values,
            seq_len, rewards.stride(0),
            gamma=gamma,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    # Chunked fallback for seq_len > 131072.
    v = gamma * (1.0 - dones)
    return _run_scan(rewards, v, bootstrap_values)


def compute_eligibility_traces(
    gradients: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
    seed_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute accumulating eligibility traces via a forward associative scan.

    Recurrence: z[t] = g[t] + gamma * lambda * (1 - done[t]) * z[t-1], z[-1] = seed.

    g[t] is the per-step input: the value-function gradient ∇_w V̂(s_t, w_t)
    for general FA, or the feature vector x(s_t) in the linear special case.

    Unlike all other estimators in this package, eligibility traces accumulate
    FORWARD in time: each trace depends on the current input and the trace
    from the previous step, not the next step.

    Maps to the forward linear recurrence z[t] = u[t] + v[t] * z[t-1] with:
      u[t] = gradients[t]
      v[t] = gamma * lambda * (1 - dones[t])

    Dispatches to the fully-fused single-block kernel for seq_len <= 131072.
    Longer sequences are not supported (chunked forward scan not implemented).

    Args:
        gradients:   Per-step inputs g[t] — value-function gradients (or feature
                     vectors under linear FA), [num_envs, seq_len], float32, CUDA.
        dones:       Episode termination flags (1.0=done), same shape, float32.
        gamma:       Discount factor.
        lambda_:     Trace decay parameter.
        seed_values: Initial trace z[-1] per environment, shape [num_envs].
                     Defaults to zeros (traces start from scratch).

    Returns:
        traces: z[t], shape [num_envs, seq_len], float32.
    """
    assert gradients.is_cuda and dones.is_cuda, "gradients and dones must be on CUDA"
    assert gradients.dtype == torch.float32, f"gradients: expected float32, got {gradients.dtype}"
    assert dones.dtype == torch.float32,     f"dones: expected float32, got {dones.dtype}"
    assert gradients.shape == dones.shape,   "gradients and dones must have the same shape"

    num_envs, seq_len = gradients.shape

    assert seq_len <= _FLAT_MAX_SEQ_LEN, (
        f"seq_len={seq_len} exceeds the flat kernel limit {_FLAT_MAX_SEQ_LEN}. "
        "A chunked forward scan kernel has not been implemented yet."
    )

    gradients = gradients.contiguous()
    dones     = dones.contiguous()

    if seed_values is None:
        seed_values = torch.zeros(num_envs, device=gradients.device, dtype=torch.float32)
    else:
        assert seed_values.shape == (num_envs,), \
            f"seed_values must have shape [{num_envs}], got {seed_values.shape}"
        assert seed_values.is_cuda, "seed_values must be on CUDA"
        seed_values = seed_values.contiguous()

    out = torch.empty_like(gradients)

    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    num_warps  = _WARPS_LIGHT.get(BLOCK_SIZE, 16)
    num_stages = 2 if BLOCK_SIZE >= 1024 else 1

    eligibility_traces_fused_kernel[(num_envs,)](
        gradients, dones,
        out, seed_values,
        seq_len, gradients.stride(0),
        gamma=gamma, lambda_=lambda_,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
