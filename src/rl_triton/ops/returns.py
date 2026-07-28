import torch
import triton

from rl_triton.kernels.discounted_returns_fused import discounted_returns_fused_kernel
from rl_triton.kernels.eligibility_traces_fused import eligibility_traces_fused_kernel
from rl_triton.kernels.lambda_returns_fused import lambda_returns_fused_kernel
from rl_triton.ops._scan import _run_scan, _run_scan_forward, _FLAT_MAX_SEQ_LEN, _CORRECTNESS_WARNINGS

# Below 512, BLOCK_SIZE used to fall through .get()'s default (16 warps) on
# both tables below — see src/rl_triton/ops/gae.py's _WARPS for the H200
# measurement basis (flat device time for num_warps in {1,2,4} at BLOCK_SIZE
# 8-128, driven by the 32-blocks/SM hard cap on Hopper, register-count-
# independent at this scale since none of these kernels spill at low warps).

# Lambda returns carries more registers (next_values, u computation) — same budget as GAE.
_WARPS_LAMBDA = {
    8: 2, 16: 2, 32: 2, 64: 2, 128: 2, 256: 4,
    512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32,
}

# Discounted returns and eligibility traces carry very few registers (2 loads, 1 scan).
# More warps fit without spilling, giving the scan tree more parallelism.
_WARPS_LIGHT = {
    8: 2, 16: 2, 32: 2, 64: 2, 128: 2, 256: 4,
    512: 8, 1024: 16, 2048: 16, 4096: 16, 8192: 32, 16384: 32,
}


def compute_lambda_returns(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor | None = None,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute TD(λ) targets (λ-returns) via a backward associative scan.

    Recurrence:

    - G[t] = r[t] + γ*(1-done[t]) * [(1-λ)*V(s_{t+1}) + λ*G[t+1]]
             + γ*truncated[t]*bootstrap[t]
    - done[t] = terminated[t] | truncated[t]

    Special cases: λ=0 reduces to one-step TD; λ=1 reduces to discounted returns.

    `terminated[t]` stops both the value bootstrap and trace propagation.
    `truncated[t]` stops trace propagation and injects bootstrap_values[env, t]
    = V(s_{t+1}^true) as the continuation value.  The caller must zero
    next_values[env, t] at truncated steps to avoid double-counting.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        next_values:      V(s_{t+1}), [num_envs, seq_len], float32, CUDA.
                          Must be zeroed by caller at truncated steps.
        terminateds:      True termination flags (1.0=terminated),
                          [num_envs, seq_len], float32, CUDA.
        truncateds:       Time-limit truncation flags (1.0=truncated),
                          [num_envs, seq_len], float32, CUDA.
                          If None, terminateds is used for both gating roles.
        gamma:            Discount factor (default 0.99).
        lambda_:          Trace parameter in [0, 1] (default 0.95).
        bootstrap_values: True continuation values V(s_{t+1}^true),
                          [num_envs, seq_len], float32, CUDA.
                          Nonzero at truncated steps and at t=T-1 when the window
                          ends mid-episode; zero elsewhere.
                          If None, defaults to all zeros.

    Returns:
        lambda_returns: G[t], shape [num_envs, seq_len], float32.
    """
    num_envs, seq_len = rewards.shape
    has_truncations   = truncateds is not None

    # Cheap structural checks — always-on.
    for name, t in [("rewards", rewards), ("next_values", next_values), ("terminateds", terminateds)]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"
    if has_truncations:
        assert truncateds.is_cuda,                "truncateds must be on CUDA"
        assert truncateds.dtype == torch.float32, "truncateds: expected float32"
        assert truncateds.shape == rewards.shape, \
            f"truncateds shape {truncateds.shape} != rewards shape {rewards.shape}"
    if bootstrap_values is not None:
        assert bootstrap_values.is_cuda,                "bootstrap_values must be on CUDA"
        assert bootstrap_values.dtype == torch.float32, "bootstrap_values: expected float32"
        assert bootstrap_values.shape == rewards.shape, \
            f"bootstrap_values shape {bootstrap_values.shape} != rewards shape {rewards.shape}"

    # Expensive tensor scans — correctness-warning path only.
    if _CORRECTNESS_WARNINGS():
        if has_truncations:
            assert not (terminateds.bool() & truncateds.bool()).any(), \
                "terminated and truncated are mutually exclusive: a step cannot be both"

    rewards     = rewards.contiguous()
    next_values = next_values.contiguous()
    terminateds = terminateds.contiguous()
    if has_truncations:
        truncateds = truncateds.contiguous()
    if bootstrap_values is not None:
        bootstrap_values = bootstrap_values.contiguous()

    out = torch.empty_like(rewards)

    if seq_len <= _FLAT_MAX_SEQ_LEN:
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        num_warps  = _WARPS_LAMBDA.get(BLOCK_SIZE, 16)
        num_stages = 2 if BLOCK_SIZE >= 2048 else 1
        if has_truncations:
            if bootstrap_values is None:
                bootstrap_values = torch.zeros_like(rewards)
            lambda_returns_fused_kernel[(num_envs,)](
                rewards, next_values, terminateds, truncateds,
                out, bootstrap_values,
                seq_len, rewards.stride(0),
                gamma=gamma, lambda_=lambda_,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages,
                HAS_TRUNCATIONS=True, HAS_BOOTSTRAP=True,
            )
        else:
            scalar_bootstrap = bootstrap_values[:, -1].contiguous() \
                               if bootstrap_values is not None else None
            has_bootstrap = scalar_bootstrap is not None
            lambda_returns_fused_kernel[(num_envs,)](
                rewards, next_values, terminateds, None,
                out, scalar_bootstrap,
                seq_len, rewards.stride(0),
                gamma=gamma, lambda_=lambda_,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages,
                HAS_TRUNCATIONS=False, HAS_BOOTSTRAP=has_bootstrap,
            )
        return out

    # Chunked fallback for seq_len > 131072.
    if not has_truncations:
        truncateds = torch.zeros_like(terminateds)
    if bootstrap_values is None:
        bootstrap_values = torch.zeros_like(rewards)
    not_done = 1.0 - (terminateds + truncateds).clamp(max=1.0)
    carry    = bootstrap_values[:, -1]
    u = rewards + gamma * (1.0 - lambda_) * not_done * next_values \
        + gamma * truncateds * bootstrap_values
    v = gamma * lambda_ * not_done
    return _run_scan(u, v, carry)


def compute_discounted_returns(
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor | None = None,
    gamma: float = 0.99,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute discounted returns (reward-to-go) via a backward associative scan.

    Recurrence:

    - G[t] = r[t] + γ*(1-done[t])*G[t+1] + γ*truncated[t]*bootstrap[t]
    - done[t] = terminated[t] | truncated[t]

    `terminated[t]` stops the return propagation (episode ends, G[t+1]=0).
    `truncated[t]` also stops propagation but injects bootstrap_values[env, t]
    = V(s_{t+1}^true) as the continuation value instead of zero.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        terminateds:      True termination flags (1.0=terminated),
                          [num_envs, seq_len], float32, CUDA.
        truncateds:       Time-limit truncation flags (1.0=truncated),
                          [num_envs, seq_len], float32, CUDA.
                          If None, terminateds is used for both gating roles.
        gamma:            Discount factor (default 0.99).
        bootstrap_values: True continuation values V(s_{t+1}^true),
                          [num_envs, seq_len], float32, CUDA.
                          Nonzero at truncated steps and at t=T-1 when the window
                          ends mid-episode; zero elsewhere.
                          If None, defaults to all zeros.

    Returns:
        returns: G[t], shape [num_envs, seq_len], float32.
    """
    num_envs, seq_len = rewards.shape
    has_truncations   = truncateds is not None

    # Cheap structural checks — always-on.
    for name, t in [("rewards", rewards), ("terminateds", terminateds)]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"
    if has_truncations:
        assert truncateds.is_cuda,                "truncateds must be on CUDA"
        assert truncateds.dtype == torch.float32, "truncateds: expected float32"
        assert truncateds.shape == rewards.shape, \
            f"truncateds shape {truncateds.shape} != rewards shape {rewards.shape}"
    if bootstrap_values is not None:
        assert bootstrap_values.is_cuda,                "bootstrap_values must be on CUDA"
        assert bootstrap_values.dtype == torch.float32, "bootstrap_values: expected float32"
        assert bootstrap_values.shape == rewards.shape, \
            f"bootstrap_values shape {bootstrap_values.shape} != rewards shape {rewards.shape}"

    # Expensive tensor scans — correctness-warning path only.
    if _CORRECTNESS_WARNINGS():
        if has_truncations:
            assert not (terminateds.bool() & truncateds.bool()).any(), \
                "terminated and truncated are mutually exclusive: a step cannot be both"

    rewards     = rewards.contiguous()
    terminateds = terminateds.contiguous()
    if has_truncations:
        truncateds = truncateds.contiguous()
    if bootstrap_values is not None:
        bootstrap_values = bootstrap_values.contiguous()

    out = torch.empty_like(rewards)

    if seq_len <= _FLAT_MAX_SEQ_LEN:
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        num_warps  = _WARPS_LIGHT.get(BLOCK_SIZE, 16)
        num_stages = 2 if BLOCK_SIZE >= 1024 else 1
        if has_truncations:
            if bootstrap_values is None:
                bootstrap_values = torch.zeros_like(rewards)
            discounted_returns_fused_kernel[(num_envs,)](
                rewards, terminateds, truncateds,
                out, bootstrap_values,
                seq_len, rewards.stride(0),
                gamma=gamma,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages,
                HAS_TRUNCATIONS=True, HAS_BOOTSTRAP=True,
            )
        else:
            scalar_bootstrap = bootstrap_values[:, -1].contiguous() \
                               if bootstrap_values is not None else None
            has_bootstrap = scalar_bootstrap is not None
            discounted_returns_fused_kernel[(num_envs,)](
                rewards, terminateds, None,
                out, scalar_bootstrap,
                seq_len, rewards.stride(0),
                gamma=gamma,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages,
                HAS_TRUNCATIONS=False, HAS_BOOTSTRAP=has_bootstrap,
            )
        return out

    # Chunked fallback for seq_len > 131072.
    if not has_truncations:
        truncateds = torch.zeros_like(terminateds)
    if bootstrap_values is None:
        bootstrap_values = torch.zeros_like(rewards)
    done  = (terminateds + truncateds).clamp(max=1.0)
    carry = bootstrap_values[:, -1]
    u = rewards + gamma * truncateds * bootstrap_values
    v = gamma * (1.0 - done)
    return _run_scan(u, v, carry)


def compute_eligibility_traces(
    gradients: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
    seed_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute accumulating eligibility traces via a forward associative scan.

    Recurrence:

    - z[t] = g[t] + γ·λ·(1 - done[t]) * z[t-1],  z[-1] = seed

    g[t] is the per-step input: the value-function gradient ∇_w V̂(s_t) for
    general function approximation, or the feature vector x(s_t) in the linear
    case.  Unlike all other kernels, this scan runs forward in time.
    Limited to seq_len <= 131072.

    Args:
        gradients:   g[t] — value-function gradients or feature vectors,
                     [num_envs, seq_len], float32, CUDA.
        dones:       Episode termination flags (1.0=done), [num_envs, seq_len], float32, CUDA.
        gamma:       Discount factor.
        lambda_:     Trace decay parameter.
        seed_values: Initial trace z[-1] per environment, shape [num_envs].
                     Defaults to zeros.

    Returns:
        traces: z[t], shape [num_envs, seq_len], float32.
    """
    num_envs, seq_len = gradients.shape

    # Cheap structural checks — always-on.
    assert gradients.is_cuda and dones.is_cuda, "gradients and dones must be on CUDA"
    assert gradients.dtype == torch.float32, f"gradients: expected float32, got {gradients.dtype}"
    assert dones.dtype == torch.float32,     f"dones: expected float32, got {dones.dtype}"
    assert gradients.shape == dones.shape,   "gradients and dones must have the same shape"
    if seed_values is not None:
        assert seed_values.shape == (num_envs,), \
            f"seed_values must have shape [{num_envs}], got {seed_values.shape}"
        assert seed_values.is_cuda, "seed_values must be on CUDA"

    assert seq_len <= _FLAT_MAX_SEQ_LEN, (
        f"seq_len={seq_len} exceeds the flat kernel limit {_FLAT_MAX_SEQ_LEN}. "
        "A chunked forward scan kernel has not been implemented yet."
    )

    gradients = gradients.contiguous()
    dones     = dones.contiguous()

    has_seed = seed_values is not None
    if has_seed:
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
        HAS_SEED=has_seed,
    )
    return out
