import torch

from rl_triton.ops._scan import _FLAT_MAX_SEQ_LEN, _run_scan, _run_scan_forward


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

    Special cases:
      λ=0: reduces to one-step TD targets  G[t] = r[t] + γ(1-d[t])*V(s_{t+1})
      λ=1: reduces to discounted returns   G[t] = r[t] + γ(1-d[t])*G[t+1]

    The natural bootstrap for a truncated episode is V(s_T) — not zero — because
    the recurrence mixes V(s_{t+1}) into u at every step; zeroing the boundary
    would incorrectly cut the value estimate at the last step.  Pass
    bootstrap_values=next_values[:, -1] for truncated episodes, or omit it for
    terminated ones (bootstrap=0 is correct when d[T-1]=1).

    Dispatches to the flat single-block kernel for seq_len <= 131072 and falls
    back to the chunked kernel for longer sequences.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        next_values:      V(s_{t+1}), same shape, float32, CUDA.
        dones:            Episode termination flags (1.0=done), same shape, float32.
        gamma:            Discount factor.
        lambda_:          Trace parameter in [0, 1].
        bootstrap_values: Per-environment boundary value G[T], shape [num_envs].
                          Defaults to zeros (terminated episodes).

    Returns:
        lambda_returns: G[t], shape [num_envs, seq_len], float32.
    """
    for name, t in [("rewards", rewards), ("next_values", next_values), ("dones", dones)]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"

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


def compute_eligibility_traces(
    features: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float,
    seed_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute accumulating eligibility traces via a forward associative scan.

    Recurrence: e[t] = x[t] + gamma * lambda * (1 - done[t]) * e[t-1], e[-1] = seed.

    Unlike all other estimators in this package, eligibility traces accumulate
    FORWARD in time: each trace depends on the current feature and the trace
    from the previous step, not the next step.

    Maps to the forward linear recurrence e[t] = u[t] + v[t] * e[t-1] with:
      u[t] = features[t]
      v[t] = gamma * lambda * (1 - dones[t])

    Limited to seq_len <= 131072 (flat kernel only; a chunked forward kernel
    has not been implemented — see NOTES.md).

    Args:
        features:    Input features x[t], [num_envs, seq_len], float32, CUDA.
        dones:       Episode termination flags (1.0=done), same shape, float32.
        gamma:       Discount factor.
        lambda_:     Trace decay parameter.
        seed_values: Initial trace e[-1] per environment, shape [num_envs].
                     Defaults to zeros (traces start from scratch).

    Returns:
        traces: e[t], shape [num_envs, seq_len], float32.
    """
    assert features.is_cuda and dones.is_cuda, "features and dones must be on CUDA"
    assert features.dtype == torch.float32, f"features: expected float32, got {features.dtype}"
    assert dones.dtype == torch.float32,    f"dones: expected float32, got {dones.dtype}"
    assert features.shape == dones.shape,   "features and dones must have the same shape"

    v = gamma * lambda_ * (1.0 - dones)
    return _run_scan_forward(features, v, seed_values)


def compute_episodic_prefix_sum(
    inputs: torch.Tensor,
    dones: torch.Tensor,
    seed_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cumulative sum that resets to zero at episode boundaries.

    Recurrence:  C[t] = x[t] + (1 - d[t]) * C[t-1],  C[-1] = seed

    Maps to the forward linear recurrence e[t] = u[t] + v[t] * e[t-1] with:
      u[t] = inputs[t]
      v[t] = 1 - dones[t]   (1.0 while episode ongoing, 0.0 at terminal step)

    When d[t] = 1, v[t] = 0 and C[t] = x[t]: the accumulation resets.
    When d[t] = 0, v[t] = 1 and C[t] = x[t] + C[t-1]: standard prefix sum.

    Limited to seq_len <= 131072 (flat kernel only; see NOTES.md).

    Args:
        inputs:      Values to accumulate x[t], [num_envs, seq_len], float32, CUDA.
        dones:       Episode termination flags (1.0 = done), same shape, float32.
        seed_values: Initial carry C[-1] per environment, shape [num_envs].
                     Defaults to zeros (accumulation starts from scratch).

    Returns:
        prefix_sums: C[t], shape [num_envs, seq_len], same dtype as inputs.
    """
    assert inputs.is_cuda and dones.is_cuda, "inputs and dones must be on CUDA"
    assert inputs.dtype == torch.float32, f"inputs: expected float32, got {inputs.dtype}"
    assert dones.dtype == torch.float32,  f"dones: expected float32, got {dones.dtype}"
    assert inputs.shape == dones.shape,   "inputs and dones must have the same shape"

    v = (1.0 - dones).to(inputs.dtype)
    return _run_scan_forward(inputs, v, seed_values)
