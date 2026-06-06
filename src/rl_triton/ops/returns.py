import torch

from rl_triton.ops._scan import _run_scan, _run_scan_forward


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

    Limited to seq_len <= 131072 (flat kernel only; a chunked forward kernel
    has not been implemented — see NOTES.md).

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

    v = gamma * lambda_ * (1.0 - dones)
    return _run_scan_forward(gradients, v, seed_values)
