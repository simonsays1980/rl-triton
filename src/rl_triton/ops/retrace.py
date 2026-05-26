import torch

from rl_triton.ops._scan import _run_scan


def compute_retrace_triton(
    action_probs_target: torch.Tensor,
    action_probs_behavior: torch.Tensor,
    q_values: torch.Tensor,
    next_q_values_all: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambda_: float = 1.0,
    c_bar: float = 1.0,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Retrace(λ) Q-value targets via a backward associative scan.

    Retrace(λ) (Munos et al. 2016) corrects off-policy Q-value estimates using
    truncated importance sampling traces.  Unlike V-Trace, IS ratios are not
    applied to the TD error itself — only to the decay factor.

    Discrete actions only.  Retrace requires E_π[Q(s_{t+1},a)] as an exact sum
    over all actions, which is tractable only for discrete action spaces.  For
    continuous actions this expectation is an integral with no closed form;
    use V-Trace (compute_vtrace_triton) instead, which only needs log-density
    ratios at the taken action.

    Recurrence:
      Δ[t] = δ[t] + decay[t] * Δ[t+1],   Δ[T] = bootstrap
      δ[t]     = r[t] + γ * E_π[Q(s_{t+1},a)] * (1-done[t]) - Q(s_t, a_t)
      decay[t] = γ * c[t+1] * (1-done[t])
      c[t]     = λ * min(c_bar, π(a_t|s_t) / μ(a_t|s_t))

    Note: decay at time t uses c[t+1] (one step ahead).  The last step uses
    c=0 (no lookahead beyond the trajectory end).

    Q-value targets:
      Q_ret[t] = Q(s_t, a_t) + Δ[t]

    Dispatches to the flat single-block kernel for seq_len <= 131072 and falls
    back to the chunked kernel for longer sequences.

    Args:
        action_probs_target:   Target policy probabilities over all actions,
                               [num_envs, seq_len, num_actions], float32, CUDA.
        action_probs_behavior: Behavior policy probability of the taken action,
                               [num_envs, seq_len], float32, CUDA.
        q_values:              Q(s_t, a_t) for the taken action,
                               [num_envs, seq_len], float32, CUDA.
        next_q_values_all:     Q(s_{t+1}, a) for all actions,
                               [num_envs, seq_len, num_actions], float32, CUDA.
        actions:               Indices of the taken action,
                               [num_envs, seq_len], int64, CUDA.
        rewards:               Per-step rewards, [num_envs, seq_len], float32, CUDA.
        dones:                 Episode termination flags (1.0=done),
                               [num_envs, seq_len], float32, CUDA.
        gamma:                 Discount factor.
        lambda_:               Trace decay parameter (default 1.0).
        c_bar:                 IS ratio clip for trace weights (default 1.0).
        bootstrap_values:      Per-environment boundary Δ[T], shape [num_envs].
                               Use Q_ret[T] - Q(s_T, a_T) for truncated episodes,
                               0 for terminated ones.  Defaults to zeros.

    Returns:
        retrace_targets: Q_ret[t], shape [num_envs, seq_len], float32.
    """
    for name, t in [
        ("action_probs_behavior", action_probs_behavior),
        ("q_values",              q_values),
        ("rewards",               rewards),
        ("dones",                 dones),
    ]:
        assert t.is_cuda,                 f"{name} must be on CUDA"
        assert t.dtype == torch.float32,  f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape,  f"{name} shape {t.shape} != rewards shape {rewards.shape}"

    assert action_probs_target.is_cuda,                "action_probs_target must be on CUDA"
    assert action_probs_target.dtype == torch.float32, "action_probs_target: expected float32"
    assert action_probs_target.shape[:2] == rewards.shape, (
        f"action_probs_target shape {action_probs_target.shape} incompatible with rewards {rewards.shape}"
    )
    assert next_q_values_all.is_cuda,                "next_q_values_all must be on CUDA"
    assert next_q_values_all.dtype == torch.float32, "next_q_values_all: expected float32"
    assert next_q_values_all.shape == action_probs_target.shape, (
        f"next_q_values_all {next_q_values_all.shape} != action_probs_target {action_probs_target.shape}"
    )
    assert actions.is_cuda,              "actions must be on CUDA"
    assert actions.dtype == torch.int64, f"actions: expected int64, got {actions.dtype}"
    assert actions.shape == rewards.shape, \
        f"actions shape {actions.shape} != rewards shape {rewards.shape}"

    action_probs_target   = action_probs_target.contiguous()
    action_probs_behavior = action_probs_behavior.contiguous()
    q_values              = q_values.contiguous()
    next_q_values_all     = next_q_values_all.contiguous()
    actions               = actions.contiguous()
    rewards               = rewards.contiguous()
    dones                 = dones.contiguous()

    # u[t] = r[t] + γ * E_π[Q(s_{t+1},a)] * (1-d[t]) - Q(s_t,a_t)
    expected_next_q = (action_probs_target * next_q_values_all).sum(dim=-1)
    u = rewards + gamma * expected_next_q * (1.0 - dones) - q_values

    # v[t] = γ * c[t+1] * (1-d[t]),  c[t] = λ * min(c_bar, π(a_t) / μ(a_t))
    pi_a   = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c      = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)
    c_next = torch.empty_like(c)
    c_next[:, :-1] = c[:, 1:]
    c_next[:, -1]  = 0.0
    v = gamma * c_next * (1.0 - dones)

    return _run_scan(u, v, bootstrap_values) + q_values
