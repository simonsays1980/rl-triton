import torch

from rl_triton.ops._scan import _run_scan, _FLAT_MAX_SEQ_LEN, _CORRECTNESS_WARNINGS
from rl_triton.ops.retrace_fused import compute_retrace_fused


def compute_retrace(
    action_probs_target: torch.Tensor,
    action_probs_behavior: torch.Tensor,
    q_values: torch.Tensor,
    next_q_values_all: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor,
    gamma: float,
    lambda_: float = 1.0,
    c_bar: float = 1.0,
    rho_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Retrace(λ) Q-value targets and advantages via a backward associative scan.

    Retrace(λ) (Munos et al. 2016) corrects off-policy Q-value estimates using
    truncated IS traces applied only to the decay factor, not the TD error.
    Discrete actions only — use compute_vtrace for continuous action spaces.

    Recurrence:

    - Δ[t] = δ[t] + decay[t] * Δ[t+1],   Δ[T] = 0
    - δ[t]     = r[t] + γ · E_π[Q(s_{t+1},a)] · (1-terminated[t]) - Q(s_t,a_t)
    - decay[t] = γ · c[t+1] · (1-done[t]),  done[t] = terminated[t] | truncated[t]
    - c[t]     = λ · min(c_bar, π(a_t|s_t) / μ(a_t|s_t))

    Q-value targets: Q_ret[t] = Q(s_t, a_t) + Δ[t]
    Advantages:      A[t]     = ρ[t] · (r[t] + γ · Q_ret[t+1] · (1-terminated[t]) - Q(s_t,a_t))
                     ρ[t]     = min(rho_bar, π(a_t|s_t) / μ(a_t|s_t))

    The Q-bootstrap γ·E_π[Q(s_{t+1},·)] is folded into δ[t] via
    `next_q_values_all`, so no separate bootstrap_values argument is needed.
    `terminated` gates the one-step bootstrap in δ[t]; `terminated | truncated`
    gates trace decay in β[t].

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
        terminateds:           True termination flags (1.0 = terminated).
                               Zeros the bootstrap γ·E_π[Q(s_{t+1},·)] in δ[t].
                               [num_envs, seq_len], float32, CUDA.
        truncateds:            Time-limit truncation flags (1.0 = truncated).
                               Keeps the bootstrap in δ[t] but severs the trace.
                               [num_envs, seq_len], float32, CUDA.
        gamma:                 Discount factor.
        lambda_:               Trace decay parameter (default 1.0).
        c_bar:                 IS ratio clip for trace weights (default 1.0).
        rho_bar:               IS ratio clip for advantage scaling (default 1.0).

    Returns:
        retrace_targets: Q_ret[t], shape [num_envs, seq_len], float32.
        advantages:      A[t],     shape [num_envs, seq_len], float32.
    """
    # Cheap structural checks — always-on.
    for name, t in [
        ("action_probs_behavior", action_probs_behavior),
        ("q_values",              q_values),
        ("rewards",               rewards),
        ("terminateds",           terminateds),
        ("truncateds",            truncateds),
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

    # Expensive tensor scan — correctness-warning path only.
    if _CORRECTNESS_WARNINGS():
        assert not (terminateds.bool() & truncateds.bool()).any(), \
            "terminated and truncated are mutually exclusive: a step cannot be both"

    action_probs_target   = action_probs_target.contiguous()
    action_probs_behavior = action_probs_behavior.contiguous()
    q_values              = q_values.contiguous()
    next_q_values_all     = next_q_values_all.contiguous()
    actions               = actions.contiguous()
    rewards               = rewards.contiguous()
    terminateds           = terminateds.contiguous()
    truncateds            = truncateds.contiguous()

    num_envs, seq_len = rewards.shape

    # Fused kernel for seq_len <= 131072; chunked fallback for longer sequences.
    # done[t] = terminated[t] | truncated[t] is computed in-kernel from the two
    # raw flags for the fused path — no separate PyTorch combine here.
    if seq_len <= _FLAT_MAX_SEQ_LEN:
        return compute_retrace_fused(
            action_probs_target, action_probs_behavior,
            q_values, next_q_values_all, actions,
            rewards, truncateds, terminateds,
            gamma=gamma, lambda_=lambda_, c_bar=c_bar, rho_bar=rho_bar,
        )

    # Chunked fallback for seq_len > 131072. _run_scan takes precomputed u/v,
    # so dones is materialized here (this path's PyTorch-op overhead is
    # negligible relative to the chunked kernel's cost at long seq_len).
    dones = (terminateds + truncateds).clamp(max=1.0)
    expected_next_q = (action_probs_target * next_q_values_all).sum(dim=-1)
    u = rewards + gamma * expected_next_q * (1.0 - terminateds) - q_values

    pi_a   = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c      = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)
    c_next = torch.empty_like(c)
    c_next[:, :-1] = c[:, 1:]
    c_next[:, -1]  = 0.0
    v = gamma * c_next * (1.0 - dones)

    retrace_targets = _run_scan(u, v) + q_values

    rho = torch.clamp(pi_a / action_probs_behavior, max=rho_bar)
    next_q_ret = torch.empty_like(retrace_targets)
    next_q_ret[:, :-1] = retrace_targets[:, 1:]
    next_q_ret[:, -1]  = 0.0
    advantages = rho * (rewards + gamma * next_q_ret * (1.0 - terminateds) - q_values)

    return retrace_targets, advantages
