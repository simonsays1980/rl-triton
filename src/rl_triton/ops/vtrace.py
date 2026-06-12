import torch

from rl_triton.ops._scan import _run_scan, _FLAT_MAX_SEQ_LEN
from rl_triton.ops.vtrace_fused import compute_vtrace_fused


def compute_vtrace(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute V-Trace targets and advantages via a backward associative scan.

    V-Trace (Espeholt et al. 2018, IMPALA) corrects for off-policy data using
    clipped importance sampling ratios ρ and c.

    Recurrence:

    - Δ[t] = δ[t] + decay[t] * Δ[t+1],   Δ[T] = bootstrap
    - δ[t]     = ρ[t] * (r[t] + γ * V(s_{t+1}) * (1 - terminated[t]) - V(s_t))
    - decay[t] = γ * c[t] * (1 - terminated[t])
    - ρ[t]     = min(ρ_bar, π_target[t] / π_behavior[t])
    - c[t]     = min(c_bar, π_target[t] / π_behavior[t])

    Outputs:

    - vs[t] = Δ[t] + V(s_t)   (critic targets)
    - A[t]  = ρ[t] * (r[t] + γ * vs[t+1] * (1 - terminated[t]) - V(s_t))   (actor advantages)

    V(s_{t+1}) is read from values[:, t+1]; no separate next_values tensor needed.

    Pass only true terminations in `terminateds`.  For a terminated step,
    terminated[t]=1 zeros the carry so no value bleeds across the episode
    boundary.  For a truncated step, set terminated[t]=0 and supply V(s_T)
    via `bootstrap_values`.

    Args:
        log_pi_target:    Log probabilities under target policy, [num_envs, seq_len], float32, CUDA.
        log_pi_behavior:  Log probabilities under behavior policy, [num_envs, seq_len], float32, CUDA.
        values:           V(s_t), [num_envs, seq_len], float32, CUDA.
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        terminateds:      True termination flags only (1.0=terminated),
                          [num_envs, seq_len], float32, CUDA.
                          Do not pass (terminated | truncated) here — truncated steps
                          must have terminated[t]=0 with V(s_T) in bootstrap_values.
        gamma:            Discount factor.
        rho_bar:          IS ratio clip for δ (default 1.0).
        c_bar:            IS ratio clip for decay (default 1.0).
        bootstrap_values: V(s_T) per environment, shape [num_envs].
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Defaults to zeros.

    Returns:
        vtrace_targets:    vs[t], shape [num_envs, seq_len], float32.
        vtrace_advantages: A[t],  shape [num_envs, seq_len], float32.
    """
    for name, t in [
        ("log_pi_target",   log_pi_target),
        ("log_pi_behavior", log_pi_behavior),
        ("values",          values),
        ("rewards",         rewards),
        ("terminateds",     terminateds),
    ]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"

    num_envs, seq_len = rewards.shape

    log_pi_target   = log_pi_target.contiguous()
    log_pi_behavior = log_pi_behavior.contiguous()
    values          = values.contiguous()
    rewards         = rewards.contiguous()
    terminateds     = terminateds.contiguous()

    if bootstrap_values is None:
        bootstrap_values = torch.zeros(num_envs, device=rewards.device, dtype=torch.float32)
    else:
        assert bootstrap_values.shape == (num_envs,), \
            f"bootstrap_values must have shape [{num_envs}], got {bootstrap_values.shape}"
        assert bootstrap_values.is_cuda, "bootstrap_values must be on CUDA"
        bootstrap_values = bootstrap_values.contiguous()

    # Linear recurrence: u[t] = rho[t]*delta[t], v[t] = gamma*c[t]*(1-terminated[t]).
    # Fused kernel for seq_len <= 131072 (IS ratios, scan, targets, advantages
    # in one GPU kernel). Chunked scan + PyTorch elementwise ops for longer.
    if seq_len <= _FLAT_MAX_SEQ_LEN:
        return compute_vtrace_fused(
            log_pi_target, log_pi_behavior,
            values, rewards, terminateds,
            gamma=gamma, rho_bar=rho_bar, c_bar=c_bar,
            bootstrap_values=bootstrap_values,
        )

    # Chunked path for seq_len > 131072.
    # Build next_values from values shifted by 1, bootstrap at boundary.
    next_values = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = bootstrap_values

    is_ratios      = torch.exp(log_pi_target - log_pi_behavior)
    rho            = torch.clamp(is_ratios, max=rho_bar)
    c              = torch.clamp(is_ratios, max=c_bar)
    not_terminated = 1.0 - terminateds
    u = rho * (rewards + gamma * next_values * not_terminated - values)
    v = gamma * c * not_terminated

    value_deltas   = _run_scan(u, v, bootstrap_values)
    vtrace_targets = value_deltas + values

    next_vtrace_targets = torch.empty_like(vtrace_targets)
    next_vtrace_targets[:, :-1] = vtrace_targets[:, 1:]
    next_vtrace_targets[:, -1]  = bootstrap_values

    vtrace_advantages = rho * (rewards + gamma * next_vtrace_targets * not_terminated - values)
    return vtrace_targets, vtrace_advantages
