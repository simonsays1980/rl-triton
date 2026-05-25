import torch
import triton

from rl_triton.kernels.vtrace import vtrace_scan_kernel
from rl_triton.ops.vtrace_chunked import compute_vtrace_chunked

# tl.associative_scan requires BLOCK_SIZE <= 2^17.  Above this the flat kernel
# cannot launch, so we fall back to the chunked kernel automatically.
_FLAT_MAX_SEQ_LEN = 131072


def compute_vtrace_triton(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute V-Trace targets and advantages via a backward associative scan.

    V-Trace (Espeholt et al. 2018, IMPALA) corrects for off-policy data using
    clipped importance sampling ratios ρ and c.

    Scan recurrence:
      Δ[t] = δ[t] + decay[t] * Δ[t+1],   Δ[T] = bootstrap
      δ[t]     = ρ[t] * (r[t] + γ * V(s_{t+1}) * (1 - done[t]) - V(s_t))
      decay[t] = γ * c[t] * (1 - done[t])
      ρ[t]     = min(ρ_bar, π_target[t] / π_behavior[t])
      c[t]     = min(c_bar, π_target[t] / π_behavior[t])

    V-Trace targets (for critic loss):
      v[t] = Δ[t] + V(s_t)

    V-Trace advantages (for actor loss):
      A[t] = ρ[t] * (r[t] + γ * v[t+1] * (1 - done[t]) - V(s_t))

    Dispatches to the flat single-block kernel for seq_len <= 131072, and falls
    back to the chunked kernel for longer sequences.

    Args:
        log_pi_target:    Log probabilities under target policy, [num_envs, seq_len], float32, CUDA.
        log_pi_behavior:  Log probabilities under behavior policy, same shape.
        values:           Value function predictions V(s_t), same shape.
        next_values:      Value function predictions V(s_{t+1}), same shape.
        rewards:          Per-step rewards, same shape.
        dones:            Episode termination flags (1.0 = done), same shape, float32.
        gamma:            Discount factor.
        rho_bar:          IS ratio clip for δ (default 1.0).
        c_bar:            IS ratio clip for decay (default 1.0).
        bootstrap_values: Per-environment boundary value Δ[T], shape [num_envs], float32.
                          Use γ * c[T] * V(s_T) for truncated episodes, 0 for terminated.
                          Defaults to zeros (all episodes terminated).

    Returns:
        vtrace_targets:    shape [num_envs, seq_len], float32.
        vtrace_advantages: shape [num_envs, seq_len], float32.
    """
    for name, t in [
        ("log_pi_target",   log_pi_target),
        ("log_pi_behavior", log_pi_behavior),
        ("values",          values),
        ("next_values",     next_values),
        ("rewards",         rewards),
        ("dones",           dones),
    ]:
        assert t.is_cuda,                         f"{name} must be on CUDA"
        assert t.dtype == torch.float32,          f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape,          f"{name} shape {t.shape} != rewards shape {rewards.shape}"

    num_envs, seq_len = rewards.shape

    log_pi_target   = log_pi_target.contiguous()
    log_pi_behavior = log_pi_behavior.contiguous()
    values          = values.contiguous()
    next_values     = next_values.contiguous()
    rewards         = rewards.contiguous()
    dones           = dones.contiguous()

    if bootstrap_values is None:
        bootstrap_values = torch.zeros(num_envs, device=rewards.device, dtype=torch.float32)
    else:
        assert bootstrap_values.shape == (num_envs,), \
            f"bootstrap_values must have shape [{num_envs}], got {bootstrap_values.shape}"
        assert bootstrap_values.is_cuda, "bootstrap_values must be on CUDA"
        bootstrap_values = bootstrap_values.contiguous()

    # Importance sampling ratios, computed in log-space for numerical stability.
    is_ratios = torch.exp(log_pi_target - log_pi_behavior)
    rho = torch.clamp(is_ratios, max=rho_bar)
    c   = torch.clamp(is_ratios, max=c_bar)

    # Scan inputs.
    deltas = rho * (rewards + gamma * next_values * (1.0 - dones) - values)
    decays = gamma * c * (1.0 - dones)

    # Run the backward scan.
    if seq_len > _FLAT_MAX_SEQ_LEN:
        value_deltas = compute_vtrace_chunked(deltas, decays, bootstrap_values)
    else:
        value_deltas = torch.empty_like(rewards)
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        vtrace_scan_kernel[(num_envs,)](
            deltas, decays, value_deltas,
            bootstrap_values,
            seq_len,
            rewards.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # V-Trace targets: v[t] = Δ[t] + V(s_t).
    vtrace_targets = value_deltas + values

    # V-Trace advantages: A[t] = ρ[t] * (r[t] + γ * v[t+1] * (1-done[t]) - V(s_t)).
    # v[t+1] is vtrace_targets shifted left by one; the last position uses next_values
    # as the bootstrap (correct for both terminated and truncated endings).
    next_vtrace_targets = torch.empty_like(vtrace_targets)
    next_vtrace_targets[:, :-1] = vtrace_targets[:, 1:]
    next_vtrace_targets[:, -1]  = next_values[:, -1]

    vtrace_advantages = rho * (rewards + gamma * next_vtrace_targets * (1.0 - dones) - values)

    return vtrace_targets, vtrace_advantages
