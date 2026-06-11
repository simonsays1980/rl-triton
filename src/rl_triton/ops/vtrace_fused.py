import torch
import triton

from rl_triton.kernels.vtrace_fused import vtrace_fused_kernel

_FLAT_MAX_SEQ_LEN = 131072
_WARPS = {512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32}


def compute_vtrace_fused(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fully-fused V-Trace targets and advantages via a single Triton kernel.

    V(s_{t+1}) is read directly from values[:, t+1] inside the kernel — the
    caller does not need to pass a separate next_values tensor.  For the last
    timestep, V(s_T) is taken from bootstrap_values (same as the scan boundary).

    Args:
        log_pi_target:    Log probs under target policy, [num_envs, seq_len], float32, CUDA.
        log_pi_behavior:  Log probs under behavior policy, same shape.
        values:           V(s_t), same shape.  V(s_{t+1}) is read as values[:, t+1].
        rewards:          Per-step rewards, same shape.
        dones:            Episode termination flags (1.0=done), same shape, float32.
        gamma:            Discount factor.
        rho_bar:          IS ratio clip for delta (default 1.0).
        c_bar:            IS ratio clip for decay (default 1.0).
        bootstrap_values: Per-env boundary value Delta[T] / V(s_T), shape [num_envs].
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Defaults to zeros.

    Returns:
        vtrace_targets:    [num_envs, seq_len], float32.
        vtrace_advantages: [num_envs, seq_len], float32.
    """
    for name, t in [
        ("log_pi_target",   log_pi_target),
        ("log_pi_behavior", log_pi_behavior),
        ("values",          values),
        ("rewards",         rewards),
        ("dones",           dones),
    ]:
        assert t.is_cuda,                f"{name} must be on CUDA"
        assert t.dtype == torch.float32, f"{name}: expected float32, got {t.dtype}"
        assert t.shape == rewards.shape, f"{name} shape {t.shape} != rewards shape {rewards.shape}"

    num_envs, seq_len = rewards.shape
    assert seq_len <= _FLAT_MAX_SEQ_LEN, (
        f"seq_len={seq_len} exceeds the flat kernel limit {_FLAT_MAX_SEQ_LEN}. "
        "Use compute_vtrace for longer sequences (it auto-dispatches to chunked)."
    )

    log_pi_target   = log_pi_target.contiguous()
    log_pi_behavior = log_pi_behavior.contiguous()
    values          = values.contiguous()
    rewards         = rewards.contiguous()
    dones           = dones.contiguous()

    if bootstrap_values is None:
        bootstrap_values = torch.zeros(num_envs, device=rewards.device, dtype=torch.float32)
    else:
        assert bootstrap_values.shape == (num_envs,), \
            f"bootstrap_values must have shape [{num_envs}], got {bootstrap_values.shape}"
        assert bootstrap_values.is_cuda, "bootstrap_values must be on CUDA"
        bootstrap_values = bootstrap_values.contiguous()

    vtrace_targets    = torch.empty_like(rewards)
    vtrace_advantages = torch.empty_like(rewards)

    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    num_warps  = _WARPS.get(BLOCK_SIZE, 16)
    num_stages = 2 if BLOCK_SIZE >= 2048 else 1

    vtrace_fused_kernel[(num_envs,)](
        log_pi_target, log_pi_behavior,
        values,
        rewards, dones,
        vtrace_targets, vtrace_advantages,
        bootstrap_values,
        seq_len,
        rewards.stride(0),
        gamma=gamma,
        rho_bar=rho_bar,
        c_bar=c_bar,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return vtrace_targets, vtrace_advantages
