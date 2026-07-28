import torch
import triton

from rl_triton.kernels.vtrace_fused import vtrace_fused_kernel
from rl_triton.ops._scan import _FLAT_MAX_SEQ_LEN, _CORRECTNESS_WARNINGS

# Below 512, BLOCK_SIZE used to fall through .get()'s default (16 warps),
# grossly over-provisioned for small single-block reductions — see
# src/rl_triton/ops/gae.py's _WARPS for the H200 measurement basis (device
# time flat for num_warps in {1,2,4} at BLOCK_SIZE 8-128, 2-3x worse at the
# old default). Spot-checked directly on this kernel (more registers than
# GAE's) before applying: same flat-then-degrade shape, bit-identical output
# at every num_warps tested.
_WARPS = {
    8: 2, 16: 2, 32: 2, 64: 2, 128: 2, 256: 4,
    512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32,
}


def compute_vtrace_fused(
    log_pi_target: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor | None = None,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    bootstrap_values: torch.Tensor | None = None,
    last_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fully-fused V-Trace targets and advantages via a single Triton kernel.

    Dispatches to HAS_TRUNCATIONS=False when truncateds is None and no interior
    bootstraps are needed.  That path reads 5 full-width tensors (vs 7 when
    truncateds are present) and passes a scalar [num_envs] bootstrap rather
    than a 2D [num_envs, seq_len] tensor.

    V(s_{t+1}) is read directly from values[:, t+1] at interior non-truncated steps.
    At truncated steps and the window boundary, bootstrap_values[env, t] provides
    the true continuation value V(s_{t+1}^true).

    bootstrap_values supplies the true continuation value V(s_{t+1}) wherever
    the stored values[t+1] is invalid.  Two situations require this, under one
    rule: (a) truncated steps, where values[t+1] belongs to the next episode;
    and (b) the final column t=T-1, where values[t+1] lies past the buffer.
    In both cases set bootstrap_values to the true continuation value; leave
    it zero everywhere else.

    The final column has a dual role: it feeds delta[T-1] as the next-state
    value AND seeds the scan carry Δ_T.  Both uses take the same number, so a
    single entry in bootstrap_values[:, -1] is sufficient.  Set it to V(s_T)
    if the episode continues past the window, or 0 if it terminated at T-1.

    Most users have no interior truncations and should use the `last_value`
    argument (shape [num_envs]) instead, which populates the boundary column
    automatically.

    `terminated[t]` gates the one-step bootstrap inside δ[t].
    `truncated[t]` stops trace propagation and injects bootstrap_values[t].

    Args:
        log_pi_target:    Log probs under target policy, [num_envs, seq_len], float32, CUDA.
        log_pi_behavior:  Log probs under behavior policy, same shape.
        values:           V(s_t), same shape.
        rewards:          Per-step rewards, same shape.
        terminateds:      True termination flags (1.0=terminated), same shape, float32.
        truncateds:       Time-limit truncation flags (1.0=truncated), same shape.
                          If None, no interior truncations — uses HAS_TRUNCATIONS=False fast path.
        gamma:            Discount factor (default 0.99).
        rho_bar:          IS ratio clip for delta (default 1.0).
        c_bar:            IS ratio clip for decay (default 1.0).
        bootstrap_values: True continuation values V(s_{t+1}^true),
                          [num_envs, seq_len], float32, CUDA.
                          Set bootstrap_values[env, t] = V(s_{t+1}^true) at every
                          truncated step and at t=T-1 if the window ends mid-episode.
                          Zero elsewhere.  If None, defaults to zeros.
                          Mutually exclusive with last_value.
        last_value:       Convenience arg for the common case of no interior
                          truncations: V(s_T) per environment, shape [num_envs],
                          float32, CUDA.  Populates bootstrap_values[:, -1]
                          automatically.  Mutually exclusive with bootstrap_values.

    Returns:
        vtrace_targets:    [num_envs, seq_len], float32.
        vtrace_advantages: [num_envs, seq_len], float32.
    """
    num_envs, seq_len = rewards.shape
    has_truncations   = truncateds is not None

    # Cheap structural checks — always-on: catch shape/dtype/device bugs at call time.
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
    if has_truncations:
        assert truncateds.is_cuda,                "truncateds must be on CUDA"
        assert truncateds.dtype == torch.float32, "truncateds: expected float32"
        assert truncateds.shape == rewards.shape, \
            f"truncateds shape {truncateds.shape} != rewards shape {rewards.shape}"
    if last_value is not None:
        assert bootstrap_values is None, \
            "pass either last_value (shape [num_envs], convenience for the " \
            "window boundary) or bootstrap_values (shape [num_envs, seq_len], " \
            "full per-step control), not both."
        assert last_value.shape == (num_envs,), \
            f"last_value must have shape [{num_envs}], got {last_value.shape}"
        assert last_value.is_cuda,                "last_value must be on CUDA"
        assert last_value.dtype == torch.float32, "last_value: expected float32"
    if bootstrap_values is not None:
        assert bootstrap_values.is_cuda,                "bootstrap_values must be on CUDA"
        assert bootstrap_values.dtype == torch.float32, "bootstrap_values: expected float32"
        assert bootstrap_values.shape == rewards.shape, \
            f"bootstrap_values shape {bootstrap_values.shape} != rewards shape {rewards.shape}"

    # Expensive tensor scans — correctness-warning path only (not in benchmark hot loop).
    if _CORRECTNESS_WARNINGS():
        if has_truncations:
            assert not (terminateds.bool() & truncateds.bool()).any(), \
                "terminated and truncated are mutually exclusive: a step cannot be both"
        if has_truncations and bootstrap_values is not None:
            interior = torch.ones_like(truncateds, dtype=torch.bool)
            interior[:, -1] = False
            stray = (bootstrap_values != 0) & (truncateds == 0) & interior
            assert not stray.any(), (
                "bootstrap_values must be zero at non-truncated interior steps. "
                "Nonzero entries there double-count via the additive v_next trick "
                "and corrupt delta. Populate bootstrap_values only at truncated "
                "steps and at the final column (window boundary)."
            )

    assert seq_len <= _FLAT_MAX_SEQ_LEN, (
        f"seq_len={seq_len} exceeds the flat kernel limit {_FLAT_MAX_SEQ_LEN}. "
        "Use compute_vtrace for longer sequences (it auto-dispatches to chunked)."
    )

    log_pi_target   = log_pi_target.contiguous()
    log_pi_behavior = log_pi_behavior.contiguous()
    values          = values.contiguous()
    rewards         = rewards.contiguous()
    terminateds     = terminateds.contiguous()

    vtrace_targets    = torch.empty_like(rewards)
    vtrace_advantages = torch.empty_like(rewards)

    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    num_warps  = _WARPS.get(BLOCK_SIZE, 16)
    num_stages = 2 if BLOCK_SIZE >= 2048 else 1

    if has_truncations:
        truncateds = truncateds.contiguous()
        if last_value is not None:
            bv = torch.zeros_like(rewards)
            bv[:, -1] = last_value
            bootstrap_values = bv
        elif bootstrap_values is not None:
            bootstrap_values = bootstrap_values.contiguous()
        else:
            bootstrap_values = torch.zeros_like(rewards)

        vtrace_fused_kernel[(num_envs,)](
            log_pi_target, log_pi_behavior,
            values,
            rewards, terminateds, truncateds,
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
            HAS_TRUNCATIONS=True,
            HAS_BOOTSTRAP=True,
        )
    else:
        # Fast path: no truncateds tensor, scalar bootstrap per env.
        # No zero-tensor allocations — truncateds_ptr is constexpr None in the kernel,
        # and when there's no last_value/bootstrap_values either, bootstrap_ptr is
        # also skipped (HAS_BOOTSTRAP=False) instead of materializing zeros.
        if last_value is not None:
            scalar_bootstrap = last_value.contiguous()
        elif bootstrap_values is not None:
            scalar_bootstrap = bootstrap_values[:, -1].contiguous()
        else:
            scalar_bootstrap = None
        has_bootstrap = scalar_bootstrap is not None

        vtrace_fused_kernel[(num_envs,)](
            log_pi_target, log_pi_behavior,
            values,
            rewards, terminateds, None,
            vtrace_targets, vtrace_advantages,
            scalar_bootstrap,
            seq_len,
            rewards.stride(0),
            gamma=gamma,
            rho_bar=rho_bar,
            c_bar=c_bar,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
            HAS_TRUNCATIONS=False,
            HAS_BOOTSTRAP=has_bootstrap,
        )

    return vtrace_targets, vtrace_advantages
