import torch

from rl_triton.ops._scan import _run_scan, _FLAT_MAX_SEQ_LEN, _CORRECTNESS_WARNINGS
from rl_triton.ops.vtrace_fused import compute_vtrace_fused


def compute_vtrace(
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
    Compute V-Trace targets and advantages via a backward associative scan.

    V-Trace (Espeholt et al. 2018, IMPALA) corrects for off-policy data using
    clipped importance sampling ratios ρ and c.

    Recurrence:

    - Δ[t] = δ[t] + β[t] * Δ[t+1],   Δ[T] = 0
    - δ[t] = ρ[t] * (r[t] + γ * V(s_{t+1}) * (1 - terminated[t]) - V(s_t))
    - β[t] = γ * c[t] * (1 - done[t]),  done[t] = terminated[t] | truncated[t]
             (scan decay coefficient)
    - ρ[t] = min(ρ_bar, π_target[t] / π_behavior[t])
    - c[t] = min(c_bar, π_target[t] / π_behavior[t])

    Outputs:

    - vs[t] = Δ[t] + V(s_t)   (critic targets)
    - A[t]  = ρ[t] * (r[t] + γ * vs[t+1] * (1 - terminated[t]) - V(s_t))   (actor advantages)

    bootstrap_values supplies the true continuation value V(s_{t+1}) wherever
    the stored values[t+1] is invalid.  Two situations require this, under one
    rule: (a) truncated steps, where values[t+1] belongs to the next episode;
    and (b) the final column t=T-1, where values[t+1] lies past the buffer.
    In both cases set bootstrap_values to the true continuation value; leave
    it zero everywhere else.

    The final column feeds delta[T-1] as the next-state value ONLY.  The
    scan's additive boundary carry Δ[T] is always 0: an advantage carry
    represents trace mass from steps past the buffer, and there is none
    there.  (β[T-1], the multiplicative decay coefficient above, is
    unaffected by this -- only the additive Δ[T] term changes.)  A single
    entry in bootstrap_values[:, -1] is sufficient -- set it to V(s_T) if the
    episode continues past the window, or 0 if it terminated at T-1.  (The
    same bootstrap_values[:, -1] is also used directly, not as a carry, when
    computing next_vtrace_targets[:, -1] for the advantage formula -- that use
    is single-counted and correct as-is.)

    Most users have no interior truncations and should use the `last_value`
    argument (shape [num_envs]) instead, which populates the boundary column
    automatically.

    `terminated[t]` gates the one-step bootstrap in δ[t]: set to 1 at true episode
    ends.  `truncated[t]` stops trace propagation and injects bootstrap_values[env, t]
    = V(s_{t+1}^true), correcting for stale reset observations in the values buffer
    under Gymnasium next-step autoreset.

    Args:
        log_pi_target:    Log probabilities under target policy, [num_envs, seq_len], float32, CUDA.
        log_pi_behavior:  Log probabilities under behavior policy, [num_envs, seq_len], float32, CUDA.
        values:           V(s_t), [num_envs, seq_len], float32, CUDA.
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        terminateds:      True termination flags (1.0=terminated),
                          [num_envs, seq_len], float32, CUDA.
        truncateds:       Time-limit truncation flags (1.0=truncated),
                          [num_envs, seq_len], float32, CUDA.
                          If None, terminateds is used for both gating roles.
        gamma:            Discount factor (default 0.99).
        rho_bar:          IS ratio clip for δ (default 1.0).
        c_bar:            IS ratio clip for decay (default 1.0).
        bootstrap_values: True continuation values V(s_{t+1}^true),
                          [num_envs, seq_len], float32, CUDA.
                          Set bootstrap_values[env, t] = V(s_{t+1}^true) at every
                          truncated step and at t=T-1 if the window ends mid-episode.
                          Zero elsewhere.  If None, defaults to all zeros.
                          Mutually exclusive with last_value.
        last_value:       Convenience arg for the common case of no interior
                          truncations: V(s_T) per environment, shape [num_envs],
                          float32, CUDA.  Populates bootstrap_values[:, -1]
                          automatically.  Mutually exclusive with bootstrap_values.

    Returns:
        vtrace_targets:    vs[t], shape [num_envs, seq_len], float32.
        vtrace_advantages: A[t],  shape [num_envs, seq_len], float32.
    """
    num_envs, seq_len = rewards.shape
    has_truncations   = truncateds is not None

    if _CORRECTNESS_WARNINGS():
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
            assert not (terminateds.bool() & truncateds.bool()).any(), \
                "terminated and truncated are mutually exclusive: a step cannot be both"
        if last_value is not None:
            assert bootstrap_values is None, \
                "pass either last_value (shape [num_envs], convenience for the " \
                "window boundary) or bootstrap_values (shape [num_envs, seq_len], " \
                "full per-step control), not both."
            assert last_value.shape == (num_envs,), \
                f"last_value must have shape [{num_envs}], got {last_value.shape}"
        if bootstrap_values is not None:
            assert bootstrap_values.is_cuda,                "bootstrap_values must be on CUDA"
            assert bootstrap_values.dtype == torch.float32, "bootstrap_values: expected float32"
            assert bootstrap_values.shape == rewards.shape, \
                f"bootstrap_values shape {bootstrap_values.shape} != rewards shape {rewards.shape}"
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

    log_pi_target   = log_pi_target.contiguous()
    log_pi_behavior = log_pi_behavior.contiguous()
    values          = values.contiguous()
    rewards         = rewards.contiguous()
    terminateds     = terminateds.contiguous()
    if has_truncations:
        truncateds = truncateds.contiguous()

    # Fused kernel for seq_len <= 131072.
    # Pass truncateds=None for the no-truncation path so compute_vtrace_fused
    # dispatches HAS_TRUNCATIONS=False -- no zero-tensor allocations for truncateds
    # or 2D bootstrap_values in that path.
    if seq_len <= _FLAT_MAX_SEQ_LEN:
        return compute_vtrace_fused(
            log_pi_target, log_pi_behavior,
            values, rewards, terminateds,
            truncateds=truncateds if has_truncations else None,
            gamma=gamma, rho_bar=rho_bar, c_bar=c_bar,
            bootstrap_values=bootstrap_values,
            last_value=last_value,
        )

    # Chunked path for seq_len > 131072.  Materialize None inputs here only.
    if not has_truncations:
        truncateds = torch.zeros_like(terminateds)
    if last_value is not None:
        bootstrap_values = torch.zeros_like(rewards)
        bootstrap_values[:, -1] = last_value
    elif bootstrap_values is None:
        bootstrap_values = torch.zeros_like(rewards)

    not_terminated = 1.0 - terminateds
    not_done       = 1.0 - (terminateds + truncateds).clamp(max=1.0)
    next_values    = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = 0.0
    next_values = next_values * (1.0 - truncateds)
    next_values[:, -1]  = 0.0
    next_values = next_values + bootstrap_values

    is_ratios      = torch.exp(log_pi_target - log_pi_behavior)
    rho            = torch.clamp(is_ratios, max=rho_bar)
    c              = torch.clamp(is_ratios, max=c_bar)
    u = rho * (rewards + gamma * next_values * not_terminated - values)
    v = gamma * c * not_done   # beta[t], the scan decay coefficient

    # Additive boundary carry Delta[T] = 0: bootstrap_values[:, -1] already
    # entered u[:, -1] via next_values above (weight 1). Seeding the scan's
    # boundary carry with it too would double-count it -- see kernels/gae.py's
    # module docstring for the same bug class in GAE. beta (v) itself is
    # unaffected by this fix.
    value_deltas   = _run_scan(u, v)
    vtrace_targets = value_deltas + values

    next_vtrace_targets = torch.empty_like(vtrace_targets)
    next_vtrace_targets[:, :-1] = vtrace_targets[:, 1:]
    next_vtrace_targets[:, -1]  = 0.0
    next_vtrace_targets = next_vtrace_targets * (1.0 - truncateds)
    next_vtrace_targets[:, -1]  = 0.0
    next_vtrace_targets = next_vtrace_targets + bootstrap_values

    vtrace_advantages = rho * (rewards + gamma * next_vtrace_targets * not_terminated - values)
    return vtrace_targets, vtrace_advantages
