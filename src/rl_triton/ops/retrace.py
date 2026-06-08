import torch

from rl_triton.ops._scan import _correctness_warn, _run_scan, _FLAT_MAX_SEQ_LEN
from rl_triton.ops.retrace_fused import compute_retrace_fused


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
    truncateds: torch.Tensor | None = None,
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

    Bootstrap and terminal vs. truncated boundaries
    ------------------------------------------------
    Unlike GAE and V-Trace, Retrace never requires a separately supplied
    bootstrap value.  The one-step Q-bootstrap

        γ · E_π[Q(s_{t+1}, a)]

    is already folded into every TD error δ[t] via `next_q_values_all`.  All
    the caller needs to supply is the right boundary mask.

    Two distinct reasons a trajectory can end mid-sequence must be handled
    differently:

      terminated[t] = 1  — the environment signalled a true episode end.
                           s_{t+1} is a reset state with no meaningful value.
                           The bootstrap γ·E_π[Q(s_{t+1},·)] must be zeroed.

      truncated[t]  = 1  — the sequence window ended but the episode continues
                           (time-limit cutoff or replay-buffer window boundary).
                           s_{t+1} is a real state; the bootstrap must be kept.

    The `dones` flag (= terminated OR truncated) is used only to stop trace
    propagation across any boundary (v[t]=0 when done[t]=1).  The one-step TD
    bootstrap inside u[t] is gated by `terminated` alone.

    Gymnasium users: pass `terminated` as `dones` and `truncated` as
    `truncateds`.  Older single-done-flag APIs: pass the done flag as `dones`
    and leave `truncateds=None`; the bootstrap is then zeroed on every boundary
    (conservative, correct for purely episodic data).

    Set RL_TRITON_CORRECTNESS_WARNINGS=1 to enable a runtime warning when
    `truncateds` contains 1s that are not covered by `dones`, which is always
    a caller error.

    Recurrence:
      Δ[t] = δ[t] + decay[t] * Δ[t+1],   Δ[T] = 0
      δ[t]     = r[t] + γ · E_π[Q(s_{t+1},a)] · (1-terminated[t]) - Q(s_t,a_t)
      decay[t] = γ · c[t+1] · (1-done[t])
      c[t]     = λ · min(c_bar, π(a_t|s_t) / μ(a_t|s_t))

    Note: decay[T-1] is always zero because c[T] is out-of-bounds (no action
    was taken beyond the trajectory end).  This stops trace continuation but
    does not affect the one-step bootstrap already folded into δ[T-1].

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
        dones:                 Episode boundary flags (1.0 = terminated OR truncated).
                               Gates trace decay: v[t]=0 when done[t]=1.
                               [num_envs, seq_len], float32, CUDA.
        gamma:                 Discount factor.
        lambda_:               Trace decay parameter (default 1.0).
        c_bar:                 IS ratio clip for trace weights (default 1.0).
        truncateds:            Optional time-limit truncation flags (1.0 = truncated,
                               not truly terminal).  When provided, the one-step
                               bootstrap γ·E_π[Q(s_{t+1},·)] is kept for truncated
                               steps and zeroed only for true terminations.
                               Shape [num_envs, seq_len], float32, CUDA.
                               If None, dones is treated as pure termination.

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

    if truncateds is not None:
        assert truncateds.is_cuda,                "truncateds must be on CUDA"
        assert truncateds.dtype == torch.float32, "truncateds: expected float32"
        assert truncateds.shape == rewards.shape, \
            f"truncateds shape {truncateds.shape} != rewards shape {rewards.shape}"
        if (truncateds > dones).any():
            _correctness_warn(
                "truncateds has entries where truncateds=1 but dones=0. "
                "A step can only be truncated if it is also marked done; "
                "the trace decay will not be stopped at those steps."
            )

    action_probs_target   = action_probs_target.contiguous()
    action_probs_behavior = action_probs_behavior.contiguous()
    q_values              = q_values.contiguous()
    next_q_values_all     = next_q_values_all.contiguous()
    actions               = actions.contiguous()
    rewards               = rewards.contiguous()
    dones                 = dones.contiguous()

    # terminated[t]=1 only for true episode ends; bootstrap is zeroed there.
    # For truncations the episode continues so the bootstrap is kept.
    # When truncateds is not supplied dones is treated as pure termination.
    if truncateds is not None:
        terminated = (dones - truncateds.contiguous()).clamp(min=0.0)
    else:
        terminated = dones

    num_envs, seq_len = rewards.shape

    if seq_len <= _FLAT_MAX_SEQ_LEN:
        return compute_retrace_fused(
            action_probs_target, action_probs_behavior,
            q_values, next_q_values_all, actions,
            rewards, dones, terminated,
            gamma=gamma, lambda_=lambda_, c_bar=c_bar,
        )

    # Chunked fallback for seq_len > 131072: build u and v in PyTorch, scan via _run_scan.
    expected_next_q = (action_probs_target * next_q_values_all).sum(dim=-1)
    u = rewards + gamma * expected_next_q * (1.0 - terminated) - q_values

    pi_a   = action_probs_target.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    c      = lambda_ * torch.clamp(pi_a / action_probs_behavior, max=c_bar)
    c_next = torch.empty_like(c)
    c_next[:, :-1] = c[:, 1:]
    c_next[:, -1]  = 0.0
    v = gamma * c_next * (1.0 - dones)

    return _run_scan(u, v) + q_values
