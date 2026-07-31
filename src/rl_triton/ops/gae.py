import torch
import triton

from rl_triton.kernels.gae import gae_kernel, gae_fused_kernel
from rl_triton.ops._scan import _run_scan, _FLAT_MAX_SEQ_LEN, _CORRECTNESS_WARNINGS

# Below 512, next_power_of_2(seq_len) used to fall through .get()'s default
# (16 warps = 512 threads) regardless of BLOCK_SIZE, e.g. scanning 8 elements
# with 512 threads resident (~98% masked-off lanes). Measured on H200 SXM
# (tests/benchmark_gae_vs_pufferlib.py massively-parallel-sim regime, Step 1
# of the warps-floor investigation): device time is FLAT for num_warps in
# {1, 2, 4} at BLOCK_SIZE 8-128 (they all hit the same 32-blocks/SM hard cap
# on Hopper) and degrades 2.1x-2.7x at the old default of 16 warps, worse at
# higher num_envs (fewer resident blocks per SM -> more grid waves to cover
# the same total env count). 128 specifically must NOT use 4 warps: it wins
# by ~2% at num_envs=4096 but REGRESSES ~11% at num_envs=32768 versus 2 warps
# -- 2 is the robust choice, tied with 1 at both scales. 256 is the opposite:
# 4 warps wins outright at large num_envs (44.6us vs 45.4-45.9us at 1-2).
_WARPS = {
    8: 2, 16: 2, 32: 2, 64: 2, 128: 2, 256: 4,
    512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32,
}


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    truncateds: torch.Tensor | None = None,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    bootstrap_values: torch.Tensor | None = None,
    last_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation via a backward associative scan.

    Recurrence:

    - A[t] = δ[t] + γ·λ·(1-done[t]) * A[t+1]
    - δ[t] = r[t] + γ·(1-terminated[t]) * v_next[t] - V(s_t)
    - done[t] = terminated[t] | truncated[t]

    `terminated[t]` gates the one-step bootstrap inside δ[t]: set to 1 for true
    episode ends (s_{t+1} is a reset state with no meaningful value).
    `truncated[t]` stops trace propagation and injects the true continuation
    value bootstrap_values[env, t] = V(s_{t+1}^true) into δ[t].

    bootstrap_values supplies the true continuation value V(s_{t+1}) wherever
    the stored values[t+1] is invalid.  Two situations require this, under one
    rule: (a) truncated steps, where values[t+1] belongs to the next episode;
    and (b) the final column t=T-1, where values[t+1] lies past the buffer.
    In both cases set bootstrap_values to the true continuation value; leave
    it zero everywhere else.

    The final column has a dual role: it feeds delta[T-1] as the next-state
    value AND seeds the scan carry A_T.  Both uses take the same number, so a
    single entry in bootstrap_values[:, -1] is sufficient.  Set it to V(s_T)
    if the episode continues past the window, or 0 if it terminated at T-1.

    Most users have no interior truncations and should use the `last_value`
    argument (shape [num_envs]) instead, which populates the boundary column
    automatically.

    Args:
        rewards:          Per-step rewards, [num_envs, seq_len], float32, CUDA.
        values:           V(s_t), [num_envs, seq_len], float32, CUDA.
        terminateds:      True termination flags (1.0=terminated),
                          [num_envs, seq_len], float32, CUDA.
        truncateds:       Time-limit truncation flags (1.0=truncated),
                          [num_envs, seq_len], float32, CUDA.
                          If None, terminateds is used for both gating roles
                          (conservative: treats all boundaries as terminations).
        gamma:            Discount factor (default 0.99).
        lambda_:          GAE trace parameter in [0, 1] (default 0.95).
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
        advantages: A[t], shape [num_envs, seq_len], float32.
    """
    num_envs, seq_len = rewards.shape
    has_truncations   = truncateds is not None

    # Cheap structural checks -- always-on.
    for name, t in [("rewards", rewards), ("values", values), ("terminateds", terminateds)]:
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
        assert not has_truncations, \
            "last_value cannot be combined with truncateds; use bootstrap_values instead."
    if bootstrap_values is not None:
        assert bootstrap_values.is_cuda,                "bootstrap_values must be on CUDA"
        assert bootstrap_values.dtype == torch.float32, "bootstrap_values: expected float32"
        assert bootstrap_values.shape == rewards.shape, \
            f"bootstrap_values shape {bootstrap_values.shape} != rewards shape {rewards.shape}"

    # Expensive tensor scans -- correctness-warning path only (not in benchmark hot loop).
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

    rewards     = rewards.contiguous()
    values      = values.contiguous()
    terminateds = terminateds.contiguous()
    if has_truncations:
        truncateds = truncateds.contiguous()

    if last_value is not None:
        scalar_bootstrap = last_value.contiguous()
        bootstrap_values = None
    elif bootstrap_values is not None:
        bootstrap_values = bootstrap_values.contiguous()
        scalar_bootstrap = bootstrap_values[:, -1].contiguous()
    else:
        scalar_bootstrap = None

    out = torch.empty_like(rewards)

    if seq_len <= _FLAT_MAX_SEQ_LEN:
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        num_warps  = _WARPS.get(BLOCK_SIZE, 16)
        num_stages = 2 if BLOCK_SIZE >= 2048 else 1

        if has_truncations:
            if bootstrap_values is None:
                bootstrap_values = torch.zeros_like(rewards)
            gae_fused_kernel[(num_envs,)](
                rewards, values, terminateds, truncateds,
                out, bootstrap_values,
                seq_len, rewards.stride(0),
                gamma=gamma, lambda_=lambda_,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages,
                HAS_TRUNCATIONS=True, HAS_BOOTSTRAP=True,
            )
        else:
            has_bootstrap = scalar_bootstrap is not None
            gae_fused_kernel[(num_envs,)](
                rewards, values, terminateds, None,
                out, scalar_bootstrap,
                seq_len, rewards.stride(0),
                gamma=gamma, lambda_=lambda_,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages,
                HAS_TRUNCATIONS=False, HAS_BOOTSTRAP=has_bootstrap,
            )
        return out

    # Chunked path for seq_len > 131072.
    if not has_truncations:
        truncateds = torch.zeros_like(terminateds)
    if bootstrap_values is None:
        bootstrap_values = torch.zeros_like(rewards)
        if scalar_bootstrap is not None:
            bootstrap_values[:, -1] = scalar_bootstrap

    not_terminated = 1.0 - terminateds
    not_done       = 1.0 - (terminateds + truncateds).clamp(max=1.0)
    next_values    = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = 0.0
    next_values = next_values * (1.0 - truncateds)
    next_values[:, -1]  = 0.0
    next_values = next_values + bootstrap_values

    deltas = rewards + gamma * not_terminated * next_values - values
    decays = gamma * lambda_ * not_done
    carry  = bootstrap_values[:, -1]
    return _run_scan(deltas, decays, carry)
