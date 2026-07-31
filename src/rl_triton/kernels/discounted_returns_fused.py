import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def discounted_returns_fused_kernel(
    rewards_ptr, terminateds_ptr, truncateds_ptr,
    out_ptr,
    bootstrap_ptr,
    seq_len,
    stride_env,
    gamma,
    BLOCK_SIZE: tl.constexpr,
    HAS_TRUNCATIONS: tl.constexpr,
    HAS_BOOTSTRAP: tl.constexpr,
):
    """
    Fully-fused discounted returns kernel: one program per environment.

    Recurrence:
      G[t] = r[t] + γ*(1-terminated[t])*G[t+1] + γ*truncated[t]*bootstrap[t]

    Maps to A[t] = α[t] + β[t] * A[t+1] with:
      α[t] = r[t] + gamma * truncated[t] * bootstrap[t]
      β[t] = gamma * (1 - done[t]),   done[t] = terminated[t] | truncated[t]

    When HAS_TRUNCATIONS=False, truncateds_ptr is unused and bootstrap_ptr is
    [num_envs] (scalar per env), saving two full HBM reads.

    When HAS_TRUNCATIONS=True, bootstrap_ptr is [num_envs, seq_len] -- nonzero
    only at truncated steps and the window boundary.

    Args:
        rewards_ptr:     [num_envs, seq_len], float32.
        terminateds_ptr: True termination flags (1.0=terminated), same shape.
        truncateds_ptr:  Truncation flags, same shape. Unused when HAS_TRUNCATIONS=False.
        out_ptr:         Output G[t], same shape.
        bootstrap_ptr:   [num_envs, seq_len] when HAS_TRUNCATIONS=True;
                         [num_envs] scalar per env when HAS_TRUNCATIONS=False.
        seq_len:         Number of timesteps.
        stride_env:      Row stride in elements.
        gamma:           Discount factor.
        BLOCK_SIZE:      Power-of-2 >= seq_len (constexpr).
        HAS_TRUNCATIONS: Compile-time flag -- False skips truncateds and 2D bootstrap reads.
        HAS_BOOTSTRAP:   Compile-time flag, only meaningful when HAS_TRUNCATIONS=False --
                         False skips the scalar bootstrap_ptr read and uses literal 0.0.
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    rev  = seq_len - 1 - offs
    mask = offs < seq_len

    r          = tl.load(rewards_ptr     + base + rev, mask=mask, other=0.0)
    terminated = tl.load(terminateds_ptr + base + rev, mask=mask, other=1.0)

    if HAS_TRUNCATIONS:
        truncated = tl.load(truncateds_ptr + base + rev, mask=mask, other=0.0)
        bootstrap = tl.load(bootstrap_ptr  + base + rev, mask=mask, other=0.0)
        done      = tl.minimum(terminated + truncated, 1.0)

        u     = r + gamma * truncated * bootstrap
        decay = gamma * (1.0 - done)

        out_local, decay_prod = tl.associative_scan((u, decay), axis=0, combine_fn=_combine)

        carry = tl.sum(tl.where(offs == 0, bootstrap, 0.0))
        tl.store(out_ptr + base + rev, out_local + decay_prod * carry, mask=mask)
    else:
        if HAS_BOOTSTRAP:
            bootstrap = tl.load(bootstrap_ptr + env_idx)
        else:
            bootstrap = 0.0

        u     = r
        decay = gamma * (1.0 - terminated)

        out_local, decay_prod = tl.associative_scan((u, decay), axis=0, combine_fn=_combine)
        tl.store(out_ptr + base + rev, out_local + decay_prod * bootstrap, mask=mask)
