import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def eligibility_traces_fused_kernel(
    gradients_ptr, dones_ptr,
    out_ptr,
    seed_ptr,
    seq_len,
    stride_env,
    gamma,
    lambda_,
    BLOCK_SIZE: tl.constexpr,
    HAS_SEED: tl.constexpr,
):
    """
    Fully-fused eligibility traces kernel: one program per environment.

    Recurrence (forward, left-to-right):
      z[t] = g[t] + γ*λ*(1-d[t])*z[t-1],  z[-1] = seed

    Maps to A[t] = a[t] + b[t] * A[t-1] with:
      a[t] = g[t]
      b[t] = γ*λ*(1-d[t])

    Unlike all other kernels in this package this scan runs forward in time.
    Padding lanes use a=0, b=1 (identity) to leave the scan result unchanged.

    Indexing (natural order):
      offs = 0, 1, …, BLOCK_SIZE-1   (real time order: offs=0 → t=0)

    Args:
        gradients_ptr: Per-step inputs g[t], [num_envs, seq_len], float32.
        dones_ptr:     Episode termination flags (1.0=done), same shape.
        out_ptr:       Output z[t], same shape.
        seed_ptr:      Initial carry z[-1] per env, [num_envs], float32.
        seq_len:       Number of timesteps.
        stride_env:    Row stride in elements.
        gamma:         Discount factor.
        lambda_:       Trace decay parameter.
        BLOCK_SIZE:    Power-of-2 >= seq_len (constexpr).
        HAS_SEED:      Compile-time flag — False skips the seed_ptr read and uses
                       literal 0.0 (the default z[-1]=0 when no seed_values is given).
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_len

    grad = tl.load(gradients_ptr + base + offs, mask=mask, other=0.0)
    done = tl.load(dones_ptr     + base + offs, mask=mask, other=1.0)

    decay = gamma * lambda_ * (1.0 - done)

    out_local, decay_prod = tl.associative_scan((grad, decay), axis=0, combine_fn=_combine)

    if HAS_SEED:
        seed = tl.load(seed_ptr + env_idx)
    else:
        seed = 0.0
    tl.store(out_ptr + base + offs, out_local + decay_prod * seed, mask=mask)
