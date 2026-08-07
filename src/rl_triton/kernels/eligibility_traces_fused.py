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
      z[t] = g[t] + γ*λ*(1-d[t-1])*z[t-1],  z[-1] = seed,  d[-1] := 0

    Maps to A[t] = a[t] + b[t] * A[t-1] with:
      a[t] = g[t]
      b[t] = γ*λ*(1-d[t-1])

    Boundary convention: d[t]=1 means episode t ends AT t -- the SAME
    convention used by every backward kernel in this package (compute_gae,
    compute_lambda_returns, etc.), matching a raw Gymnasium
    next-step-autoreset `terminated`/`truncated` flag directly. The trace
    carried INTO t must therefore be severed when t is the FIRST step of a
    new episode, i.e. when the PRECEDING step ended one (d[t-1]=1) -- not
    when t itself ends one (d[t]=1). Gating on d[t] instead would sever the
    carry one step too early (at the old episode's own last step, which
    still legitimately belongs to that episode) and fail to sever it at the
    new episode's first step, leaking trace mass across the boundary. This
    makes eligibility traces boundary-consistent with compute_gae and every
    other kernel in this package when fed the same `dones` buffer.

    Unlike all other kernels in this package this scan runs forward in time,
    so padding lanes (offs >= seq_len) sit AFTER every valid lane in scan
    order and can only receive state from valid lanes, never feed into one --
    their exact a/b values are therefore inert regardless of mask fallback,
    unlike the backward kernels where padding precedes valid data.

    Indexing (natural order):
      offs = 0, 1, …, BLOCK_SIZE-1   (real time order: offs=0 → t=0)

    Args:
        gradients_ptr: Per-step inputs g[t], [num_envs, seq_len], float32.
        dones_ptr:     Episode termination flags (1.0=done), same shape.
                       d[t]=1 means episode ends AT t (same convention as
                       compute_gae etc.); this kernel reads d[t-1]
                       internally -- the caller does not shift anything.
        out_ptr:       Output z[t], same shape.
        seed_ptr:      Initial carry z[-1] per env, [num_envs], float32.
        seq_len:       Number of timesteps.
        stride_env:    Row stride in elements.
        gamma:         Discount factor.
        lambda_:       Trace decay parameter.
        BLOCK_SIZE:    Power-of-2 >= seq_len (constexpr).
        HAS_SEED:      Compile-time flag -- False skips the seed_ptr read and uses
                       literal 0.0 (the default z[-1]=0 when no seed_values is given).
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_len

    grad      = tl.load(gradients_ptr + base + offs,     mask=mask,              other=0.0)
    done_prev = tl.load(dones_ptr     + base + offs - 1, mask=mask & (offs > 0), other=0.0)

    decay = gamma * lambda_ * (1.0 - done_prev)

    out_local, decay_prod = tl.associative_scan((grad, decay), axis=0, combine_fn=_combine)

    if HAS_SEED:
        seed = tl.load(seed_ptr + env_idx)
    else:
        seed = 0.0
    tl.store(out_ptr + base + offs, out_local + decay_prod * seed, mask=mask)
