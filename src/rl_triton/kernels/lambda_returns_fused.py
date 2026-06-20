import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def lambda_returns_fused_kernel(
    rewards_ptr, next_values_ptr, terminateds_ptr, truncateds_ptr,
    out_ptr,
    bootstrap_ptr,
    seq_len,
    stride_env,
    gamma,
    lambda_,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully-fused TD(λ) returns kernel: one program per environment.

    Recurrence:
      G[t] = r[t] + γ*(1-done[t]) * [(1-λ)*V(s_{t+1}) + λ*G[t+1]]
           + γ*truncated[t]*bootstrap[t]

    Maps to A[t] = α[t] + β[t] * A[t+1] with:
      α[t] = r[t] + γ*(1-λ)*(1-done[t])*V(s_{t+1}) + γ*truncated[t]*bootstrap[t]
      β[t] = γ*λ*(1-done[t])

    next_values_ptr holds V(s_{t+1}) pre-aligned: next_values_ptr[env, t] = V(s_{t+1}).
    For truncated steps the caller sets next_values[t] = 0 and bootstrap[t] = V(s_{t+1}^true),
    so the two are never double-counted.

    bootstrap_ptr is [num_envs, seq_len] — nonzero only at truncated steps and
    the window boundary (t=T-1 when the episode continues past the window).
    The window-boundary carry is bootstrap[T-1], extracted via masked sum.

    Indexing (reversed):
      offs = 0, 1, …, BLOCK_SIZE-1
      rev  = seq_len - 1 - offs      (offs=0 → t=T-1)

    Args:
        rewards_ptr:     [num_envs, seq_len], float32.
        next_values_ptr: V(s_{t+1}), same shape. Zeroed by caller at truncated steps.
        terminateds_ptr: True termination flags (1.0=terminated), same shape.
        truncateds_ptr:  Time-limit truncation flags (1.0=truncated), same shape.
        out_ptr:         Output G[t], same shape.
        bootstrap_ptr:   True continuation values [num_envs, seq_len], float32.
                         Nonzero only at truncated steps and the window boundary.
        seq_len:         Number of timesteps.
        stride_env:      Row stride in elements.
        gamma:           Discount factor.
        lambda_:         Trace parameter.
        BLOCK_SIZE:      Power-of-2 >= seq_len (constexpr).
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    rev  = seq_len - 1 - offs
    mask = offs < seq_len

    r          = tl.load(rewards_ptr     + base + rev, mask=mask, other=0.0)
    nv         = tl.load(next_values_ptr + base + rev, mask=mask, other=0.0)
    terminated = tl.load(terminateds_ptr + base + rev, mask=mask, other=1.0)
    truncated  = tl.load(truncateds_ptr  + base + rev, mask=mask, other=0.0)
    bootstrap  = tl.load(bootstrap_ptr   + base + rev, mask=mask, other=0.0)
    done       = tl.minimum(terminated + truncated, 1.0)

    not_done = 1.0 - done
    u     = r + gamma * (1.0 - lambda_) * not_done * nv + gamma * truncated * bootstrap
    decay = gamma * lambda_ * not_done

    out_local, decay_prod = tl.associative_scan((u, decay), axis=0, combine_fn=_combine)

    carry = tl.sum(tl.where(offs == 0, bootstrap, 0.0))
    tl.store(out_ptr + base + rev, out_local + decay_prod * carry, mask=mask)
