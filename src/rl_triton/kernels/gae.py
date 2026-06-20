import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def gae_fused_kernel(
    rewards_ptr, values_ptr, terminateds_ptr, truncateds_ptr,
    out_ptr,
    bootstrap_ptr,
    seq_len,
    stride_env,
    gamma,
    lambda_,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully-fused GAE kernel: computes advantages in a single program per environment.

    bootstrap_ptr is [num_envs, seq_len] — nonzero only at truncated steps and at
    the window boundary (t=T-1 when the episode continues past the window).  At
    every step the kernel adds bootstrap[t] to v_next_raw (which is zero-masked
    at truncated and boundary positions), so no tl.where is needed.

      terminated[t]: 1 if episode ended naturally (bootstrap zeroed in delta).
      done[t]:       1 if episode ended for any reason (trace decay zeroed).
      α[t] = delta[t] = r[t] + gamma*(1-terminated[t])*(values[t+1] + bootstrap[t]) - V(s_t)
      β[t] = decay[t] = gamma * lambda * (1 - done[t])
      A[t] = α[t] + β[t] * A[t+1],  A[T] = bootstrap[T-1]  (carried via decay_prod)

    Args:
        rewards_ptr:     Per-step rewards [num_envs, seq_len], float32.
        values_ptr:      V(s_t), same shape.  V(s_{t+1}) is loaded from values[t+1].
        terminateds_ptr: True termination flags (1.0=terminated), same shape.
        truncateds_ptr:  Time-limit truncation flags (1.0=truncated), same shape.
        out_ptr:         Output advantages A[t], same shape.
        bootstrap_ptr:   True continuation values [num_envs, seq_len], float32.
                         Nonzero only at truncated steps and the window boundary.
        seq_len:         Number of timesteps (runtime value).
        stride_env:      Row stride in elements.
        gamma:           Discount factor (runtime value).
        lambda_:         GAE trace parameter (runtime value).
        BLOCK_SIZE:      Must be >= seq_len and a power of 2 (constexpr).
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    rev  = seq_len - 1 - offs          # offs=0 → t=T-1, offs=1 → t=T-2, …
    mask = offs < seq_len

    r          = tl.load(rewards_ptr     + base + rev, mask=mask, other=0.0)
    v          = tl.load(values_ptr      + base + rev, mask=mask, other=0.0)
    terminated = tl.load(terminateds_ptr + base + rev, mask=mask, other=1.0)
    truncated  = tl.load(truncateds_ptr  + base + rev, mask=mask, other=0.0)
    bootstrap  = tl.load(bootstrap_ptr   + base + rev, mask=mask, other=0.0)
    done       = tl.minimum(terminated + truncated, 1.0)

    # v_next[t] = values[t+1] for interior non-truncated steps; 0 otherwise.
    # bootstrap[t] carries the true continuation value at truncated/boundary steps.
    # The two are additive because exactly one is nonzero at any given step.
    v_next_raw = tl.load(values_ptr + base + rev + 1,
                         mask=mask & (offs > 0) & (truncated == 0.0), other=0.0)
    v_next = v_next_raw + bootstrap

    not_terminated = 1.0 - terminated
    not_done       = 1.0 - done
    delta = r + gamma * v_next * not_terminated - v
    decay = gamma * lambda_ * not_done

    out_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)

    # Scan carry: bootstrap[T-1] is the window-boundary continuation value.
    # offs=0 corresponds to t=T-1; bootstrap[offs=0] is already loaded above.
    # We need the scalar for the carry: extract it via a masked reduction.
    carry = tl.sum(tl.where(offs == 0, bootstrap, 0.0))
    out   = out_local + decay_prod * carry

    tl.store(out_ptr + base + rev, out, mask=mask)
