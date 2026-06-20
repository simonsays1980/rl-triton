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
    HAS_TRUNCATIONS: tl.constexpr,
):
    """
    Fully-fused GAE kernel: computes advantages in a single program per environment.

    bootstrap_ptr is [num_envs, seq_len] when HAS_TRUNCATIONS=True — nonzero only
    at truncated steps and the window boundary.  When HAS_TRUNCATIONS=False,
    bootstrap_ptr is [num_envs] (one scalar per env) and truncateds_ptr is unused,
    saving two full HBM reads.

      terminated[t]: 1 if episode ended naturally (value bootstrap zeroed in delta).
      done[t]:       1 if episode ended for any reason (trace decay zeroed).
      α[t] = delta[t] = r[t] + gamma*(1-terminated[t])*(values[t+1] + bootstrap[t]) - V(s_t)
      β[t] = decay[t] = gamma * lambda * (1 - done[t])
      A[t] = α[t] + β[t] * A[t+1],  A[T] = bootstrap[T-1]

    Args:
        rewards_ptr:     Per-step rewards [num_envs, seq_len], float32.
        values_ptr:      V(s_t), same shape.
        terminateds_ptr: True termination flags (1.0=terminated), same shape.
        truncateds_ptr:  Truncation flags [num_envs, seq_len]. Unused when HAS_TRUNCATIONS=False.
        out_ptr:         Output advantages A[t], same shape.
        bootstrap_ptr:   [num_envs, seq_len] when HAS_TRUNCATIONS=True;
                         [num_envs] scalar per env when HAS_TRUNCATIONS=False.
        seq_len:         Number of timesteps (runtime value).
        stride_env:      Row stride in elements.
        gamma:           Discount factor (runtime value).
        lambda_:         GAE trace parameter (runtime value).
        BLOCK_SIZE:      Must be >= seq_len and a power of 2 (constexpr).
        HAS_TRUNCATIONS: Compile-time flag — False skips truncateds and 2D bootstrap reads.
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    rev  = seq_len - 1 - offs
    mask = offs < seq_len

    r          = tl.load(rewards_ptr     + base + rev, mask=mask, other=0.0)
    v          = tl.load(values_ptr      + base + rev, mask=mask, other=0.0)
    terminated = tl.load(terminateds_ptr + base + rev, mask=mask, other=1.0)

    if HAS_TRUNCATIONS:
        truncated = tl.load(truncateds_ptr + base + rev, mask=mask, other=0.0)
        bootstrap = tl.load(bootstrap_ptr  + base + rev, mask=mask, other=0.0)
        done      = tl.minimum(terminated + truncated, 1.0)

        # v_next[t] = values[t+1] at non-truncated interior steps.
        # At truncated interior steps or at the boundary (offs==0, t=T-1):
        # bootstrap provides the true continuation value.
        # bootstrap[T-1] is also the scan boundary A[T]; it appears in both
        # delta[T-1] and via decay_prod*carry below (correct for GAE recurrence).
        v_next_raw = tl.load(values_ptr + base + rev + 1,
                             mask=mask & (offs > 0) & (truncated == 0.0), other=0.0)
        v_next = tl.where((offs == 0) | (truncated == 1.0), bootstrap, v_next_raw)

        not_terminated = 1.0 - terminated
        not_done       = 1.0 - done
        delta = r + gamma * v_next * not_terminated - v
        decay = gamma * lambda_ * not_done

        out_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)

        carry = tl.sum(tl.where(offs == 0, bootstrap, 0.0))
        tl.store(out_ptr + base + rev, out_local + decay_prod * carry, mask=mask)
    else:
        not_terminated = 1.0 - terminated
        bootstrap      = tl.load(bootstrap_ptr + env_idx)
        # Load values[t+1] for interior steps; boundary step (offs==0) gets bootstrap
        # injected as the next-state value so delta[T-1] = r + gamma*(1-term)*bootstrap - v.
        # The scan boundary A[T]=bootstrap is then applied via decay_prod*bootstrap below,
        # giving the full GAE recurrence at the window edge.
        v_next_raw = tl.load(values_ptr + base + rev + 1,
                             mask=mask & (offs > 0), other=0.0)
        v_next = tl.where(offs == 0, bootstrap, v_next_raw)

        delta = r + gamma * v_next * not_terminated - v
        decay = gamma * lambda_ * not_terminated

        out_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)
        tl.store(out_ptr + base + rev, out_local + decay_prod * bootstrap, mask=mask)
