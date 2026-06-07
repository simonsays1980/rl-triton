import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def gae_fused_kernel(
    rewards_ptr, values_ptr, dones_ptr,
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

    Replaces three separate PyTorch elementwise kernels (not_done, deltas, decays)
    plus the scan kernel, eliminating all intermediate tensor allocations and
    kernel launch overhead.  All inputs are read once; the result is written once.

    v_{t+1} is read directly from values[t+1] (no separate next_values tensor),
    saving one full HBM read pass (~20% bandwidth reduction).  At reversed position
    offs=0 (real time t=T-1), V(s_T) comes from bootstrap_ptr; for all other
    positions load values[rev_offsets + 1] = values[t+1].

      u[t] = delta[t] = r[t] + gamma * (1 - done[t]) * V(s_{t+1}) - V(s_t)
      v[t] = decay[t] = gamma * lambda * (1 - done[t])
      A[t] = u[t] + v[t] * A[t+1],  A[T] = bootstrap

    Args:
        rewards_ptr:   Per-step rewards [num_envs, seq_len], float32.
        values_ptr:    V(s_t), same shape.  V(s_{t+1}) is loaded from values[t+1].
        dones_ptr:     Episode termination flags (1.0=done), same shape.
        out_ptr:       Output advantages A[t], same shape.
        bootstrap_ptr: Boundary value A[T] / V(s_T) per env, [num_envs], float32.
        seq_len:       Number of timesteps (runtime value).
        stride_env:    Row stride in elements.
        gamma:         Discount factor (runtime value).
        lambda_:       GAE trace parameter (runtime value).
        BLOCK_SIZE:    Must be >= seq_len and a power of 2 (constexpr).
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs        = tl.arange(0, BLOCK_SIZE)
    rev         = seq_len - 1 - offs          # offs=0 → t=T-1, offs=1 → t=T-2, …
    mask        = offs < seq_len

    r    = tl.load(rewards_ptr + base + rev, mask=mask, other=0.0)
    v    = tl.load(values_ptr  + base + rev, mask=mask, other=0.0)
    done = tl.load(dones_ptr   + base + rev, mask=mask, other=1.0)

    # v_next[t] = values[t+1].
    # offs=0 is t=T-1; its v_next is bootstrap (V(s_T)).
    # offs>0 is t<T-1; load values[t+1] = values[rev+1].
    bootstrap  = tl.load(bootstrap_ptr + env_idx)
    v_next_raw = tl.load(values_ptr + base + rev + 1,
                         mask=mask & (offs > 0), other=0.0)
    v_next = tl.where(offs == 0, bootstrap, v_next_raw)

    not_done = 1.0 - done
    delta    = r + gamma * v_next * not_done - v
    decay    = gamma * lambda_ * not_done

    out_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)
    out = out_local + decay_prod * bootstrap

    tl.store(out_ptr + base + rev, out, mask=mask)
