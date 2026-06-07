import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def gae_fused_kernel(
    rewards_ptr, values_ptr, dones_ptr,
    out_ptr,
    bootstrap_ptr,
    num_envs,
    seq_len,
    stride_env,
    gamma,
    lambda_,
    ENVS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully-fused GAE kernel: computes advantages in a single program per environment.

    Replaces three separate PyTorch elementwise kernels (not_done, deltas, decays)
    plus the scan kernel, eliminating all intermediate tensor allocations and
    kernel launch overhead.  All inputs are read once; the result is written once.

    v_{t+1} is read directly from values[t+1] (no separate next_values tensor),
    saving one full HBM read pass (~25% bandwidth reduction).  At t=T-1 (reversed
    position 0), V(s_T) is taken from bootstrap_ptr, which doubles as both A[T]
    and V(s_T).

      u[t] = delta[t] = r[t] + gamma * (1 - done[t]) * V(s_{t+1}) - V(s_t)
      v[t] = decay[t] = gamma * lambda * (1 - done[t])
      A[t] = u[t] + v[t] * A[t+1],  A[T] = bootstrap

    Args:
        rewards_ptr:    Per-step rewards [num_envs, seq_len], float32.
        values_ptr:     V(s_t), same shape.  V(s_{t+1}) is loaded from values[t+1].
        dones_ptr:      Episode termination flags (1.0=done), same shape.
        out_ptr:        Output advantages A[t], same shape.
        bootstrap_ptr:  Boundary value A[T] / V(s_T) per env, [num_envs], float32.
        num_envs:       Total number of environments (runtime value).
        seq_len:        Number of timesteps (runtime value).
        stride_env:     Row stride in elements.
        gamma:          Discount factor (runtime value).
        lambda_:        GAE trace parameter (runtime value).
        ENVS_PER_BLOCK: Number of environments per program instance (constexpr).
        BLOCK_SIZE:     Must be >= seq_len and a power of 2 (constexpr).
    """
    for i in tl.static_range(ENVS_PER_BLOCK):
        env_idx = tl.program_id(0) * ENVS_PER_BLOCK + i
        if env_idx < num_envs:
            base = env_idx * stride_env

            offsets     = tl.arange(0, BLOCK_SIZE)
            rev_offsets = seq_len - 1 - offsets      # pos 0 = last real timestep
            mask        = offsets < seq_len

            r    = tl.load(rewards_ptr + base + rev_offsets, mask=mask, other=0.0)
            v    = tl.load(values_ptr  + base + rev_offsets, mask=mask, other=0.0)
            done = tl.load(dones_ptr   + base + rev_offsets, mask=mask, other=1.0)

            # V(s_{t+1}): at reversed pos 0 (t=T-1), use bootstrap as V(s_T).
            bootstrap  = tl.load(bootstrap_ptr + env_idx)
            v_next_raw = tl.load(values_ptr + base + rev_offsets - 1,
                                 mask=mask & (rev_offsets > 0), other=0.0)
            v_next = tl.where(offsets == 0, bootstrap, v_next_raw)

            not_done = 1.0 - done
            delta    = r + gamma * v_next * not_done - v
            decay    = gamma * lambda_ * not_done

            out_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)
            out = out_local + decay_prod * bootstrap

            tl.store(out_ptr + base + rev_offsets, out, mask=mask)
