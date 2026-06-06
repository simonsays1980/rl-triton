import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},   num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},   num_warps=8),
        triton.Config({"BLOCK_SIZE": 512},   num_warps=16),
        triton.Config({"BLOCK_SIZE": 1024},  num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024},  num_warps=16),
        triton.Config({"BLOCK_SIZE": 2048},  num_warps=16),
        triton.Config({"BLOCK_SIZE": 4096},  num_warps=16),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=16),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 65536}, num_warps=16),
    ],
    key=["seq_len"],
)
@triton.jit
def gae_fused_kernel(
    rewards_ptr, values_ptr, next_values_ptr, dones_ptr,
    out_ptr,
    bootstrap_ptr,
    seq_len,
    stride_env,
    gamma: tl.constexpr,
    lambda_: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully-fused GAE kernel: computes advantages in a single program per environment.

    Replaces three separate PyTorch elementwise kernels (not_done, deltas, decays)
    plus the scan kernel, eliminating all intermediate tensor allocations and
    kernel launch overhead.  All inputs are read once; the result is written once.

      u[t] = delta[t] = r[t] + gamma * (1 - done[t]) * V(s_{t+1}) - V(s_t)
      v[t] = decay[t] = gamma * lambda * (1 - done[t])
      A[t] = u[t] + v[t] * A[t+1],  A[T] = bootstrap

    Args:
        rewards_ptr:    Per-step rewards [num_envs, seq_len], float32.
        values_ptr:     V(s_t), same shape.
        next_values_ptr: V(s_{t+1}), same shape.
        dones_ptr:      Episode termination flags (1.0=done), same shape.
        out_ptr:        Output advantages A[t], same shape.
        bootstrap_ptr:  Boundary value A[T] per env, [num_envs], float32.
        seq_len:        Number of timesteps (runtime value).
        stride_env:     Row stride in elements.
        gamma:          Discount factor (compile-time constant).
        lambda_:        GAE trace parameter (compile-time constant).
        BLOCK_SIZE:     Must be >= seq_len and a power of 2.
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offsets     = tl.arange(0, BLOCK_SIZE)
    rev_offsets = seq_len - 1 - offsets
    mask        = offsets < seq_len

    r      = tl.load(rewards_ptr      + base + rev_offsets, mask=mask, other=0.0)
    v      = tl.load(values_ptr       + base + rev_offsets, mask=mask, other=0.0)
    v_next = tl.load(next_values_ptr  + base + rev_offsets, mask=mask, other=0.0)
    done   = tl.load(dones_ptr        + base + rev_offsets, mask=mask, other=1.0)

    not_done = 1.0 - done
    delta    = r + gamma * v_next * not_done - v
    decay    = gamma * lambda_ * not_done

    out_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)

    bootstrap = tl.load(bootstrap_ptr + env_idx)
    out       = out_local + decay_prod * bootstrap

    tl.store(out_ptr + base + rev_offsets, out, mask=mask)
