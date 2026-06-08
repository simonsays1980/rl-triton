import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def prefix_sum_fused_kernel(
    inputs_ptr, dones_ptr,
    out_ptr,
    seed_ptr,
    seq_len,
    stride_env,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully-fused episodic prefix sum kernel: one program per environment.

    Recurrence (forward, left-to-right):
      C[t] = x[t] + (1-d[t]) * C[t-1],  C[-1] = seed

    Maps to A[t] = u[t] + v[t] * A[t-1] with:
      u[t] = inputs[t]
      v[t] = 1 - dones[t]

    Fuses the intermediate v = (1-dones) tensor computation into the kernel,
    eliminating one full read-write pass over the [num_envs, seq_len] data.

    Indexing (natural forward order):
      offs = 0, 1, …, BLOCK_SIZE-1   (offs=0 → t=0)

    Args:
        inputs_ptr: Values to accumulate x[t], [num_envs, seq_len], float32.
        dones_ptr:  Episode termination flags (1.0=done), same shape.
        out_ptr:    Output C[t], same shape.
        seed_ptr:   Initial carry C[-1] per env, [num_envs], float32.
        seq_len:    Number of timesteps.
        stride_env: Row stride in elements.
        BLOCK_SIZE: Power-of-2 >= seq_len (constexpr).
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_len

    x    = tl.load(inputs_ptr + base + offs, mask=mask, other=0.0)
    done = tl.load(dones_ptr  + base + offs, mask=mask, other=1.0)

    decay = 1.0 - done

    out_local, decay_prod = tl.associative_scan((x, decay), axis=0, combine_fn=_combine)

    seed = tl.load(seed_ptr + env_idx)
    tl.store(out_ptr + base + offs, out_local + decay_prod * seed, mask=mask)
