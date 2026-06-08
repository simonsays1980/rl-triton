import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def discounted_returns_fused_kernel(
    rewards_ptr, dones_ptr,
    out_ptr,
    bootstrap_ptr,
    seq_len,
    num_envs,
    stride_env,
    gamma,
    BLOCK_SIZE:    tl.constexpr,
    ENVS_PER_BLOCK: tl.constexpr,
):
    """
    Fully-fused discounted returns kernel.

    Recurrence:
      G[t] = r[t] + γ*(1-d[t])*G[t+1],  G[T] = bootstrap

    Maps to A[t] = u[t] + v[t] * A[t+1] with:
      u[t] = r[t]
      v[t] = γ*(1-d[t])

    ENVS_PER_BLOCK environments are processed sequentially within a single
    Triton program.  This amortises the block-launch and scan-tree overhead
    across more useful work, improving occupancy when num_envs is small
    relative to the number of SMs.

    Indexing (reversed):
      offs = 0, 1, …, BLOCK_SIZE-1
      rev  = seq_len - 1 - offs      (offs=0 → t=T-1)

    Args:
        rewards_ptr:    [num_envs, seq_len], float32.
        dones_ptr:      Episode termination flags (1.0=done), same shape.
        out_ptr:        Output G[t], same shape.
        bootstrap_ptr:  Boundary G[T] per env, [num_envs], float32.
        seq_len:        Number of timesteps.
        num_envs:       Total number of environments.
        stride_env:     Row stride in elements.
        gamma:          Discount factor.
        BLOCK_SIZE:     Power-of-2 >= seq_len (constexpr).
        ENVS_PER_BLOCK: Number of environments to process per block (constexpr).
    """
    block_idx = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    rev  = seq_len - 1 - offs
    mask = offs < seq_len

    for i in tl.static_range(ENVS_PER_BLOCK):
        env_idx = block_idx * ENVS_PER_BLOCK + i
        if env_idx < num_envs:
            base = env_idx * stride_env

            r    = tl.load(rewards_ptr + base + rev, mask=mask, other=0.0)
            done = tl.load(dones_ptr   + base + rev, mask=mask, other=1.0)

            decay = gamma * (1.0 - done)

            out_local, decay_prod = tl.associative_scan((r, decay), axis=0, combine_fn=_combine)

            bootstrap = tl.load(bootstrap_ptr + env_idx)
            tl.store(out_ptr + base + rev, out_local + decay_prod * bootstrap, mask=mask)
