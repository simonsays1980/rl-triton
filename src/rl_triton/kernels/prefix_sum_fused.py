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
    HAS_SEED: tl.constexpr,
    BOUNDARY_ENDS_AT: tl.constexpr,
):
    """
    Fully-fused episodic prefix sum kernel: one program per environment.

    Two mutually-exclusive boundary conventions, selected at compile time by
    BOUNDARY_ENDS_AT -- see compute_episodic_prefix_sum's docstring for the
    full rationale (RL rollout-buffer consistency vs. sequence-packing
    position IDs).

    BOUNDARY_ENDS_AT=True (default, "ends_at"):
      C[t] = x[t] + (1-d[t-1]) * C[t-1],  C[-1] = seed,  d[-1] := 0
      d[t]=1 means the segment ENDS at t; the reset lands at t+1. Same
      convention as every backward kernel in this package (compute_gae
      etc.) and every RL-oriented forward kernel (compute_eligibility_traces).

    BOUNDARY_ENDS_AT=False ("starts_at"):
      C[t] = x[t] + (1-d[t]) * C[t-1],  C[-1] = seed
      d[t]=1 means the segment STARTS at t; the reset lands at t itself.
      This is the convention sequence-packing callers want (a document-
      boundary flag marks a new document's first token, and a RoPE local
      position counter must reset exactly there).

    Maps to A[t] = a[t] + b[t] * A[t-1] with:
      a[t] = inputs[t]
      b[t] = 1 - dones[t]        (starts_at)  or  1 - dones[t-1]  (ends_at)

    Fuses the intermediate b = (1-dones) tensor computation into the kernel,
    eliminating one full read-write pass over the [num_envs, seq_len] data.

    Indexing (natural forward order):
      offs = 0, 1, …, BLOCK_SIZE-1   (offs=0 → t=0)

    Args:
        inputs_ptr:       Values to accumulate x[t], [num_envs, seq_len], float32.
        dones_ptr:        Episode/segment boundary flags (1.0=flagged), same shape.
                           Meaning of the flag depends on BOUNDARY_ENDS_AT; the
                           caller does not shift this array either way.
        out_ptr:          Output C[t], same shape.
        seed_ptr:         Initial carry C[-1] per env, [num_envs], float32.
        seq_len:          Number of timesteps.
        stride_env:       Row stride in elements.
        BLOCK_SIZE:       Power-of-2 >= seq_len (constexpr).
        HAS_SEED:         Compile-time flag -- False skips the seed_ptr read and uses
                          literal 0.0 (the default C[-1]=0 when no seed_values is given).
        BOUNDARY_ENDS_AT: Compile-time flag selecting the boundary convention (see above).
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_len

    x = tl.load(inputs_ptr + base + offs, mask=mask, other=0.0)

    if BOUNDARY_ENDS_AT:
        done = tl.load(dones_ptr + base + offs - 1, mask=mask & (offs > 0), other=0.0)
    else:
        done = tl.load(dones_ptr + base + offs,     mask=mask,              other=1.0)

    decay = 1.0 - done

    out_local, decay_prod = tl.associative_scan((x, decay), axis=0, combine_fn=_combine)

    if HAS_SEED:
        seed = tl.load(seed_ptr + env_idx)
    else:
        seed = 0.0
    tl.store(out_ptr + base + offs, out_local + decay_prod * seed, mask=mask)
