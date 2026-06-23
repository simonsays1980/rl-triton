import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def chunked_backward_scan_kernel(
    u_ptr, v_ptr, out_ptr,
    bootstrap_ptr,
    seq_len, stride_env,
    BLOCK_SIZE: tl.constexpr,
    HAS_BOOTSTRAP: tl.constexpr,
):
    """
    Chunked backward scan: A[t] = α[t] + β[t] * A[t+1], A[T] = bootstrap.

    Shared fallback for all estimators (GAE, V-Trace, Retrace, discounted
    returns, lambda returns) when seq_len > 131072, where the flat single-block
    kernel cannot launch.  Splits the sequence into fixed-size chunks and
    processes them right-to-left, carrying the boundary value across chunk
    boundaries.

    Two-pass algorithm per chunk (standard chunked associative scan):
      1. Local scan  — tl.associative_scan within the chunk produces within-chunk
                       outputs as if A[chunk_end+1] = 0.
      2. Carry fixup — add β_prod * carry to every element, where carry is
                       A[chunk_end+1] from the previous (right) chunk and β_prod
                       is the cumulative product of β from each position to the
                       chunk boundary.

    The initial carry is the per-environment bootstrap value A[T], loaded from
    bootstrap_ptr.  Only the leftmost (last-processed) chunk may be partial; all
    intermediate chunks are full, so carry extraction from BLOCK_SIZE-1 is always
    valid for chunks that feed a successor.

    Args:
        u_ptr:         Pointer to additive terms [num_envs, seq_len], float32.
        v_ptr:         Pointer to multiplicative decay factors, same shape.
        out_ptr:       Pointer to output A[t] values, same shape.
        bootstrap_ptr: Pointer to boundary values A[T] per environment,
                       [num_envs], float32. Pass zeros for terminated episodes.
        seq_len:       Number of timesteps (runtime value).
        stride_env:    Row stride in elements (== seq_len for contiguous tensors).
        BLOCK_SIZE:    Chunk size, must be a power of 2.  Need not be >= seq_len.
        HAS_BOOTSTRAP: Compile-time flag — False skips the bootstrap_ptr read and
                       seeds carry with literal 0.0 (the default A[T]=0).
    """
    env_idx = tl.program_id(0)
    base = env_idx * stride_env

    num_chunks = tl.cdiv(seq_len, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Seed the rightmost boundary with the per-environment bootstrap value.
    if HAS_BOOTSTRAP:
        carry = tl.load(bootstrap_ptr + env_idx)
    else:
        carry = 0.0

    for chunk_idx in range(num_chunks):
        # Map chunk_idx=0 to the rightmost chunk (latest in time).
        start = seq_len - 1 - chunk_idx * BLOCK_SIZE
        rev_offsets = start - offsets
        mask = rev_offsets >= 0

        u = tl.load(u_ptr + base + rev_offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + base + rev_offsets, mask=mask, other=1.0)

        # Pass 1: local scan — outputs assuming A[chunk_end+1] = 0.
        out_local, v_prod = tl.associative_scan(
            (u, v), axis=0, combine_fn=_combine
        )

        # Pass 2: propagate the incoming carry from the right.
        # v_prod[i] = product of v[i..chunk_end], so
        # out_local[i] + v_prod[i] * carry gives the correct A[i].
        out = out_local + v_prod * carry

        tl.store(out_ptr + base + rev_offsets, out, mask=mask)

        # Extract A at the leftmost position of this chunk (BLOCK_SIZE-1 in
        # reversed view) to use as carry for the next leftward chunk.
        # tl.sum over a one-hot mask is the idiomatic Triton scalar-extract.
        carry = tl.sum(tl.where(offsets == BLOCK_SIZE - 1, out, 0.0), axis=0)
