import triton
import triton.language as tl

from rl_triton.kernels.gae import _combine


@triton.jit
def chunked_gae_kernel(
    delta_ptr, decay_ptr, adv_ptr,
    seq_len, stride_env,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Chunked backward scan for GAE: A[t] = delta[t] + decay[t] * A[t+1], A[T] = 0.

    Splits the sequence into fixed-size chunks and processes them right-to-left,
    carrying the boundary value across chunk boundaries.  This removes the
    BLOCK_SIZE >= seq_len constraint of the flat kernel (which is limited to
    BLOCK_SIZE <= 131072), enabling arbitrarily long sequences at the cost of
    one sequential loop over chunks per environment.

    Two-pass algorithm per chunk (standard chunked associative scan):
      1. Local scan  — tl.associative_scan within the chunk produces within-chunk
                       advantages as if A[chunk_end+1] = 0.
      2. Carry fixup — add decay_prod * carry_adv to every element, where
                       carry_adv is A[chunk_end+1] from the previous (right) chunk
                       and decay_prod is the cumulative product of decays from each
                       position to the chunk boundary.

    Only the leftmost (last-processed) chunk may be partial; all intermediate
    chunks are full, so carry extraction from BLOCK_SIZE-1 is always valid for
    chunks that feed a successor.

    Args:
        delta_ptr:  Pointer to TD residuals [num_envs, seq_len], float32.
        decay_ptr:  Pointer to per-step decay factors, same shape.
        adv_ptr:    Pointer to output advantages, same shape.
        seq_len:    Number of timesteps (runtime value).
        stride_env: Row stride in elements (== seq_len for contiguous tensors).
        BLOCK_SIZE: Chunk size, must be a power of 2.  Need not be >= seq_len.
    """
    env_idx = tl.program_id(0)
    base = env_idx * stride_env

    num_chunks = tl.cdiv(seq_len, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)

    # A[T] = 0; updated at each chunk boundary as we sweep right-to-left.
    carry_adv = 0.0

    for chunk_idx in range(num_chunks):
        # Map chunk_idx=0 to the rightmost chunk (latest in time).
        start = seq_len - 1 - chunk_idx * BLOCK_SIZE
        rev_offsets = start - offsets
        mask = rev_offsets >= 0

        delta = tl.load(delta_ptr + base + rev_offsets, mask=mask, other=0.0)
        decay = tl.load(decay_ptr + base + rev_offsets, mask=mask, other=1.0)

        # Pass 1: local scan — advantages assuming A[chunk_end+1] = 0.
        adv_local, decay_prod = tl.associative_scan(
            (delta, decay), axis=0, combine_fn=_combine
        )

        # Pass 2: propagate the incoming carry from the right.
        # decay_prod[i] = product of decay[i..chunk_end], so
        # adv_local[i] + decay_prod[i] * carry_adv gives the correct A[i].
        adv = adv_local + decay_prod * carry_adv

        tl.store(adv_ptr + base + rev_offsets, adv, mask=mask)

        # Extract A at the leftmost position of this chunk (BLOCK_SIZE-1 in reversed
        # view) to use as carry_adv for the next leftward chunk.
        # tl.sum over a one-hot mask is the idiomatic Triton scalar-extract.
        carry_adv = tl.sum(tl.where(offsets == BLOCK_SIZE - 1, adv, 0.0), axis=0)