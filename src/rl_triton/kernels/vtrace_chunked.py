import triton
import triton.language as tl

from rl_triton.kernels.gae import _combine


@triton.jit
def chunked_vtrace_kernel(
    delta_ptr, decay_ptr, out_ptr,
    bootstrap_ptr,
    seq_len, stride_env,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Chunked backward scan for V-Trace: Δ[t] = δ[t] + decay[t] * Δ[t+1], Δ[T] = bootstrap.

    Uses the same linear recurrence as GAE with different inputs:
      δ[t]     = ρ[t] * (r[t] + γ * V(s_{t+1}) * (1 - done[t]) - V(s_t))
      decay[t] = γ * c[t] * (1 - done[t])

    bootstrap is 0 for terminated episodes and γ * c[T] * V(s_T) for truncated ones.

    Splits the sequence into fixed-size chunks processed right-to-left, carrying
    Δ[chunk_end+1] across chunk boundaries.  Removes the BLOCK_SIZE >= seq_len
    constraint of the flat kernel (limited to BLOCK_SIZE <= 131072).

    Two-pass algorithm per chunk:
      1. Local scan  — tl.associative_scan within the chunk, assuming Δ[chunk_end+1] = 0.
      2. Carry fixup — add decay_prod * carry to every element, where carry is
                       Δ[chunk_end+1] from the previous (right) chunk and decay_prod
                       is the cumulative product of decays from each position to the
                       chunk boundary.

    The initial carry is loaded from bootstrap_ptr.  Only the leftmost
    (last-processed) chunk may be partial; all intermediate chunks are full.

    Args:
        delta_ptr:     Pointer to clipped TD residuals [num_envs, seq_len], float32.
        decay_ptr:     Pointer to per-step decay factors (γ * c * (1-done)), same shape.
        out_ptr:       Pointer to output Δ values, same shape.
        bootstrap_ptr: Pointer to bootstrap values [num_envs], float32.
                       Pass zeros for terminated episodes.
        seq_len:       Number of timesteps (runtime value).
        stride_env:    Row stride in elements (== seq_len for contiguous tensors).
        BLOCK_SIZE:    Chunk size, must be a power of 2.  Need not be >= seq_len.
    """
    env_idx = tl.program_id(0)
    base = env_idx * stride_env

    num_chunks = tl.cdiv(seq_len, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)

    carry = tl.load(bootstrap_ptr + env_idx)

    for chunk_idx in range(num_chunks):
        start = seq_len - 1 - chunk_idx * BLOCK_SIZE
        rev_offsets = start - offsets
        mask = rev_offsets >= 0

        delta = tl.load(delta_ptr + base + rev_offsets, mask=mask, other=0.0)
        decay = tl.load(decay_ptr + base + rev_offsets, mask=mask, other=1.0)

        # Pass 1: local scan assuming Δ[chunk_end+1] = 0.
        out_local, decay_prod = tl.associative_scan(
            (delta, decay), axis=0, combine_fn=_combine
        )

        # Pass 2: propagate the incoming carry.
        out = out_local + decay_prod * carry

        tl.store(out_ptr + base + rev_offsets, out, mask=mask)

        # Extract Δ at the leftmost position of this chunk to carry leftward.
        carry = tl.sum(tl.where(offsets == BLOCK_SIZE - 1, out, 0.0), axis=0)
