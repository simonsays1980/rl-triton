import triton
import triton.language as tl

from rl_triton.kernels.scan import _combine


@triton.jit
def vtrace_scan_kernel(
    delta_ptr, decay_ptr, out_ptr,
    bootstrap_ptr,
    seq_len,
    stride_env,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward scan for V-Trace: Δ[t] = δ[t] + decay[t] * Δ[t+1], Δ[T] = bootstrap.

    Uses the same linear recurrence as GAE with different inputs:
      δ[t]     = ρ[t] * (r[t] + γ * V(s_{t+1}) * (1 - done[t]) - V(s_t))
      decay[t] = γ * c[t] * (1 - done[t])

    bootstrap is 0 for terminated episodes and γ * c[T] * V(s_T) for truncated ones.

    Each program handles one environment (row).  The sequence is loaded in
    reverse order so that tl.associative_scan runs left-to-right over the
    reversed time axis, then the bootstrap carry is applied and results are
    written back in the original order.

    Padding elements use delta=0, decay=1 (identity) so they don't corrupt
    the scan.  BLOCK_SIZE must be >= seq_len and a power of 2.

    Args:
        delta_ptr:     Pointer to clipped TD residuals [num_envs, seq_len], float32.
        decay_ptr:     Pointer to per-step decay factors (γ * c * (1-done)), same shape.
        out_ptr:       Pointer to output Δ values, same shape.
        bootstrap_ptr: Pointer to bootstrap values [num_envs], float32.
                       Pass zeros for terminated episodes.
        seq_len:       Number of timesteps (runtime value).
        stride_env:    Row stride in elements (== seq_len for contiguous tensors).
        BLOCK_SIZE:    Must be >= seq_len and a power of 2.
    """
    env_idx = tl.program_id(0)
    base = env_idx * stride_env

    offsets = tl.arange(0, BLOCK_SIZE)
    rev_offsets = seq_len - 1 - offsets
    mask = offsets < seq_len

    delta = tl.load(delta_ptr + base + rev_offsets, mask=mask, other=0.0)
    decay = tl.load(decay_ptr + base + rev_offsets, mask=mask, other=1.0)

    # Local scan assuming Δ[T] = 0.
    out_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)

    # Apply bootstrap boundary condition: Δ[T] = bootstrap.
    bootstrap = tl.load(bootstrap_ptr + env_idx)
    out = out_local + decay_prod * bootstrap

    tl.store(out_ptr + base + rev_offsets, out, mask=mask)
