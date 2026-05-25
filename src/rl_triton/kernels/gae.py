import triton
import triton.language as tl


@triton.jit
def _combine(u_a, v_a, u_b, v_b):
    """
    Associative combine for linear recurrence A[t] = delta[t] + decay[t] * A[t+1].

    Represents the affine map  x -> u + v * x.
    Composing right-to-left: f_B(f_A(x)) = u_B + v_B * (u_A + v_A * x).
    """
    return u_b + v_b * u_a, v_a * v_b


@triton.jit
def gae_scan_kernel(
    delta_ptr, decay_ptr, adv_ptr,
    bootstrap_ptr,
    seq_len,
    stride_env,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward scan for GAE: A[t] = delta[t] + decay[t] * A[t+1], A[T] = bootstrap.

    bootstrap is 0 for terminated episodes and V(s_T) for truncated episodes.
    Each program handles one environment (row).  The sequence is loaded in
    reverse order so that tl.associative_scan runs left-to-right over the
    reversed time axis, then the bootstrap carry is applied and results are
    written back in the original order.

    Padding elements (offsets >= seq_len) use delta=0, decay=1 (identity) so
    they don't corrupt the scan.  BLOCK_SIZE must be >= seq_len and a power of 2.

    Args:
        delta_ptr:     Pointer to TD residuals [num_envs, seq_len], float32.
        decay_ptr:     Pointer to per-step decay factors, same shape.
        adv_ptr:       Pointer to output advantages, same shape.
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

    # Local scan assuming A[T] = 0.
    adv_local, decay_prod = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)

    # Apply the bootstrap boundary condition.
    # decay_prod[i] = product of decay[i..T-1], so multiplying by bootstrap
    # correctly propagates A[T] = bootstrap back through all positions.
    bootstrap = tl.load(bootstrap_ptr + env_idx)
    adv = adv_local + decay_prod * bootstrap

    tl.store(adv_ptr + base + rev_offsets, adv, mask=mask)
