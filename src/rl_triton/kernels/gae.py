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
    seq_len,
    stride_env,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward scan for GAE: A[t] = delta[t] + decay[t] * A[t+1], A[T] = 0.

    Each program handles one environment (row).  The sequence is loaded in
    reverse order so that tl.associative_scan runs left-to-right over the
    reversed time axis, then results are written back in the original order.

    Padding elements (offsets >= seq_len) use delta=0, decay=1 (identity) so
    they don't corrupt the scan.  BLOCK_SIZE must be >= seq_len and a power of 2.
    """
    env_idx = tl.program_id(0)
    base = env_idx * stride_env

    offsets = tl.arange(0, BLOCK_SIZE)
    rev_offsets = seq_len - 1 - offsets
    mask = offsets < seq_len

    delta = tl.load(delta_ptr + base + rev_offsets, mask=mask, other=0.0)
    decay = tl.load(decay_ptr + base + rev_offsets, mask=mask, other=1.0)

    adv, _ = tl.associative_scan((delta, decay), axis=0, combine_fn=_combine)

    tl.store(adv_ptr + base + rev_offsets, adv, mask=mask)
