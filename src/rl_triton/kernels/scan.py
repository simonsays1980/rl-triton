"""
Triton kernels for the backward and forward linear recurrence:

    Backward:  A[t] = u[t] + v[t] * A[t+1],  A[T] = bootstrap   (right-to-left)
    Forward:   e[t] = u[t] + v[t] * e[t-1],  e[-1] = seed       (left-to-right)

Both directions share the same associative structure and are used by multiple
RL algorithms:

    Backward: GAE, discounted returns, lambda-returns, V-Trace, Retrace(λ)
    Forward:  eligibility traces

The two combine functions (_combine / _combine_fwd) are the only algorithmic
difference — one composes affine maps right-to-left, the other left-to-right.
"""
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Combine functions
# ---------------------------------------------------------------------------

@triton.jit
def _combine(u_a, v_a, u_b, v_b):
    """
    Associative combine for the backward recurrence (right-to-left composition).

    Affine map x -> u + v*x.  Composing f_B after f_A:
      f_B(f_A(x)) = u_B + v_B*(u_A + v_A*x) = (u_B + v_B*u_A) + (v_A*v_B)*x
    """
    return u_b + v_b * u_a, v_a * v_b


@triton.jit
def _combine_fwd(u_a, v_a, u_b, v_b):
    """
    Associative combine for the forward recurrence (left-to-right composition).

    Affine map x -> u + v*x.  Composing f_B after f_A with A applied first:
      f_B(f_A(x)) = u_A + v_A*(u_B + v_B*x) = (u_A + v_A*u_B) + (v_A*v_B)*x
    """
    return u_a + v_a * u_b, v_a * v_b


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@triton.jit
def backward_scan_kernel(
    u_ptr, v_ptr, out_ptr,
    bootstrap_ptr,
    seq_len,
    stride_env,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward scan: A[t] = u[t] + v[t] * A[t+1], A[T] = bootstrap.

    Loads u and v in reverse time order so tl.associative_scan sweeps
    left-to-right over the reversed axis, then applies the bootstrap boundary
    condition and writes results back in original order.

    Padding lanes (offsets >= seq_len) use u=0, v=1 (identity) so they do not
    corrupt valid positions.  BLOCK_SIZE must be >= seq_len and a power of 2.

    Args:
        u_ptr:         Additive term [num_envs, seq_len], float32.
        v_ptr:         Multiplicative decay, same shape.
        out_ptr:       Output A[t] values, same shape.
        bootstrap_ptr: Boundary value A[T] per environment, [num_envs], float32.
                       Pass zeros for terminated episodes.
        seq_len:       Number of timesteps (runtime value).
        stride_env:    Row stride in elements.
        BLOCK_SIZE:    Must be >= seq_len and a power of 2.
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offsets     = tl.arange(0, BLOCK_SIZE)
    rev_offsets = seq_len - 1 - offsets
    mask        = offsets < seq_len

    u = tl.load(u_ptr + base + rev_offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + base + rev_offsets, mask=mask, other=1.0)

    out_local, v_prod = tl.associative_scan((u, v), axis=0, combine_fn=_combine)

    bootstrap = tl.load(bootstrap_ptr + env_idx)
    out       = out_local + v_prod * bootstrap

    tl.store(out_ptr + base + rev_offsets, out, mask=mask)


@triton.jit
def forward_scan_kernel(
    u_ptr, v_ptr, out_ptr,
    seed_ptr,
    seq_len,
    stride_env,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward scan: e[t] = u[t] + v[t] * e[t-1], e[-1] = seed.

    Loads u and v in natural (left-to-right) order and applies the seed
    boundary condition after the scan.

    Padding lanes (offsets >= seq_len) use u=0, v=1 (identity).
    BLOCK_SIZE must be >= seq_len and a power of 2.

    Args:
        u_ptr:      Additive term [num_envs, seq_len], float32.
        v_ptr:      Multiplicative decay, same shape.
        out_ptr:    Output e[t] values, same shape.
        seed_ptr:   Initial carry e[-1] per environment, [num_envs], float32.
                    Pass zeros for traces starting from scratch.
        seq_len:    Number of timesteps (runtime value).
        stride_env: Row stride in elements.
        BLOCK_SIZE: Must be >= seq_len and a power of 2.
    """
    env_idx = tl.program_id(0)
    base    = env_idx * stride_env

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < seq_len

    u = tl.load(u_ptr + base + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + base + offsets, mask=mask, other=1.0)

    out_local, v_prod = tl.associative_scan((u, v), axis=0, combine_fn=_combine_fwd)

    seed = tl.load(seed_ptr + env_idx)
    out  = out_local + v_prod * seed

    tl.store(out_ptr + base + offsets, out, mask=mask)
