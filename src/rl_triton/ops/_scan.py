"""
Internal scan dispatch used by all RL return estimators.

Every algorithm in this package reduces to the same linear recurrence:

    A[t] = u[t] + v[t] * A[t+1],   A[T] = bootstrap

with algorithm-specific definitions of u and v:

    Discounted returns:  u = r,          v = γ(1-d)
    GAE:                 u = δ_gae,      v = γλ(1-d)
    V-Trace:             u = ρδ_vt,      v = γc(1-d)
    Retrace(λ):          u = δ_ret,      v = γc_{t+1}(1-d)

_run_scan(u, v, bootstrap) is the single implementation of this recurrence.
It owns all input validation, contiguous enforcement, and flat/chunked dispatch.
Public functions (compute_gae_triton, compute_discounted_returns, etc.) are thin
wrappers that compute u and v from raw RL quantities, then call _run_scan.
"""
import torch
import triton

from rl_triton.kernels.gae import gae_scan_kernel
from rl_triton.kernels.gae_chunked import chunked_gae_kernel

# tl.associative_scan requires BLOCK_SIZE <= 2^17.  Above this the flat kernel
# cannot launch; the chunked kernel handles arbitrary lengths.
_FLAT_MAX_SEQ_LEN = 131072
_CHUNK_SIZE = triton.next_power_of_2(1024)


def _run_scan(
    u: torch.Tensor,
    v: torch.Tensor,
    bootstrap: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Backward linear scan: A[t] = u[t] + v[t] * A[t+1], A[T] = bootstrap.

    Dispatches to the flat single-block kernel for seq_len <= 131072 and the
    chunked kernel for longer sequences.

    Args:
        u:         Additive term [num_envs, seq_len], float32, CUDA, contiguous.
        v:         Multiplicative decay [num_envs, seq_len], float32, CUDA, contiguous.
        bootstrap: Per-environment boundary value A[T], shape [num_envs], float32.
                   Defaults to zeros (terminated episodes).

    Returns:
        out: A[t] values, shape [num_envs, seq_len], float32.
    """
    assert u.shape == v.shape,   "u and v must have the same shape"
    assert u.is_cuda and v.is_cuda, "u and v must be on CUDA"
    assert u.dtype == torch.float32, f"u: expected float32, got {u.dtype}"
    assert v.dtype == torch.float32, f"v: expected float32, got {v.dtype}"

    u = u.contiguous()
    v = v.contiguous()

    num_envs, seq_len = u.shape

    if bootstrap is None:
        bootstrap = torch.zeros(num_envs, device=u.device, dtype=torch.float32)
    else:
        assert bootstrap.shape == (num_envs,), \
            f"bootstrap must have shape [{num_envs}], got {bootstrap.shape}"
        assert bootstrap.is_cuda, "bootstrap must be on CUDA"
        bootstrap = bootstrap.contiguous()

    out = torch.empty_like(u)

    if seq_len > _FLAT_MAX_SEQ_LEN:
        chunked_gae_kernel[(num_envs,)](
            u, v, out,
            bootstrap,
            seq_len,
            u.stride(0),
            BLOCK_SIZE=_CHUNK_SIZE,
        )
    else:
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        gae_scan_kernel[(num_envs,)](
            u, v, out,
            bootstrap,
            seq_len,
            u.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out
