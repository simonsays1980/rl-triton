"""
Internal scan dispatch used by all RL sequence estimators.

Two directions of the same linear recurrence are supported:

Backward (right-to-left): A[t] = u[t] + v[t] * A[t+1],  A[T] = bootstrap
    Discounted returns:  u = r,          v = γ(1-d)
    GAE:                 u = δ_gae,      v = γλ(1-d)
    V-Trace:             u = ρδ_vt,      v = γc(1-d)
    Retrace(λ):          u = δ_ret,      v = γc_{t+1}(1-d)
    Lambda returns:      u = r+γ(1-λ)(1-d)V', v = γλ(1-d)

Forward (left-to-right): e[t] = u[t] + v[t] * e[t-1],   e[-1] = seed
    Eligibility traces:  u = g_t,        v = γλ(1-d)

_run_scan / _run_scan_forward own all input validation, contiguous enforcement,
and flat/chunked dispatch.  Public wrappers compute u and v, then call them.
"""
import os
import warnings

import torch
import triton

# Set RL_TRITON_PERF_WARNINGS=1 to enable warnings about performance bottlenecks
# such as non-contiguous inputs that trigger implicit copies.
_PERF_WARNINGS = os.environ.get("RL_TRITON_PERF_WARNINGS", "0") == "1"

# Set RL_TRITON_CORRECTNESS_WARNINGS=1 to enable assertions on inputs.
def _CORRECTNESS_WARNINGS() -> bool:
    return os.environ.get("RL_TRITON_CORRECTNESS_WARNINGS", "0") == "1"


def _perf_warn(msg: str) -> None:
    if _PERF_WARNINGS:
        warnings.warn(msg, stacklevel=3)


def _correctness_warn(msg: str) -> None:
    if _CORRECTNESS_WARNINGS():
        warnings.warn(msg, stacklevel=3)

from rl_triton.kernels.scan import backward_scan_kernel, forward_scan_kernel
from rl_triton.kernels.scan_chunked import chunked_backward_scan_kernel

# tl.associative_scan requires BLOCK_SIZE <= 2^17.  Above this the flat kernel
# cannot launch; the chunked kernel handles arbitrary lengths.
_FLAT_MAX_SEQ_LEN = 131072
_CHUNK_SIZE = triton.next_power_of_2(1024)

_WARPS = {512: 4, 1024: 8, 2048: 16, 4096: 16, 8192: 32, 16384: 32}


def _run_scan(
    u: torch.Tensor,
    v: torch.Tensor,
    bootstrap: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Backward linear scan: A[t] = u[t] + v[t] * A[t+1], A[T] = bootstrap.

    Shared implementation for GAE, V-Trace, Retrace, discounted returns, and
    lambda returns — callers compute u and v from their own inputs and delegate
    here.  Dispatches to the flat single-block kernel for seq_len <= 131072 and
    the chunked kernel for longer sequences.

    Non-contiguous inputs are accepted; a contiguous copy is made automatically.
    Set RL_TRITON_PERF_WARNINGS=1 to be warned when this happens.

    Args:
        u:         Additive term [num_envs, seq_len], float32, CUDA.
        v:         Multiplicative decay [num_envs, seq_len], float32, CUDA.
        bootstrap: Per-environment boundary value A[T], shape [num_envs], float32.
                   Defaults to zeros (terminated episodes).

    Returns:
        out: A[t] values, shape [num_envs, seq_len], float32.
    """
    if not u.is_contiguous():
        _perf_warn("u is not contiguous; a copy will be made. Call .contiguous() before the hot loop to avoid this overhead.")
    if not v.is_contiguous():
        _perf_warn("v is not contiguous; a copy will be made. Call .contiguous() before the hot loop to avoid this overhead.")

    if _CORRECTNESS_WARNINGS():
        assert u.shape == v.shape,      "u and v must have the same shape"
        assert u.is_cuda and v.is_cuda, "u and v must be on CUDA"
        assert u.dtype == torch.float32, f"u: expected float32, got {u.dtype}"
        assert v.dtype == torch.float32, f"v: expected float32, got {v.dtype}"
        if bootstrap is not None:
            assert bootstrap.shape == (u.shape[0],), \
                f"bootstrap must have shape [{u.shape[0]}], got {bootstrap.shape}"
            assert bootstrap.is_cuda, "bootstrap must be on CUDA"

    u = u.contiguous()
    v = v.contiguous()

    num_envs, seq_len = u.shape

    has_bootstrap = bootstrap is not None
    if has_bootstrap:
        bootstrap = bootstrap.contiguous()

    out = torch.empty_like(u)

    if seq_len > _FLAT_MAX_SEQ_LEN:
        chunked_backward_scan_kernel[(num_envs,)](
            u, v, out,
            bootstrap,
            seq_len,
            u.stride(0),
            BLOCK_SIZE=_CHUNK_SIZE,
            HAS_BOOTSTRAP=has_bootstrap,
        )
    else:
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        num_warps  = _WARPS.get(BLOCK_SIZE, 16)
        num_stages = 2 if BLOCK_SIZE >= 2048 else 1
        backward_scan_kernel[(num_envs,)](
            u, v, out,
            bootstrap,
            seq_len,
            u.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
            HAS_BOOTSTRAP=has_bootstrap,
        )

    return out


def _run_scan_forward(
    u: torch.Tensor,
    v: torch.Tensor,
    seed: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Forward linear scan: e[t] = u[t] + v[t] * e[t-1], e[-1] = seed.

    Processes the sequence left-to-right (natural time order).  Only the flat
    single-block kernel is supported; sequences longer than _FLAT_MAX_SEQ_LEN
    are rejected because the chunked kernel for forward scans has not been
    implemented yet (see NOTES.md).

    Args:
        u:    Additive term [num_envs, seq_len], float32, CUDA.
        v:    Multiplicative decay [num_envs, seq_len], float32, CUDA.
        seed: Initial carry e[-1] per environment, shape [num_envs], float32.
              Defaults to zeros (traces start from scratch).

    Returns:
        out: e[t] values, shape [num_envs, seq_len], float32.
    """
    if not u.is_contiguous():
        _perf_warn("u is not contiguous; a copy will be made. Call .contiguous() before the hot loop to avoid this overhead.")
    if not v.is_contiguous():
        _perf_warn("v is not contiguous; a copy will be made. Call .contiguous() before the hot loop to avoid this overhead.")

    if _CORRECTNESS_WARNINGS():
        assert u.shape == v.shape,      "u and v must have the same shape"
        assert u.is_cuda and v.is_cuda, "u and v must be on CUDA"
        assert u.dtype == torch.float32, f"u: expected float32, got {u.dtype}"
        assert v.dtype == torch.float32, f"v: expected float32, got {v.dtype}"
        if seed is not None:
            assert seed.shape == (u.shape[0],), \
                f"seed must have shape [{u.shape[0]}], got {seed.shape}"
            assert seed.is_cuda, "seed must be on CUDA"

    u = u.contiguous()
    v = v.contiguous()

    num_envs, seq_len = u.shape

    assert seq_len <= _FLAT_MAX_SEQ_LEN, (
        f"seq_len={seq_len} exceeds the flat kernel limit {_FLAT_MAX_SEQ_LEN}. "
        "A chunked forward scan kernel has not been implemented yet."
    )

    has_seed = seed is not None
    if has_seed:
        seed = seed.contiguous()

    out        = torch.empty_like(u)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    num_warps  = _WARPS.get(BLOCK_SIZE, 16)
    num_stages = 2 if BLOCK_SIZE >= 2048 else 1

    forward_scan_kernel[(num_envs,)](
        u, v, out,
        seed,
        seq_len,
        u.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
        HAS_SEED=has_seed,
    )
    return out
