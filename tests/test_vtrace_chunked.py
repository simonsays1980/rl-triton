import time

import pytest
import torch

triton = pytest.importorskip("triton")

from rl_triton.ops.vtrace import compute_vtrace_triton, _FLAT_MAX_SEQ_LEN
from rl_triton.ops.vtrace_chunked import compute_vtrace_chunked

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _reference_scan(
    deltas: torch.Tensor,
    decays: torch.Tensor,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch backward scan on pre-computed (deltas, decays) — ground truth."""
    T = deltas.shape[1]
    out = torch.zeros_like(deltas)
    carry = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()
    for t in reversed(range(T)):
        carry = deltas[:, t] + decays[:, t] * carry
        out[:, t] = carry
    return out


def _make_scan_inputs(num_envs, seq_len, device="cuda", seed=0):
    torch.manual_seed(seed)
    deltas = torch.randn(num_envs, seq_len, device=device)
    decays = torch.rand(num_envs, seq_len, device=device) * 0.99
    return deltas, decays


# ---------------------------------------------------------------------------
# Correctness — chunked kernel directly
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (1,   1),
    (1,   7),
    (4,   1024),   # exactly one chunk
    (4,   1025),   # one element spills into a second chunk
    (8,   2048),   # exactly two chunks
    (32,  3000),   # non-multiple of chunk size
    (64,  4096),
])
def test_vtrace_chunked_correctness(num_envs, seq_len):
    deltas, decays = _make_scan_inputs(num_envs, seq_len)
    expected = _reference_scan(deltas, decays)
    actual   = compute_vtrace_chunked(deltas, decays)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (4,  1024),
    (8,  2048),
    (32, 3000),
])
def test_vtrace_chunked_bootstrap(num_envs, seq_len):
    """Bootstrap propagates correctly across chunk boundaries."""
    deltas, decays = _make_scan_inputs(num_envs, seq_len, seed=5)
    bootstrap = torch.rand(num_envs, device="cuda") * 2.0

    expected = _reference_scan(deltas, decays, bootstrap_values=bootstrap)
    actual   = compute_vtrace_chunked(deltas, decays, bootstrap_values=bootstrap)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_chunked_zero_decay():
    """With decay=0, each output equals its own delta."""
    torch.manual_seed(1)
    deltas = torch.randn(8, 2048, device="cuda")
    decays = torch.zeros(8, 2048, device="cuda")
    actual = compute_vtrace_chunked(deltas, decays)
    torch.testing.assert_close(actual, deltas, atol=1e-6, rtol=1e-6)


@cuda_only
def test_vtrace_chunked_matches_flat():
    """Chunked and flat scan kernels must agree within the flat limit."""
    torch.manual_seed(2)
    deltas, decays = _make_scan_inputs(32, 4096, seed=2)

    from rl_triton.kernels.vtrace import vtrace_scan_kernel
    import triton as _triton

    bootstrap = torch.zeros(32, device="cuda")
    flat_out  = torch.empty_like(deltas)
    BLOCK_SIZE = _triton.next_power_of_2(4096)
    vtrace_scan_kernel[(32,)](
        deltas.contiguous(), decays.contiguous(), flat_out,
        bootstrap,
        4096, deltas.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    chunked_out = compute_vtrace_chunked(deltas, decays)
    torch.testing.assert_close(chunked_out, flat_out, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

@cuda_only
def test_vtrace_autodispatch_below_threshold():
    from test_vtrace import reference_vtrace, _make_inputs
    torch.manual_seed(3)
    args = _make_inputs(2, _FLAT_MAX_SEQ_LEN)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = compute_vtrace_triton(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.slow
def test_vtrace_autodispatch_above_threshold():
    from test_vtrace import reference_vtrace, _make_inputs
    torch.manual_seed(4)
    args = _make_inputs(2, _FLAT_MAX_SEQ_LEN + 1)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = compute_vtrace_triton(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _bench_gpu(fn, *args, n_warmup: int = 25, n_iter: int = 100) -> float:
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def _bench_cpu(fn, *args, n_warmup: int = 5, n_iter: int = 20) -> float:
    for _ in range(n_warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    return (time.perf_counter() - t0) / n_iter * 1000.0


def _n_iter_for_seq_len(seq_len: int) -> tuple[int, int]:
    target = 500_000
    n_warmup = max(1, min(5,  target // (2 * seq_len)))
    n_iter   = max(1, min(20, target // seq_len))
    return n_warmup, n_iter


# ---------------------------------------------------------------------------
# Performance benchmark — long sequences
# ---------------------------------------------------------------------------

LONG_SEQ_CONFIGS = [
    (16,   8_192),
    (16,  65_536),
    (16, 131_072),
    (16, 262_144),
    (16, 524_288),
]


@cuda_only
@pytest.mark.slow
def test_vtrace_chunked_performance():
    """
    Benchmark chunked vs torch.compile across long seq_len configs, including
    sequences that exceed the flat kernel limit and trigger auto-dispatch.
    """
    from test_vtrace import reference_vtrace, _make_inputs

    compiled_vtrace = torch.compile(reference_vtrace)
    _args = _make_inputs(16, 8192)
    compiled_vtrace(*_args, gamma=0.99)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>10} "
        f"{'chunked':>10} {'flat/auto':>10} {'compiled':>10} "
        f"{'vs compile':>12} {'kernel':>10}"
    )
    print(header)
    print("-" * len(header))

    for num_envs, seq_len in LONG_SEQ_CONFIGS:
        args = _make_inputs(num_envs, seq_len)
        deltas = torch.randn(num_envs, seq_len, device="cuda")
        decays = torch.rand(num_envs, seq_len, device="cuda") * 0.99

        n_warmup, n_iter = _n_iter_for_seq_len(seq_len)

        chunked_ms  = _bench_gpu(compute_vtrace_chunked, deltas, decays)
        auto_ms     = _bench_gpu(compute_vtrace_triton, *args, gamma=0.99)
        compiled_ms = _bench_cpu(compiled_vtrace, *args, gamma=0.99, n_warmup=n_warmup, n_iter=n_iter)

        which   = "flat" if seq_len <= _FLAT_MAX_SEQ_LEN else "chunked"
        speedup = compiled_ms / chunked_ms

        print(
            f"{num_envs:>10} {seq_len:>10,} "
            f"{chunked_ms:>9.3f}ms {auto_ms:>9.3f}ms "
            f"{compiled_ms:>9.3f}ms (n={n_iter}) "
            f"{speedup:>10.1f}x  {which:>10}"
        )
