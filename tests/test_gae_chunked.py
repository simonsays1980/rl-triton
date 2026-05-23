import time

import pytest
import torch

triton = pytest.importorskip("triton")

from rl_triton.ops.gae import compute_gae_triton, _FLAT_MAX_SEQ_LEN
from rl_triton.ops.gae_chunked import compute_gae_chunked

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def reference_gae(deltas: torch.Tensor, decays: torch.Tensor) -> torch.Tensor:
    T = deltas.shape[1]
    adv = torch.zeros_like(deltas)
    gae = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    for t in reversed(range(T)):
        gae = deltas[:, t] + decays[:, t] * gae
        adv[:, t] = gae
    return adv


# ---------------------------------------------------------------------------
# Correctness — chunked kernel directly
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (1,   1),
    (1,   7),      # shorter than one chunk
    (4,   1024),   # exactly one chunk
    (4,   1025),   # one element spills into a second chunk
    (8,   2048),   # exactly two chunks
    (32,  3000),   # non-multiple of chunk size across several chunks
    (64,  4096),   # four chunks
])
def test_chunked_correctness(num_envs, seq_len):
    torch.manual_seed(0)
    deltas = torch.randn(num_envs, seq_len, device="cuda", dtype=torch.float32)
    decays = torch.rand(num_envs, seq_len, device="cuda", dtype=torch.float32) * 0.99

    expected = reference_gae(deltas, decays)
    actual = compute_gae_chunked(deltas, decays)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_chunked_zero_decay():
    """With decay=0, each advantage equals its own delta regardless of chunking."""
    torch.manual_seed(1)
    deltas = torch.randn(8, 2048, device="cuda")
    decays = torch.zeros(8, 2048, device="cuda")

    actual = compute_gae_chunked(deltas, decays)
    torch.testing.assert_close(actual, deltas, atol=1e-6, rtol=1e-6)


@cuda_only
def test_chunked_matches_flat():
    """Chunked and flat kernels must agree for seq_len within the flat limit."""
    torch.manual_seed(2)
    deltas = torch.randn(32, 4096, device="cuda")
    decays = torch.rand(32, 4096, device="cuda") * 0.99

    flat = compute_gae_triton(deltas, decays)
    chunked = compute_gae_chunked(deltas, decays)

    torch.testing.assert_close(chunked, flat, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Auto-dispatch — compute_gae_triton routes to chunked above the threshold
# ---------------------------------------------------------------------------

@cuda_only
def test_autodispatch_below_threshold():
    """seq_len <= _FLAT_MAX_SEQ_LEN must use the flat kernel (no regression)."""
    torch.manual_seed(3)
    seq_len = _FLAT_MAX_SEQ_LEN
    deltas = torch.randn(2, seq_len, device="cuda")
    decays = torch.rand(2, seq_len, device="cuda") * 0.99

    expected = reference_gae(deltas, decays)
    actual = compute_gae_triton(deltas, decays)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.slow
def test_autodispatch_above_threshold():
    """seq_len > _FLAT_MAX_SEQ_LEN must route to chunked and produce correct output."""
    torch.manual_seed(4)
    seq_len = _FLAT_MAX_SEQ_LEN + 1
    deltas = torch.randn(2, seq_len, device="cuda")
    decays = torch.rand(2, seq_len, device="cuda") * 0.99

    expected = reference_gae(deltas, decays)
    actual = compute_gae_triton(deltas, decays)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _bench_gpu(fn, *args, n_warmup: int = 25, n_iter: int = 100) -> float:
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
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
    """Scale iteration counts down for very long sequences to bound runtime.

    torch.compile dispatches one CUDA kernel per timestep from Python, so
    n_iter * seq_len Python calls must stay reasonable.  Target ~500k total
    dispatches: at seq_len=524288 that means just 1 warmup + 1 measured call.
    GPU-only benchmarks are unaffected (CUDA events amortise launch overhead).
    """
    target_dispatches = 500_000
    n_warmup = max(1, min(5,  target_dispatches // (2 * seq_len)))
    n_iter   = max(1, min(20, target_dispatches // seq_len))
    return n_warmup, n_iter


# ---------------------------------------------------------------------------
# Performance benchmark — long sequences
# ---------------------------------------------------------------------------

# Configs deliberately span the flat/chunked boundary at _FLAT_MAX_SEQ_LEN=131072.
# num_envs is kept small because very long sequences already saturate the GPU.
LONG_SEQ_CONFIGS = [
    (16,   8_192),    # well inside flat range
    (16,  65_536),    # still flat, approaching limit
    (16, 131_072),    # exactly at the flat limit
    (16, 262_144),    # chunked only (above threshold)
    (16, 524_288),    # chunked, 4x threshold
]


@cuda_only
@pytest.mark.slow
def test_chunked_performance():
    """
    Benchmark chunked vs torch.compile across long seq_len configs, including
    sequences that exceed the flat kernel limit and trigger auto-dispatch.

    Reports the crossover point where chunked becomes the only viable option
    and how it compares to the best PyTorch alternative at each scale.
    """
    compiled_gae = torch.compile(reference_gae)
    _d = torch.randn(16, 8192, device="cuda")
    _c = torch.rand(16, 8192, device="cuda") * 0.99
    compiled_gae(_d, _c)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>10} "
        f"{'chunked':>10} {'flat/auto':>10} {'compiled':>10} "
        f"{'vs compile':>12} {'kernel':>10}"
    )
    print(header)
    print("-" * len(header))

    for num_envs, seq_len in LONG_SEQ_CONFIGS:
        d = torch.randn(num_envs, seq_len, device="cuda")
        c = torch.rand(num_envs, seq_len, device="cuda") * 0.99

        n_warmup, n_iter = _n_iter_for_seq_len(seq_len)

        chunked_ms  = _bench_gpu(compute_gae_chunked, d, c)
        # compute_gae_triton auto-dispatches — flat for <=131072, chunked above.
        auto_ms     = _bench_gpu(compute_gae_triton, d, c)
        compiled_ms = _bench_cpu(compiled_gae, d, c, n_warmup=n_warmup, n_iter=n_iter)

        which = "flat" if seq_len <= _FLAT_MAX_SEQ_LEN else "chunked"
        speedup = compiled_ms / chunked_ms

        print(
            f"{num_envs:>10} {seq_len:>10,} "
            f"{chunked_ms:>9.3f}ms {auto_ms:>9.3f}ms "
            f"{compiled_ms:>9.3f}ms (n={n_iter}) "
            f"{speedup:>10.1f}x  {which:>10}"
        )
