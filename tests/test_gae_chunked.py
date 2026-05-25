import pytest
import torch

triton = pytest.importorskip("triton")

from rl_triton.ops.gae import compute_gae_triton, _FLAT_MAX_SEQ_LEN
from rl_triton.ops.gae_chunked import compute_gae_chunked

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def reference_gae(
    deltas: torch.Tensor,
    decays: torch.Tensor,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    T = deltas.shape[1]
    adv = torch.zeros_like(deltas)
    gae = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    if bootstrap_values is not None:
        gae = bootstrap_values.clone()
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
@pytest.mark.parametrize("num_envs,seq_len", [
    (4,   1024),
    (8,   2048),
    (32,  3000),
])
def test_chunked_bootstrap_truncated(num_envs, seq_len):
    """Truncated episodes: bootstrap propagates V(s_T) across all chunk boundaries."""
    torch.manual_seed(5)
    deltas    = torch.randn(num_envs, seq_len, device="cuda")
    decays    = torch.rand(num_envs, seq_len, device="cuda") * 0.99
    bootstrap = torch.rand(num_envs, device="cuda") * 2.0

    expected = reference_gae(deltas, decays, bootstrap_values=bootstrap)
    actual   = compute_gae_chunked(deltas, decays, bootstrap_values=bootstrap)

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

def _bench_gpu(fn, *args, n_warmup: int, n_iter: int) -> float:
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


def _n_iters(seq_len: int) -> tuple[int, int]:
    """Scale warmup/measurement iterations so benchmark stays under ~30 s total."""
    # Target: ~200 kernel calls measured, scaled down for very long sequences.
    n_warmup = max(2, min(10,  200_000 // seq_len))
    n_iter   = max(2, min(50, 1_000_000 // seq_len))
    return n_warmup, n_iter


# ---------------------------------------------------------------------------
# Performance benchmark — long sequences
# ---------------------------------------------------------------------------

# Configs span the flat/chunked boundary at _FLAT_MAX_SEQ_LEN=131072.
LONG_SEQ_CONFIGS = [
    (16,   8_192),    # well inside flat range
    (16,  65_536),    # still flat, approaching limit
    (16, 131_072),    # exactly at the flat limit
    (16, 262_144),    # above threshold — auto-dispatch uses chunked
    (16, 524_288),    # 4x threshold
]


@cuda_only
@pytest.mark.slow
def test_chunked_performance():
    """
    Benchmark two implementations across long seq_len configs:
      - chunked:  compute_gae_chunked called directly (always chunked)
      - triton:   compute_gae_triton (public API) — routes to flat or chunked

    pt.compile is excluded: a Python loop over T=65536+ steps dispatches that many
    sequential CUDA kernels and takes minutes, so it is not a meaningful comparison
    at this scale. See test_gae_performance for the pt.compile comparison at shorter
    sequences where it is a legitimate baseline.

    'auto used' shows which kernel compute_gae_triton selected.
    """
    header = (
        f"\n{'num_envs':>10} {'seq_len':>10} "
        f"{'chunked':>12} {'triton':>10} {'auto used':>10}"
    )
    print(header)
    print("-" * len(header))

    for num_envs, seq_len in LONG_SEQ_CONFIGS:
        d = torch.randn(num_envs, seq_len, device="cuda")
        c = torch.rand(num_envs, seq_len, device="cuda") * 0.99

        n_warmup, n_iter = _n_iters(seq_len)

        chunked_ms = _bench_gpu(compute_gae_chunked, d, c, n_warmup=n_warmup, n_iter=n_iter)
        triton_ms  = _bench_gpu(compute_gae_triton,  d, c, n_warmup=n_warmup, n_iter=n_iter)

        auto_used = "flat" if seq_len <= _FLAT_MAX_SEQ_LEN else "chunked"

        print(
            f"{num_envs:>10} {seq_len:>10,} "
            f"{chunked_ms:>11.3f}ms {triton_ms:>9.3f}ms "
            f"  {auto_used:>10}"
        )
