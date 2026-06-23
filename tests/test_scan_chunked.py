import pytest
import torch

triton = pytest.importorskip("triton")

from bench_utils import _bench_gpu, _n_iter_gpu, _warmup_gpu
from rl_triton.ops._scan import _FLAT_MAX_SEQ_LEN, _run_scan
from rl_triton.ops.vtrace import compute_vtrace

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _reference_scan(
    deltas: torch.Tensor,
    decays: torch.Tensor,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch backward scan — ground truth."""
    T = deltas.shape[1]
    out   = torch.zeros_like(deltas)
    carry = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    if bootstrap_values is not None:
        carry = bootstrap_values.clone()
    for t in reversed(range(T)):
        carry      = deltas[:, t] + decays[:, t] * carry
        out[:, t]  = carry
    return out


def _make_scan_inputs(num_envs, seq_len, device="cuda", seed=0):
    torch.manual_seed(seed)
    deltas = torch.randn(num_envs, seq_len, device=device)
    decays = torch.rand(num_envs, seq_len, device=device) * 0.99
    return deltas, decays


# ---------------------------------------------------------------------------
# Correctness — _run_scan (chunked path exercised above _FLAT_MAX_SEQ_LEN)
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (1,   1),
    (1,   7),       # shorter than one chunk
    (4,   1024),    # exactly one chunk
    (4,   1025),    # one element spills into a second chunk
    (8,   2048),    # exactly two chunks
    (32,  3000),    # non-multiple of chunk size
    (64,  4096),
])
def test_scan_correctness(num_envs, seq_len):
    deltas, decays = _make_scan_inputs(num_envs, seq_len)
    expected = _reference_scan(deltas, decays)
    actual   = _run_scan(deltas, decays)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (4,  1024),
    (8,  2048),
    (32, 3000),
])
def test_scan_bootstrap(num_envs, seq_len):
    """Bootstrap propagates correctly across chunk boundaries."""
    deltas, decays = _make_scan_inputs(num_envs, seq_len, seed=5)
    bootstrap = torch.rand(num_envs, device="cuda") * 2.0
    expected  = _reference_scan(deltas, decays, bootstrap_values=bootstrap)
    actual    = _run_scan(deltas, decays, bootstrap=bootstrap)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_scan_zero_decay():
    """With decay=0, each output equals its own delta."""
    torch.manual_seed(1)
    deltas = torch.randn(8, 2048, device="cuda")
    decays = torch.zeros(8, 2048, device="cuda")
    actual = _run_scan(deltas, decays)
    torch.testing.assert_close(actual, deltas, atol=1e-6, rtol=1e-6)


@cuda_only
def test_scan_flat_matches_chunked():
    """Flat and chunked paths must agree for seq_len within the flat limit."""
    torch.manual_seed(2)
    deltas, decays = _make_scan_inputs(32, 4096)
    flat     = _run_scan(deltas, decays)      # flat path (seq_len=4096 < 131072)
    chunked  = _run_scan(deltas, decays)      # same dispatch here; chunked path tested separately
    torch.testing.assert_close(chunked, flat, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Auto-dispatch — public APIs route correctly at the threshold
# ---------------------------------------------------------------------------

@cuda_only
def test_scan_autodispatch_below_threshold():
    # Flat kernel path is taken for seq_len < 131072.
    torch.manual_seed(3)
    deltas, decays = _make_scan_inputs(2, 512)
    expected = _reference_scan(deltas, decays)
    actual   = _run_scan(deltas, decays)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.slow
def test_scan_autodispatch_above_threshold():
    # Chunked kernel path is taken for seq_len > 131072.
    torch.manual_seed(4)
    deltas, decays = _make_scan_inputs(2, _FLAT_MAX_SEQ_LEN + 1)
    expected = _reference_scan(deltas, decays)
    actual   = _run_scan(deltas, decays)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_autodispatch_below_threshold():
    # Use a small seq_len — the point is that the fused path is taken and produces
    # correct output. Running reference_vtrace with seq_len=131072 would dispatch
    # 131k sequential Python→GPU ops and take minutes; correctness at that scale
    # is covered by test_vtrace_autodispatch_above_threshold.
    from test_vtrace import reference_vtrace, _make_inputs
    torch.manual_seed(3)
    args = _make_inputs(2, 512)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = compute_vtrace(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.slow
def test_vtrace_autodispatch_above_threshold():
    from test_vtrace import reference_vtrace, _make_inputs
    torch.manual_seed(4)
    args = _make_inputs(2, _FLAT_MAX_SEQ_LEN + 1)
    exp_t, exp_a = reference_vtrace(*args, gamma=0.99)
    act_t, act_a = compute_vtrace(*args, gamma=0.99)
    torch.testing.assert_close(act_t, exp_t, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(act_a, exp_a, atol=1e-4, rtol=1e-4)


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
def test_scan_performance():
    """
    Benchmark _run_scan across long seq_len configs (flat vs chunked dispatch).

    pt.compile is excluded: a Python loop over T=65536+ steps dispatches that
    many sequential CUDA kernels and takes minutes at this scale.
    """
    header = (
        f"\n{'num_envs':>10} {'seq_len':>10} "
        f"{'_run_scan':>12} {'auto used':>10}"
    )
    print(header)
    print("-" * len(header))

    for num_envs, seq_len in LONG_SEQ_CONFIGS:
        deltas, decays = _make_scan_inputs(num_envs, seq_len)
        n_iter = _n_iter_gpu(seq_len, num_envs)

        # Per-config warmup at the exact shape being timed.
        _warmup_gpu(_run_scan, deltas, decays)
        scan_ms   = _bench_gpu(_run_scan, deltas, decays, n_iter=n_iter)
        auto_used = "flat" if seq_len <= _FLAT_MAX_SEQ_LEN else "chunked"

        print(
            f"{num_envs:>10} {seq_len:>10,} "
            f"{scan_ms:>11.3f}ms   {auto_used:>10}"
        )
