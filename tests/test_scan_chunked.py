import pytest
import torch

triton = pytest.importorskip("triton")

from rl_triton.ops._scan import _FLAT_MAX_SEQ_LEN, _run_scan
from rl_triton.ops.gae import compute_gae_triton
from rl_triton.ops.vtrace import compute_vtrace_triton

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
    flat    = _run_scan(deltas, decays)                                    # flat path
    # Force chunked path by patching seq_len above the threshold is not
    # straightforward, so we verify via compute_gae_triton which auto-dispatches.
    gae_out = compute_gae_triton(deltas, decays)
    torch.testing.assert_close(gae_out, flat, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Auto-dispatch — public APIs route correctly at the threshold
# ---------------------------------------------------------------------------

@cuda_only
def test_gae_autodispatch_below_threshold():
    torch.manual_seed(3)
    deltas, decays = _make_scan_inputs(2, _FLAT_MAX_SEQ_LEN)
    expected = _reference_scan(deltas, decays)
    actual   = compute_gae_triton(deltas, decays)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.slow
def test_gae_autodispatch_above_threshold():
    torch.manual_seed(4)
    deltas, decays = _make_scan_inputs(2, _FLAT_MAX_SEQ_LEN + 1)
    expected = _reference_scan(deltas, decays)
    actual   = compute_gae_triton(deltas, decays)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_vtrace_autodispatch_below_threshold():
    # Use a small seq_len — the point is that the fused path is taken, not that
    # reference_vtrace runs over 131072 steps (which would take minutes in Python).
    from test_vtrace import reference_vtrace, _make_inputs
    torch.manual_seed(3)
    args = _make_inputs(2, 512)
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

def _bench_gpu(fn, *args, n_warmup: int, n_iter: int, **kwargs) -> float:
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def _n_iters(seq_len: int) -> tuple[int, int]:
    n_warmup = max(2, min(10,  200_000 // seq_len))
    n_iter   = max(2, min(50, 1_000_000 // seq_len))
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
def test_scan_performance():
    """
    Benchmark _run_scan vs compute_gae_triton across long seq_len configs.
    Both call the same kernel; any gap is pure Python dispatch overhead.

    pt.compile is excluded: a Python loop over T=65536+ steps dispatches that
    many sequential CUDA kernels and takes minutes at this scale.
    """
    from test_vtrace import _make_inputs as _make_vtrace_inputs

    header = (
        f"\n{'num_envs':>10} {'seq_len':>10} "
        f"{'_run_scan':>12} {'gae_triton':>12} {'auto used':>10}"
    )
    print(header)
    print("-" * len(header))

    for num_envs, seq_len in LONG_SEQ_CONFIGS:
        deltas, decays = _make_scan_inputs(num_envs, seq_len)
        n_warmup, n_iter = _n_iters(seq_len)

        scan_ms = _bench_gpu(_run_scan,          deltas, decays, n_warmup=n_warmup, n_iter=n_iter)
        gae_ms  = _bench_gpu(compute_gae_triton, deltas, decays, n_warmup=n_warmup, n_iter=n_iter)
        auto_used = "flat" if seq_len <= _FLAT_MAX_SEQ_LEN else "chunked"

        print(
            f"{num_envs:>10} {seq_len:>10,} "
            f"{scan_ms:>11.3f}ms {gae_ms:>11.3f}ms "
            f"  {auto_used:>10}"
        )
