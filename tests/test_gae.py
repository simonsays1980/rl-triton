import pytest
import torch

triton = pytest.importorskip("triton")

from rl_triton.ops.gae import compute_gae_triton

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def reference_gae(deltas: torch.Tensor, decays: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch backward scan — ground truth for correctness tests only.
    Not used as a benchmark: Python loop overhead dominates GPU compute time."""
    T = deltas.shape[1]
    adv = torch.zeros_like(deltas)
    gae = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    for t in reversed(range(T)):
        gae = deltas[:, t] + decays[:, t] * gae
        adv[:, t] = gae
    return adv


# torch.compile sees the loop structure and fuses the sequential CUDA ops —
# this is the strongest realistic PyTorch baseline without a custom kernel.
compiled_gae = torch.compile(reference_gae)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

@cuda_only
def test_gae_correctness_basic():
    torch.manual_seed(0)
    deltas = torch.randn(64, 512, device="cuda", dtype=torch.float32)
    decays = torch.rand(64, 512, device="cuda", dtype=torch.float32) * 0.99

    expected = reference_gae(deltas, decays)
    actual = compute_gae_triton(deltas, decays)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (1, 1),
    (1, 7),        # non-power-of-2
    (4, 128),
    (32, 333),     # non-power-of-2
    (128, 1024),
    (256, 2048),
])
def test_gae_correctness_shapes(num_envs, seq_len):
    torch.manual_seed(42)
    deltas = torch.randn(num_envs, seq_len, device="cuda", dtype=torch.float32)
    decays = torch.rand(num_envs, seq_len, device="cuda", dtype=torch.float32) * 0.99

    expected = reference_gae(deltas, decays)
    actual = compute_gae_triton(deltas, decays)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_gae_zero_decay():
    """With decay=0 the advantage at each step equals its own delta."""
    torch.manual_seed(1)
    deltas = torch.randn(8, 64, device="cuda")
    decays = torch.zeros(8, 64, device="cuda")

    actual = compute_gae_triton(deltas, decays)
    torch.testing.assert_close(actual, deltas, atol=1e-6, rtol=1e-6)


@cuda_only
def test_gae_non_contiguous_input():
    """Wrapper must handle non-contiguous inputs via .contiguous()."""
    torch.manual_seed(2)
    base = torch.randn(64, 512, 2, device="cuda")
    deltas = base[..., 0]   # non-contiguous view
    decays = (torch.rand(64, 512, 2, device="cuda") * 0.99)[..., 0]

    expected = reference_gae(deltas.contiguous(), decays.contiguous())
    actual = compute_gae_triton(deltas, decays)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench(fn, *args, n_warmup: int = 25, n_iter: int = 100) -> float:
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


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

@cuda_only
def test_gae_performance():
    """
    Triton kernel vs torch.compile(reference_gae) — the strongest realistic
    PyTorch baseline. The raw Python loop is shown for context only and is not
    used in the speedup assertion.
    """
    num_envs, seq_len = 512, 2048

    deltas = torch.randn(num_envs, seq_len, device="cuda")
    decays = torch.rand(num_envs, seq_len, device="cuda") * 0.99

    # Trigger torch.compile's first-run tracing outside the timed region.
    compiled_gae(deltas, decays)
    torch.cuda.synchronize()

    triton_ms  = _bench(compute_gae_triton, deltas, decays)
    compiled_ms = _bench(compiled_gae, deltas, decays)
    loop_ms    = _bench(reference_gae, deltas, decays, n_warmup=5, n_iter=20)

    speedup = compiled_ms / triton_ms

    print(f"\nGAE benchmark — num_envs={num_envs}, seq_len={seq_len}")
    print(f"  Triton             : {triton_ms:.3f} ms/iter")
    print(f"  torch.compile loop : {compiled_ms:.3f} ms/iter  (fair baseline)")
    print(f"  Python loop        : {loop_ms:.3f} ms/iter  (context only)")
    print(f"  Speedup vs compile : {speedup:.1f}x")

    assert speedup >= 1.5, (
        f"Expected >=1.5x speedup over torch.compile baseline, got {speedup:.2f}x "
        f"(Triton {triton_ms:.3f} ms vs compiled {compiled_ms:.3f} ms)"
    )
