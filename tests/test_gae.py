import time

import numpy as np
import pytest
import torch

triton = pytest.importorskip("triton")

from rl_triton.ops.gae import compute_gae_triton

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

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


def rllib_gae(deltas: torch.Tensor, decays: torch.Tensor) -> np.ndarray:
    """
    RLlib v2 compute_value_targets adapted to our (deltas, decays) interface.

    RLlib operates on a flat [T] sequence per environment and uses NumPy on CPU.
    To match our batched [num_envs, seq_len] interface we loop over environments,
    which is exactly what RLlib does in practice (one episode at a time).

    Mapping from our inputs:
      delta[t]  = r[t] + gamma * V[t+1] - V[t]          (pre-computed TD error)
      decay[t]  = gamma * lambda * (1 - terminated[t])

    We factor out gamma*lambda from decay to recover the RLlib inputs:
      gamma * lambda  = decay.max()  (constant across non-terminal steps)
      terminated[t]   = 1 if decay[t] == 0 else 0

    Because we only have deltas (not raw rewards + vf_preds separately), we pass
    the delta directly as the "intermediate" value and set gamma*(1-lambda)=0,
    which collapses compute_value_targets to the plain GAE backward scan.
    No truncation resets are applied (all episodes are treated as terminated).
    """
    num_envs, seq_len = deltas.shape
    deltas_np = deltas.cpu().numpy()
    decays_np = decays.cpu().numpy()
    results = np.empty((num_envs, seq_len), dtype=np.float32)

    for i in range(num_envs):
        d = deltas_np[i]   # [seq_len]
        c = decays_np[i]   # [seq_len]

        # Recover terminated mask: wherever decay==0, the episode ended.
        terminateds = (c == 0.0).astype(np.float32)
        continues = 1.0 - terminateds

        # Plain backward scan equivalent to compute_value_targets with lambda_=1.
        # intermediates = delta  (no gamma*(1-lambda)*V term since we work with deltas)
        adv = np.empty(seq_len, dtype=np.float32)
        last = 0.0
        for t in reversed(range(seq_len)):
            last = d[t] + continues[t] * c[t] * last
            adv[t] = last

        results[i] = adv

    return results


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
# Benchmark helpers
# ---------------------------------------------------------------------------

def _bench_gpu(fn, *args, n_warmup: int = 25, n_iter: int = 100) -> float:
    """Returns mean milliseconds per call, measured with CUDA events."""
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
    """Returns mean milliseconds per call, measured with time.perf_counter."""
    for _ in range(n_warmup):
        fn(*args)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(*args)
    return (time.perf_counter() - t0) / n_iter * 1000.0


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

BENCH_CONFIGS = [
    (64,  512),
    (128, 1024),
    (256, 1024),
    (512, 2048),
    (512, 4096),
]


@cuda_only
@pytest.mark.slow
def test_gae_performance():
    """
    Sweep over (num_envs, seq_len) configs comparing:
      - Triton kernel           (GPU, fair baseline for the assertion)
      - torch.compile loop      (GPU, strongest realistic PyTorch baseline)
      - RLlib v2 compute_value_targets  (CPU/NumPy, labelled separately)
      - Raw Python loop         (GPU, context only — not asserted against)

    The speedup assertion is Triton vs torch.compile only.
    RLlib is CPU-bound and shown purely for context of what practitioners
    typically run today.
    """
    # torch.compile sees the loop structure and fuses the sequential CUDA ops —
    # this is the strongest realistic PyTorch baseline without a custom kernel.
    # Defined here (not module-level) to avoid triggering JIT compilation on
    # every test session, even when this test is not selected.
    compiled_gae = torch.compile(reference_gae)

    # Trigger torch.compile tracing once outside the timed region.
    _d = torch.randn(64, 512, device="cuda")
    _c = torch.rand(64, 512, device="cuda") * 0.99
    compiled_gae(_d, _c)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>10} {'compiled':>10} {'rllib(cpu)':>12} {'py-loop':>10} "
        f"{'vs compile':>12} {'vs rllib':>10}"
    )
    print(header)
    print("-" * len(header))

    all_speedups = []

    for num_envs, seq_len in BENCH_CONFIGS:
        deltas = torch.randn(num_envs, seq_len, device="cuda")
        decays = torch.rand(num_envs, seq_len, device="cuda") * 0.99

        triton_ms   = _bench_gpu(compute_gae_triton, deltas, decays)
        compiled_ms = _bench_gpu(compiled_gae, deltas, decays)
        rllib_ms    = _bench_cpu(rllib_gae, deltas, decays)
        loop_ms     = _bench_gpu(reference_gae, deltas, decays, n_warmup=3, n_iter=10)

        speedup_vs_compile = compiled_ms / triton_ms
        speedup_vs_rllib   = rllib_ms / triton_ms
        all_speedups.append(speedup_vs_compile)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{triton_ms:>9.3f}ms {compiled_ms:>9.3f}ms "
            f"{rllib_ms:>10.3f}ms {loop_ms:>9.3f}ms "
            f"{speedup_vs_compile:>10.1f}x  {speedup_vs_rllib:>8.1f}x"
        )

    print(
        f"\nNote: rllib column is CPU/NumPy — different device, shown for context only."
    )

    min_speedup = min(all_speedups)
    assert min_speedup >= 1.5, (
        f"Expected >=1.5x speedup over torch.compile across all configs, "
        f"worst was {min_speedup:.2f}x"
    )
