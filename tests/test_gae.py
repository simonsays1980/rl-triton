import time

# numpy is used intentionally: RLlib v2's compute_value_targets runs on CPU with NumPy,
# so rllib_gae mirrors that faithfully rather than porting it to PyTorch.
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


def rllib_gae_triton(deltas_np: np.ndarray, decays_np: np.ndarray) -> np.ndarray:
    """
    Realistic RLlib-user adoption path: NumPy arrays in, NumPy arrays out.

    Mirrors what an RLlib user would do to swap in the Triton kernel without
    changing the surrounding pipeline — episode data stays as NumPy throughout,
    conversion to/from GPU tensor is part of the measured cost.
    """
    deltas_gpu = torch.from_numpy(deltas_np).to(device="cuda", dtype=torch.float32)
    decays_gpu = torch.from_numpy(decays_np).to(device="cuda", dtype=torch.float32)
    result = compute_gae_triton(deltas_gpu, decays_gpu)
    torch.cuda.synchronize()
    return result.cpu().numpy()


def rllib_gae(deltas: np.ndarray, decays: np.ndarray) -> np.ndarray:
    """
    RLlib v2 compute_value_targets adapted to our (deltas, decays) interface.

    Accepts NumPy arrays, mirroring what RLlib passes from SingleAgentEpisode.
    Loops over environments, which is exactly what RLlib does in practice
    (one episode at a time), and runs the backward scan in pure NumPy on CPU.
    """
    num_envs, seq_len = deltas.shape
    results = np.empty((num_envs, seq_len), dtype=np.float32)

    for i in range(num_envs):
        d = deltas[i]
        c = decays[i]

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
    Sweep over (num_envs, seq_len) configs comparing four implementations:

      triton(gpu)  — tensors already on GPU, no transfer cost (best case).
      compiled     — torch.compile on the GPU loop (strongest PyTorch baseline).
      np→triton→np — NumPy in, Triton kernel, NumPy out; the realistic adoption
                     path for an RLlib user whose episode data lives in NumPy.
      rllib(cpu)   — pure NumPy backward loop on CPU, what RLlib ships today.

    Assertion: Triton must be >=1.5x faster than torch.compile (GPU vs GPU).
    The np→triton→np column answers whether transfer overhead erases the gain.
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
        f"{'triton(gpu)':>12} {'compiled':>10} {'np→triton→np':>14} {'rllib(cpu)':>12} "
        f"{'vs compile':>12} {'vs np→tri→np':>14} {'vs rllib':>10}"
    )
    print(header)
    print("-" * len(header))

    all_speedups_compile = []
    all_speedups_e2e = []

    for num_envs, seq_len in BENCH_CONFIGS:
        deltas_gpu = torch.randn(num_envs, seq_len, device="cuda")
        decays_gpu = torch.rand(num_envs, seq_len, device="cuda") * 0.99
        # NumPy copies — what RLlib hands to the GAE function from SingleAgentEpisode.
        deltas_np = deltas_gpu.cpu().numpy()
        decays_np = decays_gpu.cpu().numpy()

        triton_ms   = _bench_gpu(compute_gae_triton, deltas_gpu, decays_gpu)
        # torch.compile still runs a sequential Python loop per timestep, so the
        # CPU dispatch overhead is real — wall-clock time captures it, CUDA events
        # would only measure the GPU portion and undercount the true cost.
        compiled_ms = _bench_cpu(compiled_gae, deltas_gpu, decays_gpu)
        e2e_ms      = _bench_cpu(rllib_gae_triton, deltas_np, decays_np)
        rllib_ms    = _bench_cpu(rllib_gae, deltas_np, decays_np)

        speedup_vs_compile = compiled_ms / triton_ms
        speedup_vs_e2e     = e2e_ms / triton_ms
        speedup_vs_rllib   = rllib_ms / triton_ms
        all_speedups_compile.append(speedup_vs_compile)
        all_speedups_e2e.append(rllib_ms / e2e_ms)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{triton_ms:>11.3f}ms {compiled_ms:>9.3f}ms "
            f"{e2e_ms:>13.3f}ms {rllib_ms:>11.3f}ms "
            f"{speedup_vs_compile:>10.1f}x  {speedup_vs_e2e:>12.1f}x  {speedup_vs_rllib:>8.1f}x"
        )

    print(
        "\nTriton(gpu)  : tensors already on GPU — best case, no transfer cost."
        "\nnp→triton→np : NumPy in, NumPy out — realistic RLlib adoption path."
        "\nrllib(cpu)   : pure NumPy backward loop on CPU — what RLlib ships today."
        "\ncompiled     : torch.compile on the GPU loop — strongest PyTorch baseline."
    )

    min_speedup = min(all_speedups_compile)
    assert min_speedup >= 1.5, (
        f"Expected >=1.5x speedup over torch.compile across all configs, "
        f"worst was {min_speedup:.2f}x"
    )
