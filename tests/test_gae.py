# numpy is used intentionally: the CPU baseline mirrors how frameworks like RLlib
# compute GAE — a plain backward loop over NumPy arrays on CPU.
import numpy as np
import pytest
import torch

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu

triton = pytest.importorskip("triton")

from rl_triton.ops.gae import compute_gae_triton

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def reference_gae(
    deltas: torch.Tensor,
    decays: torch.Tensor,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch backward scan — ground truth for correctness tests only.
    Not used as a benchmark: Python loop overhead dominates GPU compute time."""
    T = deltas.shape[1]
    adv = torch.zeros_like(deltas)
    gae = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    if bootstrap_values is not None:
        gae = bootstrap_values.clone()
    for t in reversed(range(T)):
        gae = deltas[:, t] + decays[:, t] * gae
        adv[:, t] = gae
    return adv


def numpy_to_triton_to_numpy(deltas_np: np.ndarray, decays_np: np.ndarray) -> np.ndarray:
    """
    End-to-end adoption path: NumPy arrays in, NumPy arrays out.

    Mirrors what a user would do to swap in the Triton kernel without changing
    the surrounding pipeline — episode data stays as NumPy throughout, and
    conversion to/from GPU tensor is part of the measured cost.
    """
    deltas_gpu = torch.from_numpy(deltas_np).to(device="cuda", dtype=torch.float32)
    decays_gpu = torch.from_numpy(decays_np).to(device="cuda", dtype=torch.float32)
    result = compute_gae_triton(deltas_gpu, decays_gpu)
    torch.cuda.synchronize()
    return result.cpu().numpy()


def numpy_gae(deltas: np.ndarray, decays: np.ndarray) -> np.ndarray:
    """
    Pure NumPy backward scan on CPU — baseline for end-to-end benchmark comparisons.
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
# Ground-truth value tests
# ---------------------------------------------------------------------------
#
# These tests verify the kernel against values computed by hand from the
# recurrence A[t] = delta[t] + decay[t] * A[t+1], A[T] = bootstrap.
# They catch bugs that would be invisible to reference_gae comparisons —
# e.g. a wrong scan direction that both implementations share.

@cuda_only
def test_gae_known_values_single_env():
    # Hand-computed for seq_len=3, terminated (bootstrap=0):
    #   A[2] = 3.0 + 0.7 * 0.0 = 3.0
    #   A[1] = 2.0 + 0.8 * 3.0 = 4.4
    #   A[0] = 1.0 + 0.9 * 4.4 = 4.96
    deltas = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    decays = torch.tensor([[0.9, 0.8, 0.7]], device="cuda")
    expected = torch.tensor([[4.96, 4.4, 3.0]], device="cuda")
    torch.testing.assert_close(compute_gae_triton(deltas, decays), expected, atol=1e-5, rtol=1e-5)


@cuda_only
def test_gae_known_values_truncated():
    # Hand-computed for seq_len=3, truncated (bootstrap=2.0):
    #   A[2] = 3.0 + 0.7 * 2.0 = 4.4
    #   A[1] = 2.0 + 0.8 * 4.4 = 5.52
    #   A[0] = 1.0 + 0.9 * 5.52 = 5.968
    deltas = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    decays = torch.tensor([[0.9, 0.8, 0.7]], device="cuda")
    bootstrap = torch.tensor([2.0], device="cuda")
    expected = torch.tensor([[5.968, 5.52, 4.4]], device="cuda")
    torch.testing.assert_close(
        compute_gae_triton(deltas, decays, bootstrap_values=bootstrap),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_gae_known_values_mixed_termination():
    # Two environments: env 0 terminated (bootstrap=0), env 1 truncated (bootstrap=5.0).
    # Env 0: A[1]=2.0+0.9*0.0=2.0,  A[0]=1.0+0.5*2.0=2.0
    # Env 1: A[1]=2.0+0.9*5.0=6.5,  A[0]=1.0+0.5*6.5=4.25
    deltas    = torch.tensor([[1.0, 2.0], [1.0, 2.0]], device="cuda")
    decays    = torch.tensor([[0.5, 0.9], [0.5, 0.9]], device="cuda")
    bootstrap = torch.tensor([0.0, 5.0], device="cuda")
    expected  = torch.tensor([[2.0, 2.0], [4.25, 6.5]], device="cuda")
    torch.testing.assert_close(
        compute_gae_triton(deltas, decays, bootstrap_values=bootstrap),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_gae_known_values_episode_boundary():
    # decay=0 at t=1 simulates an episode boundary within the trajectory.
    #   A[2] = 3.0 + 0.9 * 0.0 = 3.0
    #   A[1] = 2.0 + 0.0 * 3.0 = 2.0   <- boundary resets carry
    #   A[0] = 1.0 + 0.9 * 2.0 = 2.8
    deltas = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    decays = torch.tensor([[0.9, 0.0, 0.9]], device="cuda")
    expected = torch.tensor([[2.8, 2.0, 3.0]], device="cuda")
    torch.testing.assert_close(compute_gae_triton(deltas, decays), expected, atol=1e-5, rtol=1e-5)


@cuda_only
def test_gae_known_values_batch():
    # Two environments with different values — verifies batch indexing.
    # Env 0: A[1]=2.0, A[0]=1.0+0.5*2.0=2.0
    # Env 1: A[1]=4.0, A[0]=3.0+0.5*4.0=5.0
    deltas = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    decays = torch.tensor([[0.5, 0.9], [0.5, 0.9]], device="cuda")
    expected = torch.tensor([[2.0, 2.0], [5.0, 4.0]], device="cuda")
    torch.testing.assert_close(compute_gae_triton(deltas, decays), expected, atol=1e-5, rtol=1e-5)


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
      np→triton→np — NumPy in, Triton kernel, NumPy out; realistic adoption path
                     when episode data lives in NumPy.
      numpy(cpu)   — pure NumPy backward loop on CPU.

    Assertions:
      - Triton must be >=1.5x faster than torch.compile (GPU vs GPU).
      - np→triton→np must be >=1.5x faster than numpy(cpu) — below this the
        kernel gain would likely be lost to inter-worker communication overhead
        in a distributed setup.
    """
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
        f"{'triton(gpu)':>12} {'compiled':>10} {'np->triton->np':>16} {'numpy(cpu)':>12} "
        f"{'vs compile':>12} {'np->tri->np vs numpy':>22} {'vs numpy':>10}"
    )
    print(header)
    print("-" * len(header))

    all_speedups_compile = []
    all_speedups_e2e = []

    for num_envs, seq_len in BENCH_CONFIGS:
        deltas_gpu = torch.randn(num_envs, seq_len, device="cuda")
        decays_gpu = torch.rand(num_envs, seq_len, device="cuda") * 0.99
        deltas_np = deltas_gpu.cpu().numpy()
        decays_np = decays_gpu.cpu().numpy()

        gpu_warmup, gpu_iter = _n_iter_gpu(seq_len, num_envs)

        triton_ms   = _bench_gpu(compute_gae_triton,  deltas_gpu, decays_gpu, n_warmup=gpu_warmup, n_iter=gpu_iter)
        compiled_ms = _bench_cpu(compiled_gae,        deltas_gpu, decays_gpu)
        e2e_ms      = _bench_cpu(numpy_to_triton_to_numpy, deltas_np,  decays_np)
        numpy_ms    = _bench_cpu(numpy_gae,               deltas_np,  decays_np)

        speedup_vs_compile   = compiled_ms / triton_ms
        speedup_e2e_vs_numpy = numpy_ms / e2e_ms
        speedup_vs_numpy     = numpy_ms / triton_ms
        all_speedups_compile.append(speedup_vs_compile)
        all_speedups_e2e.append(speedup_e2e_vs_numpy)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{triton_ms:>11.3f}ms {compiled_ms:>9.3f}ms "
            f"{e2e_ms:>15.3f}ms {numpy_ms:>11.3f}ms "
            f"{speedup_vs_compile:>10.1f}x  {speedup_e2e_vs_numpy:>20.1f}x  {speedup_vs_numpy:>8.1f}x"
        )

    print(
        "\ntriton(gpu)   : CUDA events — pure kernel time, no CPU overhead."
        "\ncompiled      : wall-clock — torch.compile dispatches one CUDA op per timestep from Python;"
        "\n                CUDA events would miss that CPU stall and make it look unrealistically fast."
        "\nnp->triton->np: wall-clock — NumPy->GPU->NumPy, realistic adoption path when episode data lives in NumPy."
        "\nnumpy(cpu)    : wall-clock — pure NumPy backward loop on CPU."
    )

    min_speedup = min(all_speedups_compile)
    assert min_speedup >= 1.5, (
        f"Expected >=1.5x speedup over torch.compile across all configs, "
        f"worst was {min_speedup:.2f}x"
    )

    min_e2e_speedup = min(all_speedups_e2e)
    assert min_e2e_speedup >= 1.5, (
        f"Expected >=1.5x end-to-end speedup (np->triton->np vs numpy(cpu)) across all configs — "
        f"below this the kernel gain is likely lost to inter-worker communication overhead in a "
        f"distributed setup. Worst was {min_e2e_speedup:.2f}x"
    )
