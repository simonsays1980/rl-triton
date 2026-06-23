import pytest
import torch

triton = pytest.importorskip("triton")

from bench_utils import _bench_cpu, _bench_gpu, _n_iter_gpu
from rl_triton.ops.prefix_sum import compute_episodic_prefix_sum

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def reference_episodic_prefix_sum(
    inputs: torch.Tensor,
    dones: torch.Tensor,
    seed_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch forward scan — ground truth for correctness tests only."""
    num_envs, seq_len = inputs.shape
    out  = torch.zeros_like(inputs)
    carry = torch.zeros(num_envs, device=inputs.device, dtype=inputs.dtype)
    if seed_values is not None:
        carry = seed_values.clone()
    for t in range(seq_len):
        carry     = inputs[:, t] + (1.0 - dones[:, t]) * carry
        out[:, t] = carry
    return out


# ---------------------------------------------------------------------------
# Ground-truth value tests
# ---------------------------------------------------------------------------

@cuda_only
def test_prefix_sum_known_values_no_reset():
    # No episode boundaries — plain cumulative sum.
    # inputs=[1,2,3], dones=[0,0,0]
    # C[0]=1, C[1]=1+2=3, C[2]=3+3=6
    inputs = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    dones  = torch.zeros(1, 3, device="cuda")
    expected = torch.tensor([[1.0, 3.0, 6.0]], device="cuda")
    torch.testing.assert_close(
        compute_episodic_prefix_sum(inputs, dones), expected, atol=1e-5, rtol=1e-5
    )


@cuda_only
def test_prefix_sum_known_values_reset_at_start():
    # done=1 at t=0 resets carry immediately: C[0] = inputs[0] + (1-1)*carry = inputs[0].
    # inputs=[5,2,3], dones=[1,0,0]
    # C[0]=5, C[1]=5+2=7, C[2]=7+3=10
    inputs = torch.tensor([[5.0, 2.0, 3.0]], device="cuda")
    dones  = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
    expected = torch.tensor([[5.0, 7.0, 10.0]], device="cuda")
    torch.testing.assert_close(
        compute_episodic_prefix_sum(inputs, dones), expected, atol=1e-5, rtol=1e-5
    )


@cuda_only
def test_prefix_sum_known_values_mid_reset():
    # done=1 at t=2 drops the carry: C[2] = inputs[2] + (1-done[2])*C[1] = 3 + 0*3 = 3.
    # inputs=[1,2,3,4], dones=[0,0,1,0]
    # C[0]=1, C[1]=1+2=3, C[2]=3 (carry zeroed), C[3]=3+4=7
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda")
    dones  = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device="cuda")
    expected = torch.tensor([[1.0, 3.0, 3.0, 7.0]], device="cuda")
    torch.testing.assert_close(
        compute_episodic_prefix_sum(inputs, dones), expected, atol=1e-5, rtol=1e-5
    )


@cuda_only
def test_prefix_sum_known_values_all_resets():
    # Every step is a terminal — each output equals its own input.
    inputs = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    dones  = torch.ones(1, 3, device="cuda")
    torch.testing.assert_close(
        compute_episodic_prefix_sum(inputs, dones), inputs, atol=1e-6, rtol=1e-6
    )


@cuda_only
def test_prefix_sum_known_values_seed():
    # Non-zero seed carries into t=0 unless done=1 at t=0.
    # seed=10, inputs=[1,2], dones=[0,0]
    # C[0]=1+(1-0)*10=11, C[1]=2+(1-0)*11=13
    inputs = torch.tensor([[1.0, 2.0]], device="cuda")
    dones  = torch.zeros(1, 2, device="cuda")
    seed   = torch.tensor([10.0], device="cuda")
    expected = torch.tensor([[11.0, 13.0]], device="cuda")
    torch.testing.assert_close(
        compute_episodic_prefix_sum(inputs, dones, seed_values=seed),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_prefix_sum_known_values_seed_cancelled_by_done():
    # done=1 at t=0 means the seed does not carry over.
    # seed=10, inputs=[1,2], dones=[1,0]
    # C[0]=1+(1-1)*10=1, C[1]=2+(1-0)*1=3
    inputs = torch.tensor([[1.0, 2.0]], device="cuda")
    dones  = torch.tensor([[1.0, 0.0]], device="cuda")
    seed   = torch.tensor([10.0], device="cuda")
    expected = torch.tensor([[1.0, 3.0]], device="cuda")
    torch.testing.assert_close(
        compute_episodic_prefix_sum(inputs, dones, seed_values=seed),
        expected, atol=1e-5, rtol=1e-5,
    )


@cuda_only
def test_prefix_sum_known_values_batch():
    # Two environments with independent accumulations.
    # Env 0: no resets, C=[1,3,6]
    # Env 1: reset at t=1, C=[4,2,5]
    inputs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 2.0, 3.0]], device="cuda")
    dones  = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
    expected = torch.tensor([[1.0, 3.0, 6.0], [4.0, 2.0, 5.0]], device="cuda")
    torch.testing.assert_close(
        compute_episodic_prefix_sum(inputs, dones), expected, atol=1e-5, rtol=1e-5
    )


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@cuda_only
def test_prefix_sum_correctness_basic():
    torch.manual_seed(0)
    num_envs, seq_len = 64, 512
    inputs = torch.randn(num_envs, seq_len, device="cuda")
    dones  = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()

    expected = reference_episodic_prefix_sum(inputs, dones)
    actual   = compute_episodic_prefix_sum(inputs, dones)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [
    (1,   1),
    (1,   7),       # non-power-of-2
    (4,   128),
    (32,  333),     # non-power-of-2
    (128, 1024),
    (256, 2048),
])
def test_prefix_sum_correctness_shapes(num_envs, seq_len):
    torch.manual_seed(42)
    inputs = torch.randn(num_envs, seq_len, device="cuda")
    dones  = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()

    expected = reference_episodic_prefix_sum(inputs, dones)
    actual   = compute_episodic_prefix_sum(inputs, dones)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_prefix_sum_no_dones():
    """With no episode boundaries the result is a standard prefix sum."""
    torch.manual_seed(1)
    inputs = torch.randn(16, 256, device="cuda")
    dones  = torch.zeros(16, 256, device="cuda")

    expected = reference_episodic_prefix_sum(inputs, dones)
    actual   = compute_episodic_prefix_sum(inputs, dones)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_prefix_sum_all_dones():
    """All terminals — output equals input elementwise."""
    torch.manual_seed(2)
    inputs = torch.randn(16, 256, device="cuda")
    dones  = torch.ones(16, 256, device="cuda")

    actual = compute_episodic_prefix_sum(inputs, dones)
    torch.testing.assert_close(actual, inputs, atol=1e-6, rtol=1e-6)


@cuda_only
def test_prefix_sum_with_seed():
    torch.manual_seed(3)
    num_envs, seq_len = 32, 128
    inputs = torch.randn(num_envs, seq_len, device="cuda")
    dones  = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()
    seed   = torch.randn(num_envs, device="cuda")

    expected = reference_episodic_prefix_sum(inputs, dones, seed_values=seed)
    actual   = compute_episodic_prefix_sum(inputs, dones, seed_values=seed)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda_only
def test_prefix_sum_non_contiguous_input():
    """Wrapper must handle non-contiguous inputs via .contiguous()."""
    torch.manual_seed(4)
    base   = torch.randn(64, 512, 2, device="cuda")
    inputs = base[..., 0]
    dones  = (torch.rand(64, 512, 2, device="cuda") < 0.05).float()[..., 0]

    expected = reference_episodic_prefix_sum(inputs.contiguous(), dones.contiguous())
    actual   = compute_episodic_prefix_sum(inputs, dones)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


# Prefix sum: no truncation/bootstrap path.
# compute_prefix_sum is a forward-scan cumulative sum with optional episode-boundary
# resets via the `dones` mask.  There is no concept of a "continuation value from
# the next episode" — the scan simply restarts from 0 (or seed_values) at each
# done step.  Truncation vs termination makes no semantic difference to a cumsum.
# No truncation correctness test or truncation benchmark is added.

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


def vectorized_episodic_prefix_sum(
    inputs: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    """
    Episodic prefix sum via torch.cumsum with a segment-correction trick.

    Used as a strong compiled baseline: a competent PyTorch user would write
    something like this to avoid a Python timestep loop.  Not production-
    hardened for all edge cases — only used for benchmarking.

    The idea: global cumsum minus the cumsum value carried in from the start
    of each episode segment.

      running  = cumsum(inputs)               # global, ignores resets
      boundary = running * dones              # value at each terminal step
      offset   = cumsum(boundary).shift(1)   # value to subtract per segment
      result   = running - offset
    """
    running  = torch.cumsum(inputs, dim=1)
    boundary = running * dones
    offset   = torch.cumsum(boundary, dim=1) - boundary
    return running - offset


@cuda_only
@pytest.mark.slow
def test_prefix_sum_performance():
    """
    Benchmark compute_episodic_prefix_sum against three baselines:

      triton             — Triton kernel  (CUDA events)
      pt.compile(cumsum) — torch.compile on the vectorized cumsum trick  (CUDA events)
      pt.compile(loop)   — torch.compile on the reference timestep loop  (wall-clock)
      numpy(cpu)         — pure NumPy loop on CPU  (wall-clock)

    Both Triton and pt.compile(cumsum) are fully vectorized (no Python timestep
    loop), so CUDA events are appropriate for both.  pt.compile(loop) dispatches
    one op per timestep from Python; wall-clock captures that stall.

    Assertion: Triton must be >=1.5x faster than pt.compile(cumsum) — the
    strongest fully-vectorized PyTorch baseline.
    """
    compiled_vec  = torch.compile(vectorized_episodic_prefix_sum)
    compiled_loop = torch.compile(reference_episodic_prefix_sum)

    _i = torch.randn(64, 512, device="cuda")
    _d = (torch.rand(64, 512, device="cuda") < 0.05).float()
    compiled_vec(_i, _d)
    compiled_loop(_i, _d)
    torch.cuda.synchronize()

    header = (
        f"\n{'num_envs':>10} {'seq_len':>8} "
        f"{'triton':>10} {'compile(vec)':>14} {'compile(loop)':>15} {'numpy(cpu)':>12} "
        f"{'vs vec':>8} {'vs loop':>9} {'vs numpy':>10}"
    )
    print(header)
    print("-" * len(header))

    all_speedups_vec = []

    for num_envs, seq_len in BENCH_CONFIGS:
        inputs    = torch.randn(num_envs, seq_len, device="cuda")
        dones     = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()
        inputs_np = inputs.cpu().numpy()
        dones_np  = dones.cpu().numpy()
        n_warmup, n_iter = _n_iter_gpu(seq_len, num_envs)

        triton_ms = _bench_gpu(compute_episodic_prefix_sum, inputs, dones,
                               n_warmup=n_warmup, n_iter=n_iter)
        vec_ms    = _bench_gpu(compiled_vec, inputs, dones,
                               n_warmup=n_warmup, n_iter=n_iter)
        loop_ms   = _bench_cpu(compiled_loop, inputs, dones)
        numpy_ms  = _bench_cpu(
            lambda i, d: reference_episodic_prefix_sum(
                torch.from_numpy(i), torch.from_numpy(d)
            ),
            inputs_np, dones_np,
        )

        su_vec   = vec_ms  / triton_ms
        su_loop  = loop_ms / triton_ms
        su_numpy = numpy_ms / triton_ms
        all_speedups_vec.append(su_vec)

        print(
            f"{num_envs:>10} {seq_len:>8} "
            f"{triton_ms:>9.3f}ms {vec_ms:>13.3f}ms {loop_ms:>14.3f}ms {numpy_ms:>11.3f}ms "
            f"{su_vec:>6.1f}x {su_loop:>7.1f}x {su_numpy:>8.1f}x"
        )

    print(
        "\ntriton        : CUDA events — pure kernel time, no CPU overhead."
        "\ncompile(vec)  : CUDA events — vectorized cumsum correction, no Python loop."
        "\ncompile(loop) : wall-clock  — one CUDA op per timestep from Python;"
        "\n                CUDA events would miss the CPU stall."
        "\nnumpy(cpu)    : wall-clock  — pure NumPy loop on CPU."
    )

    assert min(all_speedups_vec) >= 1.5, (
        f"Expected >=1.5x speedup over pt.compile(vec), worst was {min(all_speedups_vec):.2f}x"
    )


# ---------------------------------------------------------------------------
# Seq-len limit
# ---------------------------------------------------------------------------

@cuda_only
def test_prefix_sum_seq_len_too_long():
    """seq_len > 131072 must raise — forward chunked kernel not implemented."""
    inputs = torch.zeros(1, 131073, device="cuda")
    dones  = torch.zeros(1, 131073, device="cuda")
    with pytest.raises(AssertionError, match="131072"):
        compute_episodic_prefix_sum(inputs, dones)
