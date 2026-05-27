import pytest
import torch

triton = pytest.importorskip("triton")

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
    # done=1 at t=2 resets accumulation for step 2.
    # inputs=[1,2,3,4], dones=[0,0,1,0]
    # C[0]=1, C[1]=3, C[2]=3 (reset), C[3]=3+4=7
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
