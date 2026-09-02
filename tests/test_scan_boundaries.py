"""
Block-boundary correctness tests for the shared non-commutative associative scan.

tl.associative_scan has a documented history of producing wrong results for
non-commutative combine functions at sequence lengths >= 128 and with
reverse=True. _combine in kernels/scan.py is exactly such an operator: it
composes affine maps x -> u + v*x, which do not commute in general
(f_B(f_A(x)) != f_A(f_B(x)) whenever v_A != v_B). Both _run_scan (backward,
reversed load order) and _run_scan_forward (forward, natural order) dispatch
to BLOCK_SIZE=triton.next_power_of_2(seq_len) for seq_len <= _FLAT_MAX_SEQ_LEN,
so seq_len values straddling a power-of-2 boundary exercise genuinely
different BLOCK_SIZE values in tl.associative_scan.

These tests target _run_scan/_run_scan_forward directly rather than any one
op wrapper: every backward-scan op (GAE, V-Trace, Retrace, discounted
returns, lambda returns) and the one forward-scan op (eligibility traces)
delegate to these two functions with the same combine_fn, so correctness
here covers all of them at once.
"""
import pytest
import torch

from bench_utils import assert_correctness

triton = pytest.importorskip("triton")

from rl_triton.ops._scan import _FLAT_MAX_SEQ_LEN, _run_scan, _run_scan_forward

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Straddle the flat-kernel BLOCK_SIZE boundaries at 64 and 128
# (triton.next_power_of_2: 63->64, 64->64, 65->128, 127->128, 128->128, 129->256),
# plus one length spanning multiple full blocks.
SEQ_LENS = [63, 64, 65, 127, 128, 129, 4096 + 1]


def _reference_backward(u: torch.Tensor, v: torch.Tensor, bootstrap: torch.Tensor) -> torch.Tensor:
    """Sequential backward recurrence: A[t] = u[t] + v[t]*A[t+1], A[T]=bootstrap."""
    T = u.shape[1]
    out = torch.zeros_like(u)
    carry = bootstrap.clone()
    for t in reversed(range(T)):
        carry = u[:, t] + v[:, t] * carry
        out[:, t] = carry
    return out


def _reference_forward(u: torch.Tensor, v: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    """Sequential forward recurrence: e[t] = u[t] + v[t]*e[t-1], e[-1]=seed."""
    T = u.shape[1]
    out = torch.zeros_like(u)
    carry = seed.clone()
    for t in range(T):
        carry = u[:, t] + v[:, t] * carry
        out[:, t] = carry
    return out


def _make_uv(num_envs: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    u = torch.randn(num_envs, seq_len, device="cuda")
    # v drawn away from 1.0 so composition order is non-commutative in practice,
    # not just in principle -- v all equal to (near) a constant would make
    # f_A/f_B nearly commute even though the operator itself does not.
    v = torch.rand(num_envs, seq_len, device="cuda") * 0.9 + 0.05
    return u, v


@cuda_only
@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_backward_scan_block_boundary(seq_len):
    num_envs = 8
    u, v = _make_uv(num_envs, seq_len, seed=seq_len)
    bootstrap = torch.rand(num_envs, device="cuda")

    expected = _reference_backward(u, v, bootstrap)
    actual = _run_scan(u, v, bootstrap)
    assert_correctness(actual, expected, label=f"backward_scan[seq_len={seq_len}]")


@cuda_only
@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_forward_scan_block_boundary(seq_len):
    num_envs = 8
    u, v = _make_uv(num_envs, seq_len, seed=seq_len + 1)
    seed = torch.rand(num_envs, device="cuda")

    expected = _reference_forward(u, v, seed)
    actual = _run_scan_forward(u, v, seed)
    assert_correctness(actual, expected, label=f"forward_scan[seq_len={seq_len}]")


@cuda_only
@pytest.mark.slow
def test_backward_scan_chunked_boundary():
    """Same non-commutative _combine, dispatched through the chunked kernel.

    chunked_backward_scan_kernel (scan_chunked.py) imports the identical
    _combine used by the flat kernel above, so it carries the same
    non-commutative-operator risk at a different dispatch path. Forward has
    no chunked kernel (_run_scan_forward asserts seq_len <= _FLAT_MAX_SEQ_LEN),
    so this case is backward-only. Marked slow: the O(seq_len) Python
    reference loop is expensive at this length.
    """
    num_envs = 4
    seq_len = _FLAT_MAX_SEQ_LEN + 1
    u, v = _make_uv(num_envs, seq_len, seed=seq_len)
    bootstrap = torch.rand(num_envs, device="cuda")

    expected = _reference_backward(u, v, bootstrap)
    actual = _run_scan(u, v, bootstrap)
    assert_correctness(actual, expected, label=f"backward_scan_chunked[seq_len={seq_len}]")
