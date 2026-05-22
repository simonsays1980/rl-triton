import torch
import triton

from rl_triton.kernels.gae import gae_scan_kernel


def compute_gae_triton(deltas: torch.Tensor, decays: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation via a backward associative scan.

    Recurrence: A[t] = delta[t] + decay[t] * A[t+1],  A[T] = 0.

    Args:
        deltas: TD residuals, shape [num_envs, seq_len], float32, CUDA, contiguous.
        decays: Per-step decay factors (gamma * lambda * (1 - done)), same shape.

    Returns:
        advantages: shape [num_envs, seq_len], float32.
    """
    assert deltas.shape == decays.shape, "deltas and decays must have the same shape"
    assert deltas.is_cuda and decays.is_cuda, "Inputs must be on CUDA"
    assert deltas.dtype == torch.float32, f"Expected float32, got {deltas.dtype}"
    assert decays.dtype == torch.float32, f"Expected float32, got {decays.dtype}"

    deltas = deltas.contiguous()
    decays = decays.contiguous()

    num_envs, seq_len = deltas.shape
    advantages = torch.empty_like(deltas)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)

    gae_scan_kernel[(num_envs,)](
        deltas, decays, advantages,
        seq_len,
        deltas.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return advantages
