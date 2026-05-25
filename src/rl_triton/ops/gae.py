import torch
import triton

from rl_triton.kernels.gae import gae_scan_kernel
from rl_triton.ops.gae_chunked import compute_gae_chunked

# tl.associative_scan requires BLOCK_SIZE <= 2^17.  Above this the flat kernel
# cannot launch, so we fall back to the chunked kernel automatically.
_FLAT_MAX_SEQ_LEN = 131072


def compute_gae_triton(
    deltas: torch.Tensor,
    decays: torch.Tensor,
    bootstrap_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation via a backward associative scan.

    Recurrence: A[t] = delta[t] + decay[t] * A[t+1], A[T] = bootstrap_values.

    Dispatches to the flat single-block kernel for seq_len <= 131072, and falls
    back to the chunked kernel for longer sequences — the caller never needs to
    choose.

    Args:
        deltas:           TD residuals, shape [num_envs, seq_len], float32, CUDA.
        decays:           Per-step decay factors (gamma * lambda * (1 - done)), same shape.
        bootstrap_values: Per-environment boundary value A[T], shape [num_envs], float32.
                          Use V(s_T) for truncated episodes, 0 for terminated ones.
                          Defaults to zeros (all episodes terminated).

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

    if bootstrap_values is None:
        bootstrap_values = torch.zeros(num_envs, device=deltas.device, dtype=torch.float32)
    else:
        assert bootstrap_values.shape == (num_envs,), \
            f"bootstrap_values must have shape [{num_envs}], got {bootstrap_values.shape}"
        assert bootstrap_values.is_cuda, "bootstrap_values must be on CUDA"
        bootstrap_values = bootstrap_values.contiguous()

    if seq_len > _FLAT_MAX_SEQ_LEN:
        return compute_gae_chunked(deltas, decays, bootstrap_values)

    advantages = torch.empty_like(deltas)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)

    gae_scan_kernel[(num_envs,)](
        deltas, decays, advantages,
        bootstrap_values,
        seq_len,
        deltas.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return advantages
