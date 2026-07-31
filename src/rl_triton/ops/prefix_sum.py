import torch
import triton

from rl_triton.kernels.prefix_sum_fused import prefix_sum_fused_kernel
from rl_triton.ops._scan import _FLAT_MAX_SEQ_LEN, _CORRECTNESS_WARNINGS

# Prefix sum carries the fewest registers of any kernel (2 loads, v=1-done, scan).
# Use the same aggressive warp/stage settings as the other light kernels.
# Below 512, BLOCK_SIZE used to fall through .get()'s default (16 warps) --
# see src/rl_triton/ops/gae.py's _WARPS for the H200 measurement basis (flat
# device time for num_warps in {1,2,4} at BLOCK_SIZE 8-128, driven by the
# 32-blocks/SM hard cap on Hopper, register-count-independent at this scale
# since none of these kernels spill at low warps counts).
#
# No entry above 16384, same gap as gae.py's _WARPS -- unverified, out of the
# project's target seq_len regime (see its comment for detail).
_WARPS = {
    8: 2, 16: 2, 32: 2, 64: 2, 128: 2, 256: 4,
    512: 8, 1024: 16, 2048: 16, 4096: 16, 8192: 32, 16384: 32,
}


def compute_episodic_prefix_sum(
    inputs: torch.Tensor,
    dones: torch.Tensor,
    seed_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cumulative sum that resets to zero at episode boundaries.

    Recurrence:

    - C[t] = x[t] + (1 - done[t]) * C[t-1],  C[-1] = seed
    - done[t]=1: accumulator resets, C[t] = x[t]
    - done[t]=0: accumulator continues, C[t] = x[t] + C[t-1]

    Limited to seq_len <= 131072.

    Args:
        inputs:      Values to accumulate x[t], [num_envs, seq_len], float32, CUDA.
        dones:       Episode termination flags (1.0=done), [num_envs, seq_len], float32, CUDA.
        seed_values: Initial carry C[-1] per environment, shape [num_envs].
                     Defaults to zeros.

    Returns:
        prefix_sums: C[t], shape [num_envs, seq_len], float32.
    """
    num_envs, seq_len = inputs.shape

    if _CORRECTNESS_WARNINGS():
        assert inputs.is_cuda and dones.is_cuda, "inputs and dones must be on CUDA"
        assert inputs.dtype == torch.float32, f"inputs: expected float32, got {inputs.dtype}"
        assert dones.dtype == torch.float32,  f"dones: expected float32, got {dones.dtype}"
        assert inputs.shape == dones.shape,   "inputs and dones must have the same shape"
        if seed_values is not None:
            assert seed_values.shape == (num_envs,), \
                f"seed_values must have shape [{num_envs}], got {seed_values.shape}"
            assert seed_values.is_cuda, "seed_values must be on CUDA"

    assert seq_len <= _FLAT_MAX_SEQ_LEN, (
        f"seq_len={seq_len} exceeds the flat kernel limit {_FLAT_MAX_SEQ_LEN}. "
        "A chunked forward scan kernel has not been implemented yet."
    )

    inputs = inputs.contiguous()
    dones  = dones.contiguous()

    has_seed = seed_values is not None
    if has_seed:
        seed_values = seed_values.contiguous()

    out = torch.empty_like(inputs)

    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    num_warps  = _WARPS.get(BLOCK_SIZE, 16)
    num_stages = 2 if BLOCK_SIZE >= 1024 else 1

    prefix_sum_fused_kernel[(num_envs,)](
        inputs, dones,
        out, seed_values,
        seq_len, inputs.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
        HAS_SEED=has_seed,
    )
    return out
