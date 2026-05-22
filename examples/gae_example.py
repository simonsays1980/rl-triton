"""
Standalone example: Generalized Advantage Estimation with the Triton kernel.

Run with:
    python examples/gae_example.py
"""
import torch
from rl_triton.ops.gae import compute_gae_triton


def reference_gae(deltas, decays):
    T = deltas.shape[1]
    adv = torch.zeros_like(deltas)
    gae = torch.zeros(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    for t in reversed(range(T)):
        gae = deltas[:, t] + decays[:, t] * gae
        adv[:, t] = gae
    return adv


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping example.")
        return

    torch.manual_seed(0)
    num_envs, seq_len = 256, 1024
    deltas = torch.randn(num_envs, seq_len, device="cuda")
    decays = torch.rand(num_envs, seq_len, device="cuda") * 0.99

    ref = reference_gae(deltas, decays)
    out = compute_gae_triton(deltas, decays)

    max_err = (out - ref).abs().max().item()
    print(f"num_envs={num_envs}, seq_len={seq_len}")
    print(f"Max absolute error vs reference: {max_err:.2e}")
    print(f"advantages[:2, :4]:\n{out[:2, :4]}")


if __name__ == "__main__":
    main()
