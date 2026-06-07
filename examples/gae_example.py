"""
Standalone example: Generalized Advantage Estimation with the Triton kernel.

Run with:
    python examples/gae_example.py
"""
import torch
from rl_triton.ops.gae import compute_gae_triton


def reference_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """Pure-PyTorch backward scan — matches kernel semantics exactly."""
    T     = rewards.shape[1]
    adv   = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    next_values = torch.empty_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1]  = 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - dones[:, t]
        delta    = rewards[:, t] + gamma * not_done * next_values[:, t] - values[:, t]
        carry    = delta + gamma * lambda_ * not_done * carry
        adv[:, t] = carry
    return adv


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping example.")
        return

    torch.manual_seed(0)
    num_envs, seq_len = 256, 1024
    rewards = torch.randn(num_envs, seq_len, device="cuda")
    values  = torch.randn(num_envs, seq_len, device="cuda")
    dones   = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()

    ref = reference_gae(rewards, values, dones)
    out = compute_gae_triton(rewards, values, dones, gamma=0.99, lambda_=0.95)

    max_err = (out - ref).abs().max().item()
    print(f"num_envs={num_envs}, seq_len={seq_len}")
    print(f"Max absolute error vs reference: {max_err:.2e}")
    print(f"advantages[:2, :4]:\n{out[:2, :4]}")


if __name__ == "__main__":
    main()
