---
title: Kernels
---

## Overview

All kernels in **rl-triton** share a single architectural idea: express the RL
recurrence as a **linear recurrence** of the form

$$A_t = a_t + b_t \cdot A_{t+1}$$

and solve it in $O(\log N)$ parallel steps using an **associative scan** on the
GPU, rather than the $O(N)$ sequential loop a naïve implementation requires.
The scan runs entirely inside Streaming Multiprocessor SRAM, avoiding repeated
round-trips to High Bandwidth Memory (HBM).

For sequences that fit in a single thread block (`seq_len ≤ 131072`), a
**fully-fused kernel** computes every intermediate quantity - TD errors, IS
ratios, decay products - without materialising any intermediate tensors.
Longer sequences fall back to a chunked scan that stitches results across
blocks.

---

## Choosing the Right Kernel

| Situation | Kernel |
|---|---|
| On-policy PPO / A2C advantage estimation | `compute_gae` |
| Off-policy actor-critic (IMPALA, APPO) | `compute_vtrace` |
| Off-policy Q-learning, discrete actions | `compute_retrace` |
| Critic targets with bias-variance control | `compute_lambda_returns` |
| Simple reward-to-go baseline | `compute_discounted_returns` |
| TD(λ) parameter updates with traces | `compute_eligibility_traces` |
| Episode-scoped cumulative statistics | `compute_episodic_prefix_sum` |

---

## API Reference

::: rl_triton.ops.gae.compute_gae

::: rl_triton.ops.vtrace.compute_vtrace

::: rl_triton.ops.retrace.compute_retrace

::: rl_triton.ops.returns.compute_lambda_returns

::: rl_triton.ops.returns.compute_discounted_returns

::: rl_triton.ops.returns.compute_eligibility_traces

::: rl_triton.ops.prefix_sum.compute_episodic_prefix_sum
