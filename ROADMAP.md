# Roadmap

`rl-triton` is a focused compute library. The goal is not to cover every RL
algorithm, but to provide the fastest correct implementations of the numerical
primitives that appear in the inner loop of real training systems. Each version
adds new primitives, better performance on existing ones, or broader
compatibility with production training infrastructure.

---

## v0.1 - Foundation (current)

Seven fully-fused Triton kernels, all faster than `torch.compile` vectorized
at production batch sizes (128+ envs, 1024+ steps):

- **GAE** - Generalized Advantage Estimation, backward associative scan
- **V-Trace** - IS-weighted targets and advantages (Espeholt et al. 2018)
- **Retrace(λ)** - Off-policy return estimation with truncated IS ratios
- **Lambda Returns** - TD(λ) targets mixing one-step TD and Monte Carlo
- **Discounted Returns** - γ-discounted cumulative reward sum
- **Eligibility Traces** - Per-parameter credit assignment traces
- **Prefix Sum** - Parallel scan primitive underlying several of the above

---

## v0.2 - Fusion and Precision

- [ ] **Multi-Block Hierarchical 2D Grid Scan** to maximize hardware occupancy of
      low-environment/high-sequence configurations.
- [ ] **Fused GAE + advantage normalisation** (single-pass, eliminates
      redundant reads of the advantages tensor)
- [ ] **Fused value loss kernel** (GAE + MSE/Huber in one pass; `values`
      and `returns` read once instead of three times)
- [ ] **bfloat16 support** across all kernels (accumulate in float32;
      expected to push large-batch speedups from ~2x to ~4x)
- [ ] **CUDA graph support** for small-batch workloads (64x512) where
      kernel launch overhead (~20μs) currently dominates

---

## v0.3 - Modern RL Coverage

Unlike the forward-only scan kernels above, the loss kernels in this
section sit in the gradient path and require hand-written backward passes
(via `torch.autograd.Function`). They will ship only where fusion delivers
a measurable win over autograd-handled PyTorch, which profiling will decide
per kernel.

- [ ] **PPO clipped surrogate loss** (differentiable; fused `exp` + `clamp`
      + `multiply` over the full batch, with a hand-written backward)
- [ ] **GRPO** - Group Relative Policy Optimisation loss kernel
      (differentiable), used in recent LLM reasoning model training
      (DeepSeek-R1, Qwen)
- [ ] **KL divergence estimators** - k1, k2, k3 (Schulman estimator) for
      PPO and RLHF KL penalty terms (differentiable when fused into the loss)
- [ ] **n-step returns** as a complement to Lambda Returns for shorter
      horizons (forward-only, like the existing scan kernels)
      
---

## v0.4 - Multi-GPU Compatibility

- [ ] **Device consistency validation** (clear errors when input tensors
      are on different devices, replacing cryptic CUDA errors)
- [ ] **DDP / FSDP compatibility tests** (verify all kernels work
      correctly within `torch.distributed` without triggering
      synchronisation barriers)
- [ ] **Multi-GPU usage examples** (DDP training loop examples with
      CleanRL and RLlib)
- [ ] **CleanRL integration** - drop-in replacement for CleanRL's GAE
      and return computation
- [ ] **RLlib integration** - connector-level integration for RLlib's
      postprocessing pipeline

---

## Longer Term

- **H100 / Hopper-specific variants** (warp group MMA and TMA
  instructions for scan kernels where applicable)
- **Persistent kernels** (resident kernels processing batches from a
  queue, amortising launch overhead across training steps)
- **JAX interoperability** via `jax.pure_callback`
- **Colab / tutorial notebooks**

---

## Contributing

If a kernel you need is not listed here, open an issue. Requests backed
by a concrete use case and a reference implementation get prioritised.
See [CONTRIBUTING.md](./CONTRIBUTING.md) for the kernel + wrapper + test
structure required for new additions.