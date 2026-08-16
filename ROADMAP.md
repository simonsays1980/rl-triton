# Roadmap

`rl-triton` is a focused compute library. The goal is not to cover every RL
algorithm, but to provide the fastest correct implementations of the numerical
primitives that appear in the inner loop of real training systems. Each version
adds new primitives, better performance on existing ones, or broader
compatibility with production training infrastructure.

---

## Why this order: the paper's Amdahl result

The project's paper (`docs/paper.tex`) measures GAE end-to-end inside a
synthetic PPO update (Isaac-Gym-Ant-like actor-critic, 4096 envs, seq_len=128,
4 epochs x 4 minibatches), not just as an isolated kernel call. At a realistic
policy size (hidden 1024x1024), GAE is **≤0.13% of total step time** on
either the Triton or `torch.compile` arm, and end-to-end speedup from GAE's
kernel alone is **within noise of 1x** (0.983-0.998x measured across sizes
and net modes) -- the isolated 1.6-2.57x per-call kernel speedup becomes
invisible once it shares the step with a real backward pass. This is
Amdahl's law, and that configuration is its least favourable point.

The paper identifies four regimes where credit assignment's addressable
share of step time actually rises, and this roadmap is prioritised around
them:

1. **Smaller policy nets** -- hidden 1024→256 moves GAE's share
   0.13%→0.53% in the paper's measurements.
2. **Longer `seq_len`** -- the baseline scan is O(T) serial while per-step
   network cost is roughly constant in T, so the addressable share grows
   with sequence length.
3. **Per-epoch / per-minibatch advantage recompute** -- any pipeline that
   recomputes GAE more than once per rollout multiplies its share of step
   time accordingly.
4. **Asynchronous actor-learner training** (IMPALA, APPO/Sample Factory),
   where learner-side V-Trace latency directly sets behaviour-policy
   staleness -- a latency-critical path, not just a throughput share.

Work below is ranked higher when it has a plausible end-to-end effect in
one of these four regimes, removes a whole pass over data that today costs
a separate kernel launch and HBM round-trip, or unlocks a pipeline the
library cannot currently serve at all regardless of speed (dtype
compatibility). Work that only improves the isolated per-call ratio at a
shape the paper shows is launch-overhead-dominated -- and therefore
invisible end-to-end at the large-net baseline above -- is ranked lower,
even when the ratio improvement itself would be large.

---

## v0.1 - Foundation (current)

Seven fully-fused Triton kernels. Below the Retrace dispatch ceiling
(`seq_len<=2048`), all seven are faster than `torch.compile` vectorized at
production batch sizes (128+ envs, 1024+ steps); above that ceiling,
Retrace(λ)'s fused kernel is a confirmed, measured loss against
`torch.compile` and the library reroutes to a slower-but-bounded fallback
(see the v0.2 backlog item below).

- **GAE** - Generalized Advantage Estimation, backward associative scan
- **V-Trace** - IS-weighted targets and advantages (Espeholt et al. 2018)
- **Retrace(λ)** - Off-policy return estimation with truncated IS ratios
- **Lambda Returns** - TD(λ) targets mixing one-step TD and Monte Carlo
- **Discounted Returns** - γ-discounted cumulative reward sum
- **Eligibility Traces** - Per-parameter credit assignment traces
- **Prefix Sum** - Parallel scan primitive underlying several of the above

---

## v0.2 - Fusion and Precision

Ordered by end-to-end effect first (regimes 1-4 above), then by whether the
item removes a whole data pass or unlocks a pipeline outright, then by
isolated-ratio-only improvements.

- [ ] **Fused GAE + advantage normalisation** (single-pass, eliminates
      redundant reads of the advantages tensor). Removes a whole pass;
      directly reduces GAE's step-time share in regimes 1-3 above.
- [ ] **Fused value loss kernel** (GAE + MSE/Huber in one pass; `values`
      and `returns` read once instead of three times). Removes a whole
      pass; same rationale as above, and moves work into the loss
      computation the backward pass already dominates.
- [ ] **bf16 I/O with float32 accumulation.** All kernels currently raise
      `TypeError` on non-float32 input (paper §5.4). This is a
      compatibility unlock, not a speedup: callers with bf16 rollout
      buffers -- the native dtype of RLHF/LLM-post-training pipelines,
      which the paper's own Applicability section names as a target use
      case (GRPO, RLHF) -- must otherwise materialise a float32 copy,
      paying extra memory in the regime where memory binds, an HBM
      round-trip that gives back part of the fused kernel's win, and
      hand-written casts inside otherwise-bf16 autocast graphs. A kernel
      that cannot accept the pipeline's native dtype cannot compete for
      that work at any speed. Prerequisite deliverable: a precision-drift
      study across algorithms and sequence lengths (bf16 I/O, float32
      accumulation), which the paper defers pending exactly this. This
      needs measuring, not assuming -- credit-assignment errors in this
      library are silent (finite, plausible-looking output, not NaN/Inf),
      as the v0.1.2 window-boundary bootstrap double-counting bug
      (CHANGELOG.md) demonstrated; a precision regression at long T could
      look the same way.
- [ ] **Warp-count tuning above `BLOCK_SIZE=16384`.** Currently falls back
      to an untuned default of 16 warps above that size. Tuning at small
      `BLOCK_SIZE` recovered 2.1-2.7x there; the paper flags comparable
      headroom above 16384 as plausible but unmeasured. Cheap -- a tuning
      table extension, no new correctness surface, no new kernel path.
- [ ] **Retrace's register-pressure regression above `seq_len=2048`.**
      Confirmed, measured loss against `torch.compile` (0.63-0.73x at
      seq_len=4096, degrading to 0.16-0.24x at seq_len=8192), caused by
      re-reading the 3D action-probability tensor in-kernel to compute
      c_{t+1}, which pushes register demand past budget and collapses
      occupancy to 25%. The current dispatch ceiling at seq_len=2048
      only bounds the loss (reroutes to a fallback that is itself
      0.38-0.47x against `torch.compile`); it does not fix it. A real
      fix needs a kernel structure that avoids holding both the original
      and shifted action-probability tensor live in registers at once --
      likely the same multi-block/hierarchical grid restructuring as the
      2D grid scan item below, since both are register-pressure-at-scale
      problems.
- [ ] **Multi-Block Hierarchical 2D Grid Scan** to maximize hardware
      occupancy of low-environment/high-sequence configurations.
- [ ] **Chunked fallback for the two forward scans** (eligibility traces,
      episodic prefix sum). Every other kernel already falls back to an
      unfused, two-pass chunked path above the flat kernel's
      `seq_len<=131072` limit; these two currently have no fallback at
      all and simply fail above that length.
- [ ] **CUDA graph support** for small-batch workloads (64x512) where
      kernel launch overhead (~20us) currently dominates. This targets
      the launch-bound regime the paper's Amdahl measurement shows is
      invisible end-to-end at realistic policy sizes (GAE ≤0.13% of step
      time) -- worth doing, but ranked below work with a demonstrated or
      plausible end-to-end effect.

---

## v0.3 - Modern RL Coverage

Unlike the forward-only scan kernels above, the loss kernels in this
section sit in the gradient path and require hand-written backward passes
(via `torch.autograd.Function`). They will ship only where fusion delivers
a measurable win over autograd-handled PyTorch, which profiling will decide
per kernel.

- [ ] **PPO clipped surrogate loss** (differentiable; fused `exp` + `clamp`
      + `multiply` over the full batch, with a hand-written backward)
- [ ] **GRPO loss kernel and group-relative advantage normalisation** -
      Group Relative Policy Optimisation (differentiable), used in recent
      LLM reasoning model training (DeepSeek-R1, Qwen). GRPO's discounted
      reward-to-go already ships today via `compute_discounted_returns`;
      what's missing is the loss kernel itself and the group-relative
      advantage normalisation step.
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

Integration work is ranked here because it is what would generate real
end-to-end profiling evidence in the regimes identified above (per-epoch
advantage recompute, async actor-learner staleness) rather than more
isolated-kernel benchmarks.

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
