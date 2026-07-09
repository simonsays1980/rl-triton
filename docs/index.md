---
title: rl-triton
hide:
  - toc
---

<style>
.md-content__inner > h1 { display: none; }
</style>

<!-- docs/index.md -->
<p align="center">
  <img src="assets/banner_dark.svg" alt="rl-triton" width="100%"/>
</p>

High-performance [Triton](https://github.com/openai/triton) GPU kernels for common reinforcement learning computations.

## Kernels

| Function | Algorithm | Description | seq_len > 131072 |
|---|---|---|---|
| `compute_gae` | GAE | Generalized Advantage Estimation – backward scan over `δ + γλ·A` | chunked fallback |
| `compute_vtrace` | V-Trace | IS-weighted targets and advantages – fused single-kernel for seq_len ≤ 131072 | chunked fallback |
| `compute_retrace` | Retrace(λ) | Off-policy return estimate with truncated IS ratios | chunked fallback |
| `compute_lambda_returns` | TD(λ) | λ-return targets mixing one-step TD and Monte Carlo | chunked fallback |
| `compute_discounted_returns` | Returns | Discounted reward-to-go | chunked fallback |
| `compute_eligibility_traces` | Elig. traces | Accumulating forward traces `e[t] = x[t] + γλ(1-d)e[t-1]` | not supported |
| `compute_episodic_prefix_sum` | Prefix sum | Episodic cumulative sum with done-mask resets | not supported |

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
import torch
from rl_triton import compute_gae

rewards     = torch.randn(64, 512, device="cuda")
values      = torch.randn(64, 512, device="cuda")
terminateds = torch.zeros(64, 512, device="cuda")
advantages  = compute_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.95)
```

## Testing

```bash
# Correctness tests
pytest tests/ -v

# PR performance safeguard (one config per algorithm, requires CUDA)
pytest -m perf -v

# Full slow benchmark suite (all configs, requires CUDA)
pytest -m slow -v
```

## Benchmarking

The release benchmark runs all algorithms across multiple (num_envs, seq_len) configs
and updates the Performance section of this README automatically:

```bash
python tests/bench_release.py --gpu "RTX 2000 Ada"
```

Use `--no-update` to print results without modifying the README.

See [Benchmarks](benchmarks.md) for full results.
