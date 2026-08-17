![rl-triton banner](assets/banner_dark.svg)


High-performance [Triton](https://github.com/openai/triton) GPU kernels for common reinforcement learning computations.

[![GPU Tests](https://github.com/simonsays1980/rl-triton/actions/workflows/gpu-tests.yml/badge.svg)](https://github.com/simonsays1980/rl-triton/actions/workflows/gpu-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21984504.svg)](https://doi.org/10.5281/zenodo.21984504)

## Kernels

<!-- KERNELS_START -->
| Function | Algorithm | Description |
|---|---|---|
| `compute_gae` | GAE | Generalized Advantage Estimation – backward scan over `δ + γλ·A` |
| `compute_vtrace` | V-Trace | IS-weighted targets and advantages – fused single-kernel for seq_len ≤ 131072 |
| `compute_retrace` | Retrace(λ) | Off-policy return estimate with truncated IS ratios |
| `compute_lambda_returns` | TD(λ) | λ-return targets mixing one-step TD and Monte Carlo |
| `compute_discounted_returns` | Returns | Discounted reward-to-go |
| `compute_eligibility_traces` | Elig. traces | Accumulating forward traces `e[t] = x[t] + γλ(1-d[t-1])e[t-1]` |
| `compute_episodic_prefix_sum` | Prefix sum | Episodic cumulative sum with done-mask resets |
<!-- KERNELS_END -->

## Documentation

Full docs (kernel tutorials, GPU-concepts background, getting-started guide,
API reference): **[simonsays1980.github.io/rl-triton](https://simonsays1980.github.io/rl-triton)**

## Paper

A preprint describing the kernels, the associative-scan formulation, and the
benchmark methodology is available:
[paper/rl-triton.pdf](paper/rl-triton.pdf)
(preprint -- arXiv version forthcoming).

## Installation

### Requirements

- Linux with a CUDA-capable GPU (Triton compiles and runs GPU kernels; there is no
  CPU fallback).
- Python >=3.10 (tested on 3.10, 3.11).
- PyTorch >=2.2, Triton >=2.3 (installed automatically as dependencies).

### Install

**Use it in your project:**

```bash
pip install git+https://github.com/simonsays1980/rl-triton
```

**From source, editable (for modifying the kernels):**

<!-- INSTALL_START -->
```bash
git clone https://github.com/simonsays1980/rl-triton
cd rl-triton
pip install -e .
```

**Contributors (adds test/dev tooling):**

```bash
pip install -e ".[dev]"
```
<!-- INSTALL_END -->

The package installs as `rl-triton` but imports as `rl_triton` (Python identifiers
can't contain hyphens):

```python
from rl_triton import compute_gae
```

## Usage

<!-- USAGE_START -->
```python
import torch
from rl_triton import compute_gae

rewards     = torch.randn(64, 512, device="cuda")
values      = torch.randn(64, 512, device="cuda")
terminateds = torch.zeros(64, 512, device="cuda")
advantages  = compute_gae(rewards, values, terminateds, gamma=0.99, lambda_=0.95)
```
<!-- USAGE_END -->

## Testing

<!-- TESTING_START -->
```bash
# Correctness tests
pytest tests/ -v

# PR performance safeguard (one config per algorithm, requires CUDA)
pytest -m perf -v

# Full slow benchmark suite (all configs, requires CUDA)
pytest -m slow -v
```
<!-- TESTING_END -->

## Benchmarking

The release benchmark runs all algorithms across multiple (num_envs, seq_len) configs.
Publishing results is a two-step, human-reviewed process: a sweep stages its output in
`docs/benchmark-history/unreleased.md` (never written directly to `benchmarks.md`), and
`--promote <version>` moves a reviewed staging candidate into `benchmarks.md` at release
time. See [benchmarks/README.md](benchmarks/README.md) for the full placement policy.

```bash
python tests/bench_release.py --gpu "NVIDIA H100 80GB HBM3" --parent-sweep
```

<!-- BENCH_START -->
## Performance

*Full sweep, methodology, and truncation-path results: [benchmarks.md](benchmarks.md).*

Headline production-regime numbers below (num_envs=4096, seq_len=128 -- the
PufferLib/Gigaflow-default rollout size), measured on two GPUs spanning a wide
performance range. The Triton-vs-`torch.compile` margin is shape-dependent and does not
move consistently in one direction between these two cards across the full config grid --
see `NOTES.md` for the open investigation; no cross-GPU trend is asserted here. Sourced
directly from `benchmarks.md`'s v0.1.3 production-regime and with-truncations tables at
this exact (num_envs, seq_len).

### NVIDIA H100 80GB HBM3

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 2.36× |
| V-Trace | 2.78× |
| Retrace | 1.91× |
| lambda-returns | 2.85× |
| discounted-returns | 2.70× |
| eligibility-traces | 2.48× |
| prefix-sum | 2.39× |

With truncations (terminations + time-limit truncations + bootstrap values), same config.
Eligibility-traces and episodic-prefix-sum have no row here, for a structural reason, not an
unwired gap: both kernels take only a single `dones` flag with no terminated/truncated
distinction and no bootstrap values, so there is no truncation-path baseline to compare
against for either. Every other algorithm, including Retrace (terminated/truncated are both
mandatory, distinct arguments, and no separate bootstrap_values parameter -- the continuation
value is folded into `next_q_values_all` every step, see `docs/kernels/retrace.md` §4), has a
row below.

The with-truncations margin is smaller than the plain margin for GAE, V-Trace, and
lambda-returns -- not because truncations make the `torch.compile` baseline slower (it runs the
same compiled graph either way), but because the Triton kernel has a `HAS_TRUNCATIONS`
compile-time fast path that skips the truncateds array and the 2D bootstrap-values tensor
entirely when there are no truncations, making the plain-path kernel faster in absolute terms
than the truncation-path kernel. The plain-path headline above should not be read as
representative of truncation-heavy workloads.

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 1.7× |
| V-Trace | 2.7× |
| Retrace | 1.9× |
| lambda-returns | 2.6× |
| discounted-returns | 2.6× |

### NVIDIA RTX 2000 Ada Generation

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 2.57× |
| V-Trace | 3.46× |
| Retrace | 1.62× |
| lambda-returns | 5.14× |
| discounted-returns | 5.70× |
| eligibility-traces | 2.28× |
| prefix-sum | 2.25× |

With truncations, same config and the same structural omissions as above:

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 1.6× |
| V-Trace | 3.3× |
| Retrace | 1.6× |
| lambda-returns | 4.4× |
| discounted-returns | 4.6× |

<!-- BENCH_END -->
