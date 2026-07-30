![rl-triton banner](assets/banner_dark.svg)


High-performance [Triton](https://github.com/openai/triton) GPU kernels for common reinforcement learning computations.

## Kernels

| Function | Algorithm | Description |
|---|---|---|
| `compute_gae` | GAE | Generalized Advantage Estimation – backward scan over `δ + γλ·A` |
| `compute_vtrace` | V-Trace | IS-weighted targets and advantages – fused single-kernel for seq_len ≤ 131072 |
| `compute_retrace` | Retrace(λ) | Off-policy return estimate with truncated IS ratios |
| `compute_lambda_returns` | TD(λ) | λ-return targets mixing one-step TD and Monte Carlo |
| `compute_discounted_returns` | Returns | Discounted reward-to-go |
| `compute_eligibility_traces` | Elig. traces | Accumulating forward traces `e[t] = x[t] + γλ(1-d)e[t-1]` |
| `compute_episodic_prefix_sum` | Prefix sum | Episodic cumulative sum with done-mask resets |

## Paper

A preprint describing the kernels, the associative-scan formulation, and the
benchmark methodology is available:
[paper/rl-triton.pdf](paper/rl-triton.pdf)
(preprint — arXiv version forthcoming).

<!-- TODO: swap to arXiv link + add BibTeX + CITATION.cff once posted -->

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

Headline production-regime numbers below (num_envs=4096, seq_len=128 — the
PufferLib/Gigaflow-default rollout size), measured on two GPUs spanning a wide
performance range. The Triton-vs-`torch.compile` margin is shape-dependent and does not
move consistently in one direction between these two cards across the full config grid —
see `NOTES.md` for the open investigation; no cross-GPU trend is asserted here.

### NVIDIA H100 80GB HBM3

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 3.5× |
| V-Trace | 3.4× |
| Retrace | 1.8× |
| lambda-returns | 3.0× |
| discounted-returns | 3.2× |
| eligibility-traces | 2.2× |
| prefix-sum | 2.4× |

With truncations (terminations + time-limit truncations + bootstrap values), same config.
Two algorithms have no row here, for two different reasons: Retrace's kernel supports
truncations and has a valid vectorized baseline (`vectorized_retrace`), but that baseline
isn't wired into this headline table yet — a gap to close, not a validity problem.
Eligibility-traces and episodic-prefix-sum have no truncation-path concept at all (both
kernels take only a single `dones` flag, no bootstrap values) — structurally not
applicable, not an unwired gap.

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 2.0× |
| V-Trace | 2.8× |
| lambda-returns | 2.5× |
| discounted-returns | 2.5× |

### NVIDIA RTX 2000 Ada Generation

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 5.2× |
| V-Trace | 5.7× |
| Retrace | 1.6× |
| lambda-returns | 4.6× |
| discounted-returns | 4.4× |
| eligibility-traces | 2.1× |
| prefix-sum | 2.2× |

With truncations, same config and same two omissions as above:

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 2.5× |
| V-Trace | 4.5× |
| lambda-returns | 3.7× |
| discounted-returns | 3.7× |

<!-- BENCH_END -->
