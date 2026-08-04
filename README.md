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
(preprint -- arXiv version forthcoming).

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

Headline production-regime numbers below (num_envs=4096, seq_len=128 -- the
PufferLib/Gigaflow-default rollout size), measured on two GPUs spanning a wide
performance range. The Triton-vs-`torch.compile` margin is shape-dependent and does not
move consistently in one direction between these two cards across the full config grid --
see `NOTES.md` for the open investigation; no cross-GPU trend is asserted here. Sourced
directly from `benchmarks.md`'s v0.1.2 production-regime and with-truncations tables at
this exact (num_envs, seq_len); the with-truncations numbers come from the swept grid
table's 4096×128 row, not a separately re-measured single-shape headline run, so treat
them as consistent with -- not necessarily bit-identical to -- a rerun of that one shape
in isolation.

### NVIDIA H100 80GB HBM3

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 3.0× |
| V-Trace | 3.1× |
| Retrace | 1.7× |
| lambda-returns | 3.3× |
| discounted-returns | 3.4× |
| eligibility-traces | 2.4× |
| prefix-sum | 2.3× |

With truncations (terminations + time-limit truncations + bootstrap values), same config.
Eligibility-traces and episodic-prefix-sum have no row here, for a structural reason, not an
unwired gap: both kernels take only a single `dones` flag with no terminated/truncated
distinction and no bootstrap values, so there is no truncation-path baseline to compare
against for either. Every other algorithm, including Retrace (terminated/truncated are both
mandatory, distinct arguments, and no separate bootstrap_values parameter -- the continuation
value is folded into `next_q_values_all` every step, see `docs/kernels/retrace.md` §4), has a
row below.

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 1.6× |
| V-Trace | 2.6× |
| Retrace | 1.8× |
| lambda-returns | 2.4× |
| discounted-returns | 2.6× |

### NVIDIA RTX 2000 Ada Generation

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 3.0× |
| V-Trace | 3.3× |
| Retrace | 1.5× |
| lambda-returns | 3.9× |
| discounted-returns | 4.0× |
| eligibility-traces | 2.2× |
| prefix-sum | 2.2× |

With truncations, same config and the same structural omissions as above:

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 1.4× |
| V-Trace | 2.5× |
| Retrace | 1.6× |
| lambda-returns | 3.2× |
| discounted-returns | 3.3× |

<!-- BENCH_END -->
