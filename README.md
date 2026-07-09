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

<!-- BENCH_START -->
## Performance

*Measured on NVIDIA RTX 2000 Ada Generation · 2026-07-09 · [`triton`](https://github.com/openai/triton) kernels vs `torch.compile` baselines and NumPy CPU.*

**Methodology.**  All GPU timings use CUDA events (start/stop around the kernel, explicit sync before start); reported value is the min-of-medians across 5 independent trials to filter clock-state noise.  CPU timings are wall-clock (perf_counter), run until at least 0.5 s of samples.  A `torch.compile` warmup call is issued once per (function, config) pair before timing so JIT compilation, autotuning, and first-touch allocation never land in the timed region.

**Columns.**  **triton**: pure kernel time (CUDA events).  **compile(vec)**: `torch.compile` applied to the fastest hand-vectorized PyTorch equivalent – cumsum / suffix-product implementations with no Python loop; this is the strongest GPU baseline a competent engineer would write.  **compile(assoc)**: `torch.compile` applied to the unified associative-scan implementation that also handles truncations – called here with zero truncations to show the cost of a single regime-agnostic implementation vs the specialized no-truncation path (GAE, V-Trace, λ-returns, discounted returns only).  **compile(vec-trunc)**: `torch.compile` of the vectorized truncation baseline, used in the with-truncations tables.  **loop (gpu)**: uncompiled sequential Python loop dispatching GPU ops – the pattern used by CleanRL, RLlib, and most RL codebases today; no `torch.compile`, no vectorization; wall-clock timing.  **np→triton→np**: end-to-end wall-clock for the NumPy adoption path (CPU → GPU transfer, kernel, GPU → CPU transfer).  **numpy cpu**: sequential NumPy loop on CPU – same algorithm as the kernel, no GPU; establishes the CPU reference for each algorithm.
#### GAE (`compute_gae`)

| num_envs | seq_len | triton (ms) | compile(vec) (ms) | vs vec | compile(assoc) (ms) | vs assoc | loop gpu (ms) | vs loop | np→triton→np (ms) | numpy cpu (ms) | np→tri→np vs numpy | vs numpy |
|:--------:|:-------:|:-----------:|:-----------------:|:------:|:-------------------:|:--------:|:-------------:|:-------:|:------------------:|:--------------:|:-------------------:|:--------:|
|       64 |     512 |      0.040 |         0.061 |    1.5x |          0.118 |      3.0x |     39.693 |  1001.1x |          0.170 |     17.285 |   101.5x |    436.0x |
|      128 |    1024 |      0.040 |         0.094 |    2.3x |          0.140 |      3.5x |    101.307 |  2524.6x |          0.360 |     41.381 |   115.1x |   1031.2x |
|      256 |    1024 |      0.040 |         0.143 |    3.6x |          0.147 |      3.7x |     83.521 |  2108.3x |          0.603 |     55.962 |    92.7x |   1412.6x |
|      512 |    2048 |      0.068 |         0.439 |    6.4x |          1.035 |     15.2x |    172.795 |  2535.1x |          1.616 |    110.270 |    68.2x |   1617.8x |
|      512 |    4096 |      0.197 |         0.908 |    4.6x |          2.881 |     14.7x |    397.756 |  2023.1x |          3.060 |    226.156 |    73.9x |   1150.3x |

#### GAE – with truncations (`compute_gae`)

| num_envs | seq_len | triton (ms) | compile(vec-trunc) (ms) | vs vec-trunc |
|:--------:|:-------:|:-----------:|:-----------------------:|:------------:|
|       64 |     512 |      0.072 |             0.129 |       1.8x |
|      128 |    1024 |      0.071 |             0.147 |       2.1x |
|      256 |    1024 |      0.078 |             0.161 |       2.1x |
|      512 |    2048 |      0.114 |             1.053 |       9.3x |
|      512 |    4096 |      0.307 |             2.879 |       9.4x |

#### V-Trace (`compute_vtrace`)

| num_envs | seq_len | triton (ms) | compile(vec) (ms) | vs vec | compile(assoc) (ms) | vs assoc | loop gpu (ms) | vs loop | np→triton→np (ms) | numpy cpu (ms) | np→tri→np vs numpy | vs numpy |
|:--------:|:-------:|:-----------:|:-----------------:|:------:|:-------------------:|:--------:|:-------------:|:-------:|:------------------:|:--------------:|:-------------------:|:--------:|
|       64 |     512 |      0.046 |         0.094 |    2.0x |          0.139 |      3.0x |     13.310 |   287.4x |          0.260 |     31.546 |   121.2x |    681.3x |
|      128 |    1024 |      0.047 |         0.112 |    2.4x |          0.170 |      3.6x |     26.232 |   554.3x |          0.589 |     39.616 |    67.3x |    837.1x |
|      256 |    1024 |      0.046 |         0.172 |    3.7x |          0.170 |      3.7x |     26.186 |   566.7x |          1.041 |     72.856 |    70.0x |   1576.7x |
|      512 |    2048 |      0.176 |         0.748 |    4.2x |          1.271 |      7.2x |     51.869 |   294.6x |          2.779 |    108.903 |    39.2x |    618.5x |
|      512 |    4096 |      0.317 |         1.684 |    5.3x |          3.496 |     11.0x |    106.227 |   335.1x |          5.232 |    164.316 |    31.4x |    518.4x |

#### V-Trace – with truncations (`compute_vtrace`)

| num_envs | seq_len | triton (ms) | compile(vec-trunc) (ms) | vs vec-trunc |
|:--------:|:-------:|:-----------:|:-----------------------:|:------------:|
|       64 |     512 |      0.047 |             0.143 |       3.1x |
|      128 |    1024 |      0.048 |             0.164 |       3.4x |
|      256 |    1024 |      0.049 |             0.180 |       3.6x |
|      512 |    2048 |      0.221 |             1.269 |       5.7x |
|      512 |    4096 |      0.411 |             3.494 |       8.5x |

#### Retrace(λ) (`compute_retrace`)

| num_envs | seq_len | triton (ms) | compile(vec) (ms) | vs vec | loop gpu (ms) | vs loop | np→triton→np (ms) | numpy cpu (ms) | np→tri→np vs numpy | vs numpy |
|:--------:|:-------:|:-----------:|:-----------------:|:------:|:-------------:|:-------:|:------------------:|:--------------:|:-------------------:|:--------:|
|       64 |     512 |      0.052 |         0.077 |    1.5x |     13.122 |   250.2x |          0.417 |     17.623 |    42.3x |    336.0x |
|      128 |    1024 |      0.072 |         0.115 |    1.6x |     26.791 |   371.6x |          1.049 |     68.046 |    64.9x |    943.8x |
|      256 |    1024 |      0.101 |         0.166 |    1.6x |     26.199 |   259.1x |          1.784 |    143.103 |    80.2x |   1415.2x |
|      512 |    2048 |      0.409 |         0.730 |    1.8x |     51.985 |   127.1x |          5.743 |    549.803 |    95.7x |   1343.8x |
|      512 |    4096 |      0.977 |         1.463 |    1.5x |    127.534 |   130.5x |         11.131 |   1114.477 |   100.1x |   1140.2x |

#### λ-returns (`compute_lambda_returns`)

| num_envs | seq_len | triton (ms) | compile(vec) (ms) | vs vec | compile(assoc) (ms) | vs assoc | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:-----------:|:-----------------:|:------:|:-------------------:|:--------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.040 |         0.061 |    1.5x |          0.107 |      2.6x |     37.736 |   933.0x |      3.203 |     79.2x |
|      128 |    1024 |      0.041 |         0.093 |    2.3x |          0.140 |      3.4x |     76.196 |  1861.7x |      7.903 |    193.1x |
|      256 |    1024 |      0.040 |         0.153 |    3.8x |          0.145 |      3.6x |     68.489 |  1708.1x |     10.331 |    257.7x |
|      512 |    2048 |      0.067 |         0.439 |    6.5x |          0.991 |     14.8x |    139.139 |  2071.5x |     33.941 |    505.3x |
|      512 |    4096 |      0.197 |         0.900 |    4.6x |          2.835 |     14.4x |    332.615 |  1689.3x |     77.977 |    396.0x |

#### λ-returns – with truncations (`compute_lambda_returns`)

| num_envs | seq_len | triton (ms) | compile(vec-trunc) (ms) | vs vec-trunc |
|:--------:|:-------:|:-----------:|:-----------------------:|:------------:|
|       64 |     512 |      0.041 |             0.106 |       2.6x |
|      128 |    1024 |      0.042 |             0.135 |       3.2x |
|      256 |    1024 |      0.042 |             0.152 |       3.6x |
|      512 |    2048 |      0.084 |             1.022 |      12.2x |
|      512 |    4096 |      0.279 |             2.835 |      10.2x |

#### Discounted returns (`compute_discounted_returns`)

| num_envs | seq_len | triton (ms) | compile(vec) (ms) | vs vec | compile(assoc) (ms) | vs assoc | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:-----------:|:-----------------:|:------:|:-------------------:|:--------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.038 |         0.059 |    1.5x |          0.103 |      2.7x |     23.275 |   606.6x |      2.004 |     52.2x |
|      128 |    1024 |      0.038 |         0.069 |    1.8x |          0.122 |      3.2x |     49.350 |  1298.1x |      5.138 |    135.1x |
|      256 |    1024 |      0.038 |         0.068 |    1.8x |          0.121 |      3.2x |     47.614 |  1241.0x |      7.010 |    182.7x |
|      512 |    2048 |      0.063 |         0.089 |    1.4x |          0.899 |     14.3x |     95.241 |  1513.1x |     22.327 |    354.7x |
|      512 |    4096 |      0.120 |         0.245 |    2.0x |          2.636 |     21.9x |    206.878 |  1720.8x |     48.039 |    399.6x |

#### Discounted returns – with truncations (`compute_discounted_returns`)

| num_envs | seq_len | triton (ms) | compile(vec-trunc) (ms) | vs vec-trunc |
|:--------:|:-------:|:-----------:|:-----------------------:|:------------:|
|       64 |     512 |      0.041 |             0.101 |       2.5x |
|      128 |    1024 |      0.040 |             0.121 |       3.1x |
|      256 |    1024 |      0.043 |             0.122 |       2.8x |
|      512 |    2048 |      0.074 |             0.909 |      12.3x |
|      512 |    4096 |      0.236 |             2.638 |      11.2x |

#### Eligibility traces (`compute_eligibility_traces`)

| num_envs | seq_len | triton (ms) | compile(vec) (ms) | vs vec | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:-----------:|:-----------------:|:------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.037 |         0.051 |    1.4x |     25.321 |   687.5x |      1.996 |     54.2x |
|      128 |    1024 |      0.037 |         0.086 |    2.3x |     54.708 |  1486.6x |      5.096 |    138.5x |
|      256 |    1024 |      0.037 |         0.145 |    4.0x |     53.058 |  1451.9x |      7.005 |    191.7x |
|      512 |    2048 |      0.039 |         0.427 |   11.0x |     93.916 |  2421.5x |     22.197 |    572.3x |
|      512 |    4096 |      0.072 |         0.817 |   11.3x |    207.945 |  2872.8x |     48.142 |    665.1x |

#### Episodic prefix sum (`compute_episodic_prefix_sum`)

| num_envs | seq_len | triton (ms) | compile(vec) (ms) | vs vec |
|:--------:|:-------:|:-----------:|:-----------------:|:------:|
|       64 |     512 |      0.035 |         0.028 |    0.8x |
|      128 |    1024 |      0.036 |         0.035 |    1.0x |
|      256 |    1024 |      0.035 |         0.035 |    1.0x |
|      512 |    2048 |      0.039 |         0.036 |    0.9x |
|      512 |    4096 |      0.063 |         0.066 |    1.0x |
<!-- BENCH_END -->
