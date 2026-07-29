# Benchmarks archive: unreleased

## unreleased – 2026-07-29 – NVIDIA H100 80GB HBM3

*Measured on NVIDIA H100 80GB HBM3 · 2026-07-29 · [`triton`](https://github.com/openai/triton) kernels vs `torch.compile` baselines and NumPy CPU.*

**Configuration.**  dtype float32 (all kernels require it; see NOTES.md on bf16 and autocast).  gamma=0.99, lambda=0.95 (lambda=0.9 for eligibility traces).  Termination probability ~5% per step; truncation-path tables additionally inject ~5% interior truncated steps (mutually exclusive with terminations) with populated `bootstrap_values`.

**Methodology.**  All GPU full-call timings use CUDA events (start/stop around the complete `compute_*(tensors) -> tensors` call, explicit sync immediately before start); reported value is the min-of-medians across 5 independent trials to filter clock-state noise. Every config is warmed up at its exact shape (20 untimed calls) before any timed call, so `torch.compile` JIT/autotuning and Triton kernel compilation never land in the timed region. A tolerance-based correctness gate (atol=rtol=1e-4 vs. a sequential reference implementation) runs before every timed config — not bit-identical, since `tl.associative_scan` reorders float ops depending on num_warps/block layout, so cross-config last-bit differences are legitimate. A monotonicity gate (2% band) then asserts a larger problem never measures faster than a smaller one along either swept axis. CPU timings are wall-clock (perf_counter), run until at least 0.5 s of samples.

**Two timing granularities.**  **triton** (headline) is full-call wall time — what a caller pays every invocation, including launch overhead and wrapper setup (HAS_TRUNCATIONS/HAS_BOOTSTRAP dispatch, allocation, layout). All speedup ratios are computed from this number. **dev** is device-only CUDA time (`torch.profiler` CUDA activity around steady-state calls, ncu/nsys being unavailable in typical containerized GPU environments) — a diagnostic showing pure kernel execution time; where dev is much smaller than the full-call number, the gap is launch + wrapper overhead the caller still pays. The production-regime table additionally reports an **amortized** variant (N calls in one timed region) for its short-seq_len rows, to separate harness per-call sync overhead from genuine per-call cost — the single-call full-call number remains the ratio basis throughout.

**Columns.**  **triton**: full-call wall time, headline (CUDA events).  **dev**: device-only kernel time, diagnostic (see above).  **compile(vec)**: `torch.compile` applied to the strongest *correct* vectorized PyTorch equivalent found so far – a log2(T)-doubling associative scan (`parallel_suffix_scan`/`parallel_prefix_scan`, no Python loop, no log-space); the same implementation used for the with-truncations tables (called here with truncateds=0) – an earlier log-space cumsum version of this baseline silently underflowed to inf/nan at every size in this table and was replaced (see NOTES.md's log-space-underflow note for the investigation); there is no longer a separate specialized no-truncation baseline to compare against, so the prior compile(assoc) column has been dropped as redundant. This is not necessarily the fastest possible correct baseline – it pays 6-12 kernel launches per call (one per doubling step) where the Triton kernel pays 1-2, and a numerically-stable non-log-space cumsum formulation may exist and would be faster; see NOTES.md for that caveat in full.  **compile(vec-trunc)**: `torch.compile` of the vectorized truncation baseline, used in the with-truncations tables (itself asserted correct against the sequential truncation reference before being trusted as a baseline).  **loop (gpu)**: uncompiled sequential Python loop dispatching GPU ops – the pattern used by CleanRL, RLlib, and most RL codebases today; no `torch.compile`, no vectorization; wall-clock timing.  **np→triton→np**: end-to-end wall-clock for the NumPy adoption path (CPU → GPU transfer, kernel, GPU → CPU transfer).  **numpy cpu**: sequential NumPy loop on CPU – same algorithm as the kernel, no GPU; establishes the CPU reference for each algorithm.  Headline tables below show 4 representative sizes per algorithm (small/parity, mid, main-grid-large, production-adjacent-large); the full CONFIGS grid is reproducible via `python tests/bench_release.py`.
#### GAE (`compute_gae`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.037 |     0.001 |         0.118 |         0.011 |    3.2x |    7.3x |     47.303 |  1289.9x |     19.433 |    529.9x |          0.162 |   120.0x |
|      128 |    1024 |      0.043 |     0.002 |         0.142 |         0.018 |    3.3x |    9.2x |     92.291 |  2157.1x |     69.017 |   1613.1x |          0.354 |   195.0x |
|      256 |    1024 |      0.043 |     0.002 |         0.143 |         0.024 |    3.3x |    9.8x |     95.401 |  2193.7x |     88.729 |   2040.3x |          0.519 |   171.1x |
|      512 |    2048 |      0.044 |     0.008 |         0.146 |         0.084 |    3.3x |   10.4x |    185.573 |  4233.0x |    126.284 |   2880.6x |          1.417 |    89.1x |
|      512 |    4096 |      0.049 |     0.015 |         0.267 |         0.215 |    5.5x |   14.1x |    370.920 |  7630.8x |    284.185 |   5846.5x |          2.485 |   114.4x |
|      512 |     128 |      0.043 |     0.002 |         0.127 |         0.011 |    3.0x |    6.9x |     11.661 |   273.4x |     31.547 |    739.6x |          0.212 |   148.6x |
|      512 |     512 |      0.043 |     0.002 |         0.136 |         0.022 |    3.2x |    9.6x |     46.185 |  1073.9x |     50.200 |   1167.2x |          0.514 |    97.6x |
|     4096 |     128 |      0.043 |     0.004 |         0.128 |         0.034 |    3.0x |    8.5x |     11.726 |   271.8x |     45.938 |   1064.9x |          0.822 |    55.9x |
|     4096 |     512 |      0.043 |     0.009 |         0.216 |         0.168 |    5.1x |   19.2x |     46.703 |  1093.2x |    123.718 |   2896.0x |          2.483 |    49.8x |
|     4096 |    2048 |      0.094 |     0.062 |         0.781 |         0.732 |    8.3x |   11.8x |    186.108 |  1972.8x |    708.585 |   7511.3x |         28.622 |    24.8x |
|    16384 |     128 |      0.045 |     0.011 |         0.192 |         0.145 |    4.3x |   12.7x |     11.639 |   258.9x |    102.233 |   2273.9x |          2.431 |    42.1x |
|    16384 |     512 |      0.078 |     0.046 |         0.682 |         0.634 |    8.7x |   13.8x |     46.201 |   591.5x |    597.343 |   7647.3x |         29.768 |    20.1x |

#### GAE – with truncations (`compute_gae`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.054 |     0.003 |             0.121 |             0.011 |       2.2x |       3.6x |
|      128 |    1024 |      0.064 |     0.004 |             0.146 |             0.018 |       2.3x |       5.0x |
|      256 |    1024 |      0.062 |     0.004 |             0.144 |             0.024 |       2.3x |       5.4x |
|      512 |    2048 |      0.064 |     0.011 |             0.147 |             0.085 |       2.3x |       7.6x |
|      512 |    4096 |      0.086 |     0.035 |             0.265 |             0.215 |       3.1x |       6.1x |
|      512 |     128 |      0.064 |     0.003 |             0.128 |             0.011 |       2.0x |       3.3x |
|      512 |     512 |      0.064 |     0.004 |             0.136 |             0.022 |       2.1x |       5.1x |
|     4096 |     128 |      0.065 |     0.006 |             0.129 |             0.034 |       2.0x |       5.7x |
|     4096 |     512 |      0.073 |     0.021 |             0.215 |             0.167 |       3.0x |       7.9x |
|     4096 |    2048 |      0.126 |     0.077 |             0.781 |             0.731 |       6.2x |       9.5x |
|    16384 |     128 |      0.071 |     0.020 |             0.191 |             0.144 |       2.7x |       7.0x |
|    16384 |     512 |      0.120 |     0.070 |             0.682 |             0.634 |       5.7x |       9.0x |

#### V-Trace (`compute_vtrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.045 |     0.002 |         0.136 |         0.012 |    3.0x |    7.2x |     17.274 |   384.5x |     36.399 |    810.2x |          0.269 |   135.5x |
|      128 |    1024 |      0.051 |     0.002 |         0.176 |         0.022 |    3.4x |    9.5x |     34.297 |   672.8x |     71.776 |   1408.0x |          0.524 |   137.0x |
|      256 |    1024 |      0.050 |     0.003 |         0.175 |         0.030 |    3.5x |    9.9x |     34.392 |   684.6x |     84.036 |   1672.7x |          0.840 |   100.1x |
|      512 |    2048 |      0.053 |     0.010 |         0.170 |         0.100 |    3.2x |   10.2x |     69.586 |  1313.1x |    169.043 |   3190.0x |          3.680 |    45.9x |
|      512 |    4096 |      0.073 |     0.033 |         0.323 |         0.258 |    4.4x |    7.7x |    136.043 |  1851.6x |    284.218 |   3868.4x |          6.292 |    45.2x |
|      512 |     128 |      0.052 |     0.002 |         0.150 |         0.013 |    2.9x |    6.9x |      4.725 |    91.1x |     29.297 |    565.1x |          0.417 |    70.3x |
|      512 |     512 |      0.051 |     0.003 |         0.157 |         0.026 |    3.1x |    9.1x |     17.223 |   335.6x |     50.882 |    991.3x |          0.831 |    61.2x |
|     4096 |     128 |      0.051 |     0.004 |         0.150 |         0.039 |    2.9x |    9.0x |      4.553 |    89.0x |     72.709 |   1421.9x |          1.329 |    54.7x |
|     4096 |     512 |      0.061 |     0.021 |         0.252 |         0.200 |    4.1x |    9.7x |     17.339 |   283.7x |    199.757 |   3268.3x |          6.617 |    30.2x |
|     4096 |    2048 |      0.120 |     0.079 |         0.903 |         0.847 |    7.5x |   10.7x |     68.491 |   570.8x |    816.179 |   6801.5x |         57.042 |    14.3x |
|    16384 |     128 |      0.061 |     0.020 |         0.226 |         0.174 |    3.7x |    8.5x |      4.680 |    76.6x |    183.300 |   3002.2x |          4.128 |    44.4x |
|    16384 |     512 |      0.119 |     0.079 |         0.808 |         0.755 |    6.8x |    9.6x |     17.589 |   148.0x |    838.840 |   7056.2x |         57.187 |    14.7x |

#### V-Trace – with truncations (`compute_vtrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.046 |     0.002 |             0.136 |             0.012 |       3.0x |       6.0x |
|      128 |    1024 |      0.054 |     0.003 |             0.178 |             0.021 |       3.3x |       8.0x |
|      256 |    1024 |      0.053 |     0.004 |             0.177 |             0.031 |       3.3x |       8.7x |
|      512 |    2048 |      0.061 |     0.018 |             0.170 |             0.101 |       2.8x |       5.7x |
|      512 |    4096 |      0.084 |     0.041 |             0.324 |             0.258 |       3.8x |       6.3x |
|      512 |     128 |      0.053 |     0.002 |             0.151 |             0.013 |       2.8x |       6.1x |
|      512 |     512 |      0.054 |     0.003 |             0.158 |             0.026 |       2.9x |       7.8x |
|     4096 |     128 |      0.053 |     0.005 |             0.151 |             0.040 |       2.8x |       7.9x |
|     4096 |     512 |      0.071 |     0.027 |             0.251 |             0.199 |       3.6x |       7.4x |
|     4096 |    2048 |      0.177 |     0.136 |             0.902 |             0.848 |       5.1x |       6.2x |
|    16384 |     128 |      0.070 |     0.027 |             0.226 |             0.174 |       3.2x |       6.5x |
|    16384 |     512 |      0.142 |     0.099 |             0.808 |             0.755 |       5.7x |       7.6x |

#### Retrace(λ) (`compute_retrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.055 |     0.004 |         0.114 |         0.010 |    2.1x |    2.5x |     17.854 |   326.7x |     18.785 |    343.7x |          0.441 |    42.6x |
|      128 |    1024 |      0.063 |     0.005 |         0.136 |         0.015 |    2.2x |    3.0x |     35.530 |   560.5x |     72.640 |   1145.9x |          0.930 |    78.1x |
|      256 |    1024 |      0.064 |     0.007 |         0.137 |         0.020 |    2.1x |    3.0x |     35.481 |   556.3x |    145.200 |   2276.7x |          1.495 |    97.1x |
|      512 |    2048 |      0.098 |     0.046 |         0.139 |         0.076 |    1.4x |    1.6x |     71.010 |   721.9x |    579.447 |   5890.6x |          4.480 |   129.3x |
|      512 |    4096 |      0.373 |     0.284 |         0.210 |         0.159 |    0.6x |    0.6x |    141.633 |   380.1x |   1189.798 |   3192.9x |         10.663 |   111.6x |
|      512 |     128 |      0.064 |     0.003 |         0.121 |         0.010 |    1.9x |    3.5x |      4.596 |    72.1x |     36.961 |    580.1x |          0.594 |    62.2x |
|      512 |     512 |      0.064 |     0.007 |         0.130 |         0.018 |    2.0x |    2.7x |     18.017 |   283.2x |    144.813 |   2276.4x |          1.456 |    99.4x |
|     4096 |     128 |      0.064 |     0.011 |         0.119 |         0.027 |    1.8x |    2.4x |      4.607 |    71.5x |    292.137 |   4532.9x |          2.447 |   119.4x |
|     4096 |     512 |      0.125 |     0.074 |         0.192 |         0.142 |    1.5x |    1.9x |     18.102 |   145.2x |   1195.097 |   9585.9x |         10.618 |   112.6x |
|     4096 |    2048 |      0.377 |     0.327 |         0.659 |         0.607 |    1.7x |    1.9x |     71.248 |   188.9x |   4862.905 |  12892.7x |         70.157 |    69.3x |
|    16384 |     128 |      0.105 |     0.053 |         0.183 |         0.133 |    1.7x |    2.5x |      4.646 |    44.3x |   1202.443 |  11473.7x |         10.752 |   111.8x |
|    16384 |     512 |      0.326 |     0.275 |         0.612 |         0.561 |    1.9x |    2.0x |     18.196 |    55.8x |   4933.593 |  15128.5x |         71.945 |    68.6x |

#### λ-returns (`compute_lambda_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.039 |     0.001 |         0.109 |         0.010 |    2.8x |    6.7x |     41.694 |  1068.9x |      3.108 |     79.7x |
|      128 |    1024 |      0.045 |     0.002 |         0.133 |         0.017 |    2.9x |    8.6x |     82.052 |  1818.5x |      7.114 |    157.7x |
|      256 |    1024 |      0.044 |     0.002 |         0.132 |         0.023 |    3.0x |    9.6x |     82.201 |  1852.0x |      9.671 |    217.9x |
|      512 |    2048 |      0.045 |     0.008 |         0.134 |         0.079 |    3.0x |   10.0x |    162.866 |  3607.1x |     50.622 |   1121.1x |
|      512 |    4096 |      0.049 |     0.015 |         0.259 |         0.210 |    5.3x |   13.9x |    331.771 |  6749.9x |     81.807 |   1664.4x |
|      512 |     128 |      0.045 |     0.002 |         0.114 |         0.010 |    2.5x |    6.3x |     10.281 |   229.8x |      1.498 |     33.5x |
|      512 |     512 |      0.045 |     0.002 |         0.124 |         0.021 |    2.8x |    9.3x |     40.712 |   912.0x |      6.444 |    144.4x |
|     4096 |     128 |      0.044 |     0.004 |         0.114 |         0.033 |    2.6x |    8.3x |     10.167 |   229.7x |     12.651 |    285.8x |
|     4096 |     512 |      0.045 |     0.008 |         0.206 |         0.160 |    4.6x |   19.8x |     40.616 |   904.0x |     74.654 |   1661.6x |
|     4096 |    2048 |      0.094 |     0.061 |         0.759 |         0.712 |    8.1x |   11.7x |    162.062 |  1726.1x |    380.498 |   4052.7x |
|    16384 |     128 |      0.046 |     0.011 |         0.181 |         0.138 |    3.9x |   12.0x |     10.136 |   220.6x |     63.786 |   1388.1x |
|    16384 |     512 |      0.079 |     0.046 |         0.658 |         0.613 |    8.4x |   13.4x |     40.470 |   513.7x |    383.742 |   4870.8x |

#### λ-returns – with truncations (`compute_lambda_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.039 |     0.002 |             0.105 |             0.010 |       2.7x |       5.3x |
|      128 |    1024 |      0.046 |     0.002 |             0.129 |             0.016 |       2.8x |       6.9x |
|      256 |    1024 |      0.045 |     0.003 |             0.128 |             0.024 |       2.8x |       7.6x |
|      512 |    2048 |      0.046 |     0.009 |             0.132 |             0.079 |       2.9x |       8.8x |
|      512 |    4096 |      0.067 |     0.031 |             0.257 |             0.210 |       3.8x |       6.7x |
|      512 |     128 |      0.045 |     0.002 |             0.112 |             0.010 |       2.5x |       5.4x |
|      512 |     512 |      0.045 |     0.003 |             0.120 |             0.021 |       2.7x |       7.5x |
|     4096 |     128 |      0.045 |     0.004 |             0.111 |             0.032 |       2.5x |       7.5x |
|     4096 |     512 |      0.054 |     0.019 |             0.205 |             0.159 |       3.8x |       8.4x |
|     4096 |    2048 |      0.107 |     0.074 |             0.759 |             0.711 |       7.1x |       9.7x |
|    16384 |     128 |      0.054 |     0.018 |             0.181 |             0.138 |       3.4x |       7.5x |
|    16384 |     512 |      0.102 |     0.068 |             0.657 |             0.612 |       6.4x |       9.0x |

#### Discounted returns (`compute_discounted_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.035 |     0.001 |         0.098 |         0.009 |    2.8x |    6.1x |     26.025 |   748.2x |      1.870 |     53.8x |
|      128 |    1024 |      0.040 |     0.002 |         0.121 |         0.015 |    3.1x |    7.3x |     51.731 |  1309.0x |      4.522 |    114.4x |
|      256 |    1024 |      0.039 |     0.003 |         0.121 |         0.022 |    3.1x |    8.2x |     51.732 |  1322.9x |      5.871 |    150.1x |
|      512 |    2048 |      0.040 |     0.008 |         0.123 |         0.071 |    3.0x |    9.5x |    103.937 |  2571.7x |     30.026 |    742.9x |
|      512 |    4096 |      0.045 |     0.014 |         0.239 |         0.195 |    5.3x |   14.3x |    209.376 |  4670.2x |     57.538 |   1283.4x |
|      512 |     128 |      0.039 |     0.002 |         0.104 |         0.008 |    2.7x |    5.5x |      6.553 |   167.7x |      0.955 |     24.5x |
|      512 |     512 |      0.039 |     0.002 |         0.111 |         0.019 |    2.8x |    8.2x |     25.861 |   657.6x |      5.206 |    132.4x |
|     4096 |     128 |      0.040 |     0.004 |         0.104 |         0.030 |    2.6x |    7.7x |      6.575 |   164.5x |     10.853 |    271.5x |
|     4096 |     512 |      0.040 |     0.009 |         0.189 |         0.147 |    4.8x |   16.0x |     25.834 |   650.0x |     53.233 |   1339.4x |
|     4096 |    2048 |      0.088 |     0.058 |         0.694 |         0.651 |    7.9x |   11.3x |    103.747 |  1183.2x |    260.953 |   2976.2x |
|    16384 |     128 |      0.041 |     0.011 |         0.165 |         0.124 |    4.0x |   11.0x |      6.525 |   158.9x |     48.809 |   1188.8x |
|    16384 |     512 |      0.073 |     0.041 |         0.597 |         0.552 |    8.2x |   13.4x |     27.657 |   377.9x |    249.346 |   3407.1x |

#### Discounted returns – with truncations (`compute_discounted_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.036 |     0.002 |             0.098 |             0.009 |       2.7x |       5.2x |
|      128 |    1024 |      0.042 |     0.002 |             0.121 |             0.015 |       2.9x |       6.2x |
|      256 |    1024 |      0.043 |     0.003 |             0.122 |             0.022 |       2.8x |       6.8x |
|      512 |    2048 |      0.044 |     0.009 |             0.124 |             0.071 |       2.9x |       8.1x |
|      512 |    4096 |      0.058 |     0.024 |             0.241 |             0.195 |       4.2x |       8.2x |
|      512 |     128 |      0.042 |     0.002 |             0.104 |             0.008 |       2.5x |       4.7x |
|      512 |     512 |      0.043 |     0.003 |             0.114 |             0.019 |       2.7x |       6.8x |
|     4096 |     128 |      0.042 |     0.004 |             0.105 |             0.030 |       2.5x |       6.9x |
|     4096 |     512 |      0.047 |     0.015 |             0.192 |             0.149 |       4.1x |      10.1x |
|     4096 |    2048 |      0.099 |     0.067 |             0.695 |             0.651 |       7.0x |       9.7x |
|    16384 |     128 |      0.045 |     0.012 |             0.167 |             0.125 |       3.7x |      10.2x |
|    16384 |     512 |      0.089 |     0.057 |             0.595 |             0.552 |       6.7x |       9.7x |

#### Eligibility traces (`compute_eligibility_traces`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.033 |     0.001 |         0.086 |         0.006 |    2.6x |    4.7x |     26.022 |   780.4x |      1.894 |     56.8x |
|      128 |    1024 |      0.039 |     0.002 |         0.108 |         0.009 |    2.8x |    5.1x |     51.636 |  1321.6x |      4.514 |    115.5x |
|      256 |    1024 |      0.041 |     0.002 |         0.109 |         0.011 |    2.7x |    5.3x |     52.786 |  1288.7x |      6.697 |    163.5x |
|      512 |    2048 |      0.040 |     0.003 |         0.108 |         0.028 |    2.7x |    8.1x |    103.774 |  2600.6x |     27.214 |    682.0x |
|      512 |    4096 |      0.040 |     0.007 |         0.117 |         0.067 |    2.9x |   10.3x |    208.387 |  5243.2x |     62.013 |   1560.3x |
|      512 |     128 |      0.040 |     0.002 |         0.090 |         0.005 |    2.2x |    3.4x |      6.503 |   160.8x |      0.916 |     22.6x |
|      512 |     512 |      0.039 |     0.002 |         0.099 |         0.010 |    2.6x |    5.3x |     25.756 |   667.9x |      4.171 |    108.2x |
|     4096 |     128 |      0.040 |     0.004 |         0.090 |         0.013 |    2.3x |    3.2x |      6.472 |   163.6x |      7.966 |    201.4x |
|     4096 |     512 |      0.040 |     0.007 |         0.100 |         0.047 |    2.5x |    7.1x |     25.928 |   652.4x |     54.844 |   1379.9x |
|     4096 |    2048 |      0.063 |     0.035 |         0.326 |         0.284 |    5.1x |    8.2x |    103.115 |  1626.6x |    284.875 |   4493.9x |
|    16384 |     128 |      0.041 |     0.011 |         0.091 |         0.043 |    2.2x |    3.8x |      6.526 |   158.7x |     51.193 |   1245.0x |
|    16384 |     512 |      0.064 |     0.035 |         0.280 |         0.239 |    4.4x |    6.8x |     25.831 |   404.4x |    264.787 |   4145.6x |

#### Episodic prefix sum (`compute_episodic_prefix_sum`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|
|       64 |     512 |      0.030 |     0.001 |         0.083 |         0.006 |    2.8x |    4.6x |
|      128 |    1024 |      0.034 |     0.002 |         0.106 |         0.009 |    3.1x |    5.2x |
|      256 |    1024 |      0.035 |     0.002 |         0.106 |         0.011 |    3.0x |    5.2x |
|      512 |    2048 |      0.036 |     0.003 |         0.106 |         0.028 |    2.9x |    8.0x |
|      512 |    4096 |      0.037 |     0.007 |         0.119 |         0.066 |    3.2x |   10.1x |
|      512 |     128 |      0.035 |     0.002 |         0.087 |         0.005 |    2.5x |    3.4x |
|      512 |     512 |      0.034 |     0.002 |         0.096 |         0.010 |    2.8x |    5.1x |
|     4096 |     128 |      0.036 |     0.004 |         0.088 |         0.012 |    2.5x |    3.2x |
|     4096 |     512 |      0.036 |     0.007 |         0.097 |         0.046 |    2.7x |    7.1x |
|     4096 |    2048 |      0.061 |     0.035 |         0.322 |         0.283 |    5.3x |    8.1x |
|    16384 |     128 |      0.037 |     0.011 |         0.087 |         0.043 |    2.4x |    3.8x |
|    16384 |     512 |      0.060 |     0.035 |         0.277 |         0.239 |    4.6x |    6.8x |

#### Production regime — seq_len [80,128] × num_envs [4096..38400], all algorithms (plus one boundary-marker row, num_envs=16384/seq_len=16)

| algo | num_envs | seq_len | triton full-call (ms) | triton device (ms) | triton amortized (ms) | compile(vec) full-call (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GAE                  |     4096 |      80 |     0.0427 |     0.0039 |     0.0264 |        0.1508 |        0.0216 |   3.53x |   5.47x |
| GAE                  |     8192 |      80 |     0.0430 |     0.0064 |     0.0263 |        0.1572 |        0.0362 |   3.65x |   5.66x |
| GAE                  |    16384 |      80 |     0.0444 |     0.0113 |     0.0259 |        0.1586 |        0.0706 |   3.57x |   6.23x |
| GAE                  |    32768 |      80 |     0.0537 |     0.0215 |     0.0264 |        0.2168 |        0.1550 |   4.04x |   7.22x |
| GAE                  |    38400 |      80 |     0.0570 |     0.0249 |     0.0264 |        0.2457 |        0.1840 |   4.31x |   7.38x |
| GAE                  |     4096 |     128 |     0.0437 |     0.0040 |     0.0261 |        0.1528 |        0.0368 |   3.50x |   9.30x |
| GAE                  |     8192 |     128 |     0.0432 |     0.0065 |     0.0263 |        0.1636 |        0.0740 |   3.79x |  11.47x |
| GAE                  |    16384 |     128 |     0.0460 |     0.0115 |     0.0267 |        0.2233 |        0.1555 |   4.86x |  13.52x |
| GAE                  |    32768 |     128 |     0.0560 |     0.0231 |     0.0266 |        0.3715 |        0.3113 |   6.63x |  13.46x |
| GAE                  |    38400 |     128 |     0.0604 |     0.0269 |     0.0286 |        0.4192 |        0.3627 |   6.95x |  13.48x |
| GAE                  |    16384 |      16 |     0.0445 |     0.0110 |     0.0263 |        0.1504 |        0.0189 |   3.38x |   1.71x | ⚠️
| V-Trace              |     4096 |      80 |     0.0516 |     0.0042 |     0.0346 |        0.1824 |        0.0285 |   3.54x |   6.71x |
| V-Trace              |     8192 |      80 |     0.0519 |     0.0067 |     0.0348 |        0.1961 |        0.0514 |   3.78x |   7.64x |
| V-Trace              |    16384 |      80 |     0.0528 |     0.0120 |     0.0347 |        0.1940 |        0.1028 |   3.67x |   8.59x |
| V-Trace              |    32768 |      80 |     0.0664 |     0.0256 |     0.0337 |        0.2906 |        0.2213 |   4.38x |   8.66x |
| V-Trace              |    38400 |      80 |     0.0691 |     0.0299 |     0.0338 |        0.3271 |        0.2601 |   4.73x |   8.70x |
| V-Trace              |     4096 |     128 |     0.0511 |     0.0043 |     0.0349 |        0.1739 |        0.0429 |   3.40x |   9.95x |
| V-Trace              |     8192 |     128 |     0.0519 |     0.0070 |     0.0344 |        0.1819 |        0.0887 |   3.50x |  12.75x |
| V-Trace              |    16384 |     128 |     0.0611 |     0.0206 |     0.0343 |        0.2536 |        0.1851 |   4.15x |   8.98x |
| V-Trace              |    32768 |     128 |     0.0793 |     0.0405 |     0.0423 |        0.4310 |        0.3691 |   5.43x |   9.11x |
| V-Trace              |    38400 |     128 |     0.0869 |     0.0472 |     0.0492 |        0.4907 |        0.4296 |   5.64x |   9.09x |
| V-Trace              |    16384 |      16 |     0.0521 |     0.0112 |     0.0343 |        0.1854 |        0.0252 |   3.56x |   2.24x | ⚠️
| Retrace              |     4096 |      80 |     0.0642 |     0.0098 |     0.0445 |        0.1470 |        0.0279 |   2.29x |   2.83x |
| Retrace              |     8192 |      80 |     0.0745 |     0.0221 |     0.0445 |        0.1564 |        0.0586 |   2.10x |   2.65x |
| Retrace              |    16384 |      80 |     0.0935 |     0.0424 |     0.0437 |        0.1715 |        0.1208 |   1.83x |   2.85x |
| Retrace              |    32768 |      80 |     0.1311 |     0.0800 |     0.0814 |        0.2968 |        0.2455 |   2.26x |   3.07x |
| Retrace              |    38400 |      80 |     0.1441 |     0.0932 |     0.0942 |        0.3381 |        0.2873 |   2.35x |   3.08x |
| Retrace              |     4096 |     128 |     0.0662 |     0.0114 |     0.0449 |        0.1212 |        0.0272 |   1.83x |   2.39x |
| Retrace              |     8192 |     128 |     0.0804 |     0.0284 |     0.0448 |        0.1300 |        0.0697 |   1.62x |   2.46x |
| Retrace              |    16384 |     128 |     0.1044 |     0.0537 |     0.0554 |        0.1916 |        0.1378 |   1.84x |   2.57x |
| Retrace              |    32768 |     128 |     0.1532 |     0.1027 |     0.1046 |        0.3198 |        0.2674 |   2.09x |   2.60x |
| Retrace              |    38400 |     128 |     0.1699 |     0.1196 |     0.1212 |        0.3638 |        0.3111 |   2.14x |   2.60x |
| Retrace              |    16384 |      16 |     0.0640 |     0.0117 |     0.0442 |        0.1219 |        0.0156 |   1.90x |   1.33x | ⚠️
| lambda-returns       |     4096 |      80 |     0.0439 |     0.0039 |     0.0272 |        0.1379 |        0.0203 |   3.14x |   5.20x |
| lambda-returns       |     8192 |      80 |     0.0450 |     0.0064 |     0.0260 |        0.1425 |        0.0352 |   3.17x |   5.50x |
| lambda-returns       |    16384 |      80 |     0.0456 |     0.0113 |     0.0258 |        0.1429 |        0.0684 |   3.13x |   6.05x |
| lambda-returns       |    32768 |      80 |     0.0546 |     0.0215 |     0.0260 |        0.2093 |        0.1486 |   3.83x |   6.92x |
| lambda-returns       |    38400 |      80 |     0.0582 |     0.0250 |     0.0263 |        0.2371 |        0.1773 |   4.07x |   7.09x |
| lambda-returns       |     4096 |     128 |     0.0451 |     0.0039 |     0.0262 |        0.1365 |        0.0358 |   3.03x |   9.20x |
| lambda-returns       |     8192 |     128 |     0.0441 |     0.0064 |     0.0263 |        0.1428 |        0.0722 |   3.24x |  11.29x |
| lambda-returns       |    16384 |     128 |     0.0452 |     0.0114 |     0.0260 |        0.2130 |        0.1517 |   4.71x |  13.31x |
| lambda-returns       |    32768 |     128 |     0.0564 |     0.0232 |     0.0263 |        0.3533 |        0.2995 |   6.26x |  12.94x |
| lambda-returns       |    38400 |     128 |     0.0603 |     0.0268 |     0.0282 |        0.4035 |        0.3506 |   6.69x |  13.06x |
| lambda-returns       |    16384 |      16 |     0.0456 |     0.0110 |     0.0261 |        0.1498 |        0.0188 |   3.28x |   1.70x | ⚠️
| discounted-returns   |     4096 |      80 |     0.0396 |     0.0039 |     0.0243 |        0.1240 |        0.0183 |   3.13x |   4.73x |
| discounted-returns   |     8192 |      80 |     0.0389 |     0.0063 |     0.0243 |        0.1323 |        0.0309 |   3.40x |   4.87x |
| discounted-returns   |    16384 |      80 |     0.0405 |     0.0113 |     0.0239 |        0.1311 |        0.0579 |   3.24x |   5.13x |
| discounted-returns   |    32768 |      80 |     0.0513 |     0.0211 |     0.0243 |        0.1881 |        0.1321 |   3.67x |   6.25x |
| discounted-returns   |    38400 |      80 |     0.0545 |     0.0246 |     0.0261 |        0.2123 |        0.1567 |   3.90x |   6.37x |
| discounted-returns   |     4096 |     128 |     0.0399 |     0.0039 |     0.0244 |        0.1260 |        0.0329 |   3.16x |   8.49x |
| discounted-returns   |     8192 |     128 |     0.0390 |     0.0064 |     0.0241 |        0.1324 |        0.0647 |   3.40x |  10.16x |
| discounted-returns   |    16384 |     128 |     0.0406 |     0.0113 |     0.0240 |        0.1943 |        0.1386 |   4.79x |  12.27x |
| discounted-returns   |    32768 |     128 |     0.0509 |     0.0215 |     0.0241 |        0.3165 |        0.2679 |   6.21x |  12.46x |
| discounted-returns   |    38400 |     128 |     0.0549 |     0.0250 |     0.0265 |        0.3597 |        0.3118 |   6.55x |  12.47x |
| discounted-returns   |    16384 |      16 |     0.0401 |     0.0110 |     0.0241 |        0.1237 |        0.0151 |   3.09x |   1.37x | ⚠️
| eligibility-traces   |     4096 |      80 |     0.0392 |     0.0039 |     0.0224 |        0.0993 |        0.0136 |   2.53x |   3.48x |
| eligibility-traces   |     8192 |      80 |     0.0391 |     0.0064 |     0.0226 |        0.1070 |        0.0227 |   2.73x |   3.57x |
| eligibility-traces   |    16384 |      80 |     0.0412 |     0.0113 |     0.0224 |        0.1094 |        0.0443 |   2.66x |   3.93x |
| eligibility-traces   |    32768 |      80 |     0.0502 |     0.0212 |     0.0226 |        0.1484 |        0.1047 |   2.96x |   4.95x |
| eligibility-traces   |    38400 |      80 |     0.0542 |     0.0247 |     0.0264 |        0.1702 |        0.1263 |   3.14x |   5.11x |
| eligibility-traces   |     4096 |     128 |     0.0403 |     0.0039 |     0.0228 |        0.0905 |        0.0127 |   2.24x |   3.25x |
| eligibility-traces   |     8192 |     128 |     0.0394 |     0.0064 |     0.0222 |        0.0957 |        0.0217 |   2.43x |   3.42x |
| eligibility-traces   |    16384 |     128 |     0.0407 |     0.0113 |     0.0222 |        0.0964 |        0.0460 |   2.37x |   4.08x |
| eligibility-traces   |    32768 |     128 |     0.0504 |     0.0216 |     0.0231 |        0.1414 |        0.0998 |   2.80x |   4.63x |
| eligibility-traces   |    38400 |     128 |     0.0539 |     0.0250 |     0.0268 |        0.1590 |        0.1176 |   2.95x |   4.71x |
| eligibility-traces   |    16384 |      16 |     0.0405 |     0.0110 |     0.0226 |        0.0849 |        0.0074 |   2.09x |   0.67x | ⚠️
| prefix-sum           |     4096 |      80 |     0.0350 |     0.0039 |     0.0226 |        0.0952 |        0.0135 |   2.72x |   3.47x |
| prefix-sum           |     8192 |      80 |     0.0361 |     0.0064 |     0.0225 |        0.1025 |        0.0230 |   2.84x |   3.61x |
| prefix-sum           |    16384 |      80 |     0.0369 |     0.0113 |     0.0227 |        0.1022 |        0.0441 |   2.77x |   3.91x |
| prefix-sum           |    32768 |      80 |     0.0469 |     0.0211 |     0.0227 |        0.1443 |        0.1045 |   3.08x |   4.94x |
| prefix-sum           |    38400 |      80 |     0.0493 |     0.0247 |     0.0262 |        0.1660 |        0.1264 |   3.36x |   5.11x |
| prefix-sum           |     4096 |     128 |     0.0358 |     0.0039 |     0.0228 |        0.0869 |        0.0123 |   2.43x |   3.16x |
| prefix-sum           |     8192 |     128 |     0.0360 |     0.0064 |     0.0224 |        0.0930 |        0.0224 |   2.58x |   3.51x |
| prefix-sum           |    16384 |     128 |     0.0366 |     0.0113 |     0.0222 |        0.0932 |        0.0455 |   2.54x |   4.02x |
| prefix-sum           |    32768 |     128 |     0.0474 |     0.0216 |     0.0228 |        0.1380 |        0.1003 |   2.91x |   4.65x |
| prefix-sum           |    38400 |     128 |     0.0508 |     0.0250 |     0.0266 |        0.1557 |        0.1174 |   3.07x |   4.70x |
| prefix-sum           |    16384 |      16 |     0.0361 |     0.0110 |     0.0223 |        0.0804 |        0.0074 |   2.23x |   0.68x | ⚠️

*⚠️ marks the boundary-marker row (num_envs=16384, seq_len=16). **vs vec (full-call)** is the headline ratio — the complete `compute_*(tensors) -> tensors` call including launch/wrapper overhead, which a caller pays every invocation. **vs vec (device)** is a diagnostic showing the same ratio for CUDA-kernel-only time; where full-call and device speedups diverge, the gap is launch + wrapper overhead. **triton amortized** is N calls timed inside one region (separates harness per-call sync overhead from genuine per-call cost) — reported alongside, not used for any ratio.*
