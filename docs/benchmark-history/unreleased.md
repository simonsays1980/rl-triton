# Benchmarks archive: unreleased

## unreleased – 2026-07-28 – NVIDIA H100 80GB HBM3

*Measured on NVIDIA H100 80GB HBM3 · 2026-07-28 · [`triton`](https://github.com/openai/triton) kernels vs `torch.compile` baselines and NumPy CPU.*

**Configuration.**  dtype float32 (all kernels require it; see NOTES.md on bf16 and autocast).  gamma=0.99, lambda=0.95 (lambda=0.9 for eligibility traces).  Termination probability ~5% per step; truncation-path tables additionally inject ~5% interior truncated steps (mutually exclusive with terminations) with populated `bootstrap_values`.

**Methodology.**  All GPU full-call timings use CUDA events (start/stop around the complete `compute_*(tensors) -> tensors` call, explicit sync immediately before start); reported value is the min-of-medians across 5 independent trials to filter clock-state noise. Every config is warmed up at its exact shape (20 untimed calls) before any timed call, so `torch.compile` JIT/autotuning and Triton kernel compilation never land in the timed region. A tolerance-based correctness gate (atol=rtol=1e-4 vs. a sequential reference implementation) runs before every timed config — not bit-identical, since `tl.associative_scan` reorders float ops depending on num_warps/block layout, so cross-config last-bit differences are legitimate. A monotonicity gate (2% band) then asserts a larger problem never measures faster than a smaller one along either swept axis. CPU timings are wall-clock (perf_counter), run until at least 0.5 s of samples.

**Two timing granularities.**  **triton** (headline) is full-call wall time — what a caller pays every invocation, including launch overhead and wrapper setup (HAS_TRUNCATIONS/HAS_BOOTSTRAP dispatch, allocation, layout). All speedup ratios are computed from this number. **dev** is device-only CUDA time (`torch.profiler` CUDA activity around steady-state calls, ncu/nsys being unavailable in typical containerized GPU environments) — a diagnostic showing pure kernel execution time; where dev is much smaller than the full-call number, the gap is launch + wrapper overhead the caller still pays. The production-regime table additionally reports an **amortized** variant (N calls in one timed region) for its short-seq_len rows, to separate harness per-call sync overhead from genuine per-call cost — the single-call full-call number remains the ratio basis throughout.

**Columns.**  **triton**: full-call wall time, headline (CUDA events).  **dev**: device-only kernel time, diagnostic (see above).  **compile(vec)**: `torch.compile` applied to the strongest *correct* vectorized PyTorch equivalent found so far – a log2(T)-doubling associative scan (`parallel_suffix_scan`/`parallel_prefix_scan`, no Python loop, no log-space); the same implementation used for the with-truncations tables (called here with truncateds=0) – an earlier log-space cumsum version of this baseline silently underflowed to inf/nan at every size in this table and was replaced (see NOTES.md's log-space-underflow note for the investigation); there is no longer a separate specialized no-truncation baseline to compare against, so the prior compile(assoc) column has been dropped as redundant. This is not necessarily the fastest possible correct baseline – it pays 6-12 kernel launches per call (one per doubling step) where the Triton kernel pays 1-2, and a numerically-stable non-log-space cumsum formulation may exist and would be faster; see NOTES.md for that caveat in full.  **compile(vec-trunc)**: `torch.compile` of the vectorized truncation baseline, used in the with-truncations tables (itself asserted correct against the sequential truncation reference before being trusted as a baseline).  **loop (gpu)**: uncompiled sequential Python loop dispatching GPU ops – the pattern used by CleanRL, RLlib, and most RL codebases today; no `torch.compile`, no vectorization; wall-clock timing.  **np→triton→np**: end-to-end wall-clock for the NumPy adoption path (CPU → GPU transfer, kernel, GPU → CPU transfer).  **numpy cpu**: sequential NumPy loop on CPU – same algorithm as the kernel, no GPU; establishes the CPU reference for each algorithm.  Headline tables below show 4 representative sizes per algorithm (small/parity, mid, main-grid-large, production-adjacent-large); the full CONFIGS grid is reproducible via `python tests/bench_release.py`.
#### GAE (`compute_gae`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.036 |     0.001 |         0.120 |         0.011 |    3.3x |    7.3x |     48.885 |  1340.1x |     18.664 |    511.6x |          0.161 |   116.0x |
|      128 |    1024 |      0.043 |     0.002 |         0.159 |         0.018 |    3.7x |    9.3x |     96.983 |  2236.7x |     66.914 |   1543.2x |          0.329 |   203.3x |
|      256 |    1024 |      0.042 |     0.002 |         0.156 |         0.025 |    3.8x |    9.9x |     95.136 |  2285.2x |     87.956 |   2112.7x |          0.532 |   165.4x |
|      512 |    2048 |      0.043 |     0.008 |         0.158 |         0.084 |    3.7x |   10.5x |    194.277 |  4534.1x |    123.635 |   2885.4x |          1.446 |    85.5x |
|      512 |    4096 |      0.048 |     0.015 |         0.271 |         0.217 |    5.6x |   14.2x |    380.537 |  7854.5x |    299.253 |   6176.8x |          2.558 |   117.0x |
|      512 |     128 |      0.043 |     0.002 |         0.139 |         0.011 |    3.2x |    7.1x |     11.943 |   274.8x |     32.387 |    745.3x |          0.213 |   151.9x |
|      512 |     512 |      0.043 |     0.002 |         0.148 |         0.024 |    3.5x |   10.3x |     48.752 |  1145.5x |     46.521 |   1093.1x |          0.521 |    89.3x |
|     4096 |     128 |      0.042 |     0.004 |         0.138 |         0.036 |    3.3x |    9.1x |     11.952 |   285.6x |     52.990 |   1266.0x |          0.841 |    63.0x |
|     4096 |     512 |      0.044 |     0.009 |         0.222 |         0.169 |    5.1x |   19.5x |     47.510 |  1086.1x |    126.977 |   2902.7x |          2.540 |    50.0x |
|     4096 |    2048 |      0.093 |     0.062 |         0.793 |         0.739 |    8.5x |   11.9x |    190.317 |  2040.3x |    784.012 |   8404.9x |         28.992 |    27.0x |
|    16384 |     128 |      0.045 |     0.011 |         0.202 |         0.152 |    4.5x |   13.2x |     11.985 |   268.3x |    116.962 |   2618.3x |          2.603 |    44.9x |
|    16384 |     512 |      0.078 |     0.046 |         0.693 |         0.640 |    8.9x |   13.9x |     48.005 |   617.9x |    824.399 |  10610.6x |         29.354 |    28.1x |

#### GAE – with truncations (`compute_gae`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.064 |     0.003 |             0.148 |             0.011 |       2.3x |       3.8x |
|      128 |    1024 |      0.065 |     0.004 |             0.156 |             0.018 |       2.4x |       5.1x |
|      256 |    1024 |      0.064 |     0.004 |             0.156 |             0.025 |       2.4x |       5.5x |
|      512 |    2048 |      0.065 |     0.011 |             0.158 |             0.085 |       2.4x |       7.8x |
|      512 |    4096 |      0.088 |     0.035 |             0.271 |             0.216 |       3.1x |       6.2x |
|      512 |     128 |      0.065 |     0.003 |             0.138 |             0.011 |       2.1x |       3.4x |
|      512 |     512 |      0.064 |     0.004 |             0.149 |             0.024 |       2.3x |       5.4x |
|     4096 |     128 |      0.065 |     0.006 |             0.136 |             0.036 |       2.1x |       6.0x |
|     4096 |     512 |      0.073 |     0.021 |             0.224 |             0.171 |       3.1x |       8.1x |
|     4096 |    2048 |      0.126 |     0.077 |             0.792 |             0.739 |       6.3x |       9.7x |
|    16384 |     128 |      0.071 |     0.021 |             0.197 |             0.148 |       2.8x |       7.2x |
|    16384 |     512 |      0.121 |     0.071 |             0.701 |             0.650 |       5.8x |       9.2x |

#### V-Trace (`compute_vtrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.044 |     0.002 |         0.137 |         0.012 |    3.1x |    7.3x |     17.559 |   399.1x |     39.012 |    886.6x |          0.265 |   147.0x |
|      128 |    1024 |      0.051 |     0.002 |         0.190 |         0.023 |    3.8x |   10.0x |     34.709 |   683.9x |     63.003 |   1241.4x |          0.599 |   105.1x |
|      256 |    1024 |      0.051 |     0.003 |         0.188 |         0.032 |    3.7x |   10.5x |     34.291 |   672.3x |     62.761 |   1230.4x |          0.877 |    71.6x |
|      512 |    2048 |      0.052 |     0.010 |         0.183 |         0.099 |    3.5x |   10.1x |     68.929 |  1322.3x |    158.375 |   3038.2x |          6.684 |    23.7x |
|      512 |    4096 |      0.074 |     0.033 |         0.324 |         0.259 |    4.4x |    7.8x |    136.960 |  1856.0x |    279.901 |   3793.1x |          6.679 |    41.9x |
|      512 |     128 |      0.051 |     0.002 |         0.166 |         0.013 |    3.3x |    7.2x |      4.570 |    89.7x |     29.032 |    569.9x |          0.357 |    81.2x |
|      512 |     512 |      0.052 |     0.003 |         0.172 |         0.027 |    3.3x |    9.5x |     17.454 |   338.2x |     52.442 |   1016.0x |          0.856 |    61.3x |
|     4096 |     128 |      0.052 |     0.004 |         0.168 |         0.041 |    3.2x |    9.6x |      4.568 |    88.1x |     71.689 |   1382.0x |          1.377 |    52.1x |
|     4096 |     512 |      0.060 |     0.021 |         0.262 |         0.201 |    4.3x |    9.7x |     17.389 |   288.0x |    221.631 |   3670.4x |          6.527 |    34.0x |
|     4096 |    2048 |      0.119 |     0.079 |         0.922 |         0.860 |    7.8x |   10.8x |     69.051 |   582.6x |    840.230 |   7088.9x |         38.862 |    21.6x |
|    16384 |     128 |      0.061 |     0.020 |         0.240 |         0.179 |    3.9x |    8.8x |      4.546 |    74.4x |    196.467 |   3214.4x |          4.284 |    45.9x |
|    16384 |     512 |      0.118 |     0.079 |         0.827 |         0.767 |    7.0x |    9.8x |     17.395 |   146.9x |    839.281 |   7088.5x |         57.981 |    14.5x |

#### V-Trace – with truncations (`compute_vtrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.053 |     0.002 |             0.173 |             0.013 |       3.2x |       6.5x |
|      128 |    1024 |      0.055 |     0.003 |             0.194 |             0.023 |       3.5x |       8.5x |
|      256 |    1024 |      0.053 |     0.004 |             0.195 |             0.031 |       3.7x |       9.0x |
|      512 |    2048 |      0.060 |     0.017 |             0.187 |             0.099 |       3.1x |       5.7x |
|      512 |    4096 |      0.084 |     0.041 |             0.324 |             0.260 |       3.9x |       6.4x |
|      512 |     128 |      0.053 |     0.002 |             0.158 |             0.013 |       3.0x |       6.3x |
|      512 |     512 |      0.053 |     0.003 |             0.257 |             0.027 |       4.9x |       8.0x |
|     4096 |     128 |      0.052 |     0.005 |             0.160 |             0.041 |       3.1x |       8.1x |
|     4096 |     512 |      0.069 |     0.027 |             0.260 |             0.199 |       3.8x |       7.5x |
|     4096 |    2048 |      0.176 |     0.137 |             0.923 |             0.860 |       5.2x |       6.3x |
|    16384 |     128 |      0.068 |     0.027 |             0.233 |             0.177 |       3.4x |       6.7x |
|    16384 |     512 |      0.141 |     0.100 |             0.823 |             0.763 |       5.8x |       7.7x |

#### Retrace(λ) (`compute_retrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.054 |     0.004 |         0.109 |         0.010 |    2.0x |    2.5x |     17.443 |   324.3x |     18.857 |    350.5x |          0.431 |    43.8x |
|      128 |    1024 |      0.064 |     0.005 |         0.147 |         0.018 |    2.3x |    3.6x |     36.718 |   575.7x |     72.898 |   1143.0x |          0.971 |    75.1x |
|      256 |    1024 |      0.063 |     0.007 |         0.147 |         0.025 |    2.3x |    3.7x |     36.160 |   578.0x |    156.417 |   2500.3x |          1.518 |   103.0x |
|      512 |    2048 |      0.097 |     0.046 |         0.152 |         0.094 |    1.6x |    2.0x |     76.079 |   787.0x |    590.377 |   6107.0x |          5.295 |   111.5x |
|      512 |    4096 |      0.365 |     0.286 |         0.251 |         0.192 |    0.7x |    0.7x |    146.040 |   400.2x |   1216.520 |   3333.6x |         11.454 |   106.2x |
|      512 |     128 |      0.063 |     0.003 |         0.131 |         0.011 |    2.1x |    3.7x |      5.327 |    84.2x |     43.108 |    681.1x |          0.635 |    67.8x |
|      512 |     512 |      0.064 |     0.007 |         0.143 |         0.023 |    2.2x |    3.5x |     18.188 |   283.5x |    145.267 |   2264.1x |          1.541 |    94.3x |
|     4096 |     128 |      0.064 |     0.012 |         0.131 |         0.033 |    2.0x |    2.8x |      4.699 |    73.4x |    305.369 |   4769.0x |          2.584 |   118.2x |
|     4096 |     512 |      0.125 |     0.074 |         0.221 |         0.165 |    1.8x |    2.2x |     18.215 |   146.0x |   1203.405 |   9647.6x |         13.446 |    89.5x |
|     4096 |    2048 |      0.376 |     0.327 |         0.724 |         0.671 |    1.9x |    2.1x |     71.904 |   191.3x |   4946.697 |  13159.5x |         73.537 |    67.3x |
|    16384 |     128 |      0.104 |     0.053 |         0.203 |         0.147 |    2.0x |    2.8x |      4.812 |    46.4x |   1239.390 |  11939.3x |         10.747 |   115.3x |
|    16384 |     512 |      0.325 |     0.276 |         0.662 |         0.610 |    2.0x |    2.2x |     18.308 |    56.4x |   4969.727 |  15296.4x |         68.757 |    72.3x |

#### λ-returns (`compute_lambda_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.037 |     0.001 |         0.104 |         0.010 |    2.8x |    6.5x |     41.924 |  1138.2x |      2.978 |     80.8x |
|      128 |    1024 |      0.044 |     0.002 |         0.131 |         0.016 |    3.0x |    8.5x |    112.535 |  2583.9x |      6.800 |    156.1x |
|      256 |    1024 |      0.042 |     0.002 |         0.129 |         0.024 |    3.0x |    9.6x |     83.146 |  1965.4x |     11.232 |    265.5x |
|      512 |    2048 |      0.044 |     0.008 |         0.133 |         0.079 |    3.0x |   10.0x |    176.040 |  4039.1x |     42.253 |    969.5x |
|      512 |    4096 |      0.048 |     0.015 |         0.258 |         0.210 |    5.4x |   13.9x |    335.904 |  6988.7x |     85.947 |   1788.2x |
|      512 |     128 |      0.042 |     0.002 |         0.111 |         0.010 |    2.7x |    6.3x |     10.390 |   250.3x |      1.470 |     35.4x |
|      512 |     512 |      0.042 |     0.002 |         0.120 |         0.021 |    2.9x |    9.3x |     41.557 |   994.4x |      7.431 |    177.8x |
|     4096 |     128 |      0.043 |     0.004 |         0.111 |         0.032 |    2.6x |    8.2x |     11.101 |   260.6x |     11.411 |    267.9x |
|     4096 |     512 |      0.042 |     0.008 |         0.206 |         0.160 |    4.9x |   19.8x |     42.591 |  1018.3x |     76.049 |   1818.3x |
|     4096 |    2048 |      0.093 |     0.061 |         0.760 |         0.712 |    8.2x |   11.7x |    165.287 |  1781.1x |    410.483 |   4423.3x |
|    16384 |     128 |      0.044 |     0.011 |         0.181 |         0.137 |    4.1x |   12.0x |     10.422 |   238.4x |     69.133 |   1581.6x |
|    16384 |     512 |      0.078 |     0.046 |         0.665 |         0.615 |    8.5x |   13.4x |     45.078 |   576.9x |    401.255 |   5134.8x |

#### λ-returns – with truncations (`compute_lambda_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.044 |     0.002 |             0.144 |             0.011 |       3.3x |       6.2x |
|      128 |    1024 |      0.044 |     0.002 |             0.152 |             0.018 |       3.5x |       7.3x |
|      256 |    1024 |      0.043 |     0.003 |             0.152 |             0.024 |       3.5x |       7.8x |
|      512 |    2048 |      0.045 |     0.009 |             0.154 |             0.082 |       3.4x |       9.1x |
|      512 |    4096 |      0.067 |     0.031 |             0.269 |             0.212 |       4.0x |       6.9x |
|      512 |     128 |      0.044 |     0.002 |             0.118 |             0.011 |       2.7x |       5.7x |
|      512 |     512 |      0.043 |     0.003 |             0.142 |             0.023 |       3.3x |       8.4x |
|     4096 |     128 |      0.044 |     0.004 |             0.118 |             0.035 |       2.7x |       8.1x |
|     4096 |     512 |      0.052 |     0.019 |             0.219 |             0.169 |       4.2x |       9.0x |
|     4096 |    2048 |      0.106 |     0.073 |             0.780 |             0.728 |       7.4x |      10.0x |
|    16384 |     128 |      0.052 |     0.018 |             0.189 |             0.143 |       3.7x |       7.9x |
|    16384 |     512 |      0.101 |     0.068 |             0.688 |             0.638 |       6.8x |       9.4x |

#### Discounted returns (`compute_discounted_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.036 |     0.001 |         0.100 |         0.008 |    2.8x |    5.9x |     27.742 |   781.0x |      2.086 |     58.7x |
|      128 |    1024 |      0.042 |     0.002 |         0.125 |         0.015 |    3.0x |    7.2x |     54.221 |  1298.4x |      5.658 |    135.5x |
|      256 |    1024 |      0.041 |     0.003 |         0.126 |         0.021 |    3.1x |    7.8x |     53.405 |  1313.1x |      5.905 |    145.2x |
|      512 |    2048 |      0.042 |     0.008 |         0.125 |         0.071 |    3.0x |    9.5x |    106.751 |  2552.4x |     33.099 |    791.4x |
|      512 |    4096 |      0.045 |     0.014 |         0.241 |         0.197 |    5.3x |   14.4x |    218.188 |  4808.5x |     57.504 |   1267.3x |
|      512 |     128 |      0.041 |     0.002 |         0.105 |         0.008 |    2.6x |    5.5x |      6.678 |   164.4x |      0.945 |     23.3x |
|      512 |     512 |      0.040 |     0.002 |         0.116 |         0.019 |    2.9x |    8.2x |     26.750 |   661.3x |      4.216 |    104.2x |
|     4096 |     128 |      0.041 |     0.004 |         0.107 |         0.030 |    2.6x |    7.7x |      6.723 |   162.9x |     11.049 |    267.7x |
|     4096 |     512 |      0.042 |     0.009 |         0.190 |         0.148 |    4.6x |   16.0x |     26.708 |   639.1x |     59.897 |   1433.2x |
|     4096 |    2048 |      0.088 |     0.057 |         0.694 |         0.650 |    7.9x |   11.3x |    106.719 |  1211.4x |    263.619 |   2992.4x |
|    16384 |     128 |      0.043 |     0.011 |         0.166 |         0.124 |    3.9x |   11.0x |      6.683 |   156.9x |     55.272 |   1297.7x |
|    16384 |     512 |      0.070 |     0.041 |         0.593 |         0.551 |    8.4x |   13.4x |     26.716 |   379.3x |    285.593 |   4054.9x |

#### Discounted returns – with truncations (`compute_discounted_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.043 |     0.002 |             0.123 |             0.009 |       2.9x |       5.4x |
|      128 |    1024 |      0.044 |     0.002 |             0.132 |             0.015 |       3.0x |       6.1x |
|      256 |    1024 |      0.043 |     0.003 |             0.134 |             0.021 |       3.2x |       6.5x |
|      512 |    2048 |      0.045 |     0.009 |             0.133 |             0.073 |       3.0x |       8.4x |
|      512 |    4096 |      0.057 |     0.024 |             0.246 |             0.195 |       4.3x |       8.2x |
|      512 |     128 |      0.044 |     0.002 |             0.112 |             0.009 |       2.6x |       5.2x |
|      512 |     512 |      0.043 |     0.003 |             0.124 |             0.020 |       2.9x |       7.3x |
|     4096 |     128 |      0.043 |     0.004 |             0.113 |             0.032 |       2.6x |       7.3x |
|     4096 |     512 |      0.047 |     0.015 |             0.197 |             0.148 |       4.2x |      10.0x |
|     4096 |    2048 |      0.098 |     0.067 |             0.706 |             0.657 |       7.2x |       9.8x |
|    16384 |     128 |      0.045 |     0.012 |             0.173 |             0.129 |       3.9x |      10.5x |
|    16384 |     512 |      0.090 |     0.058 |             0.610 |             0.565 |       6.8x |       9.8x |

#### Eligibility traces (`compute_eligibility_traces`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.033 |     0.001 |         0.089 |         0.006 |    2.7x |    4.7x |     28.166 |   856.2x |      1.905 |     57.9x |
|      128 |    1024 |      0.039 |     0.002 |         0.110 |         0.009 |    2.9x |    5.1x |     55.976 |  1449.3x |      4.543 |    117.6x |
|      256 |    1024 |      0.038 |     0.002 |         0.110 |         0.011 |    2.9x |    5.3x |     63.898 |  1672.4x |      7.078 |    185.2x |
|      512 |    2048 |      0.039 |     0.003 |         0.109 |         0.029 |    2.8x |    8.5x |    111.171 |  2857.0x |     30.119 |    774.0x |
|      512 |    4096 |      0.038 |     0.007 |         0.120 |         0.067 |    3.2x |   10.2x |    223.753 |  5925.7x |     65.848 |   1743.9x |
|      512 |     128 |      0.039 |     0.002 |         0.090 |         0.005 |    2.3x |    3.4x |      7.205 |   186.4x |      0.920 |     23.8x |
|      512 |     512 |      0.037 |     0.002 |         0.100 |         0.010 |    2.7x |    5.4x |     27.797 |   745.6x |      4.403 |    118.1x |
|     4096 |     128 |      0.039 |     0.004 |         0.090 |         0.013 |    2.3x |    3.3x |      6.980 |   179.1x |     11.856 |    304.2x |
|     4096 |     512 |      0.039 |     0.007 |         0.102 |         0.047 |    2.6x |    7.0x |     27.779 |   703.5x |     57.798 |   1463.7x |
|     4096 |    2048 |      0.063 |     0.035 |         0.325 |         0.284 |    5.2x |    8.2x |    112.321 |  1785.4x |    303.364 |   4822.0x |
|    16384 |     128 |      0.039 |     0.011 |         0.091 |         0.043 |    2.3x |    3.8x |      6.970 |   179.0x |     50.308 |   1291.8x |
|    16384 |     512 |      0.062 |     0.035 |         0.279 |         0.239 |    4.5x |    6.8x |     27.826 |   449.6x |    275.189 |   4446.6x |

#### Episodic prefix sum (`compute_episodic_prefix_sum`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|
|       64 |     512 |      0.031 |     0.001 |         0.082 |         0.006 |    2.7x |    4.6x |
|      128 |    1024 |      0.036 |     0.002 |         0.111 |         0.012 |    3.1x |    7.1x |
|      256 |    1024 |      0.036 |     0.002 |         0.111 |         0.017 |    3.1x |    7.8x |
|      512 |    2048 |      0.036 |     0.003 |         0.111 |         0.048 |    3.1x |   14.2x |
|      512 |    4096 |      0.035 |     0.007 |         0.147 |         0.103 |    4.1x |   15.7x |
|      512 |     128 |      0.035 |     0.002 |         0.092 |         0.006 |    2.6x |    4.0x |
|      512 |     512 |      0.035 |     0.002 |         0.102 |         0.015 |    2.9x |    7.5x |
|     4096 |     128 |      0.036 |     0.004 |         0.092 |         0.017 |    2.6x |    4.5x |
|     4096 |     512 |      0.035 |     0.007 |         0.117 |         0.075 |    3.3x |   11.4x |
|     4096 |    2048 |      0.061 |     0.035 |         0.389 |         0.350 |    6.4x |   10.1x |
|    16384 |     128 |      0.036 |     0.011 |         0.100 |         0.059 |    2.7x |    5.2x |
|    16384 |     512 |      0.060 |     0.035 |         0.329 |         0.290 |    5.5x |    8.3x |

#### Production regime — seq_len [80,128] × num_envs [4096..38400], all algorithms (plus one boundary-marker row, num_envs=16384/seq_len=16)

| algo | num_envs | seq_len | triton full-call (ms) | triton device (ms) | triton amortized (ms) | compile(vec) full-call (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GAE                  |     4096 |      80 |     0.0425 |     0.0040 |     0.0270 |        0.1508 |        0.0216 |   3.55x |   5.45x |
| GAE                  |     8192 |      80 |     0.0615 |     0.0064 |     0.0409 |        0.2319 |        0.0361 |   3.77x |   5.63x |
| GAE                  |    16384 |      80 |     0.0438 |     0.0113 |     0.0268 |        0.1609 |        0.0700 |   3.67x |   6.17x |
| GAE                  |    32768 |      80 |     0.0536 |     0.0215 |     0.0271 |        0.2162 |        0.1546 |   4.03x |   7.20x |
| GAE                  |    38400 |      80 |     0.0574 |     0.0249 |     0.0270 |        0.2442 |        0.1838 |   4.25x |   7.38x |
| GAE                  |     4096 |     128 |     0.0433 |     0.0040 |     0.0270 |        0.1513 |        0.0368 |   3.49x |   9.32x |
| GAE                  |     8192 |     128 |     0.0427 |     0.0065 |     0.0268 |        0.1608 |        0.0738 |   3.76x |  11.42x |
| GAE                  |    16384 |     128 |     0.0445 |     0.0115 |     0.0266 |        0.2187 |        0.1555 |   4.92x |  13.58x |
| GAE                  |    32768 |     128 |     0.0548 |     0.0231 |     0.0267 |        0.3660 |        0.3108 |   6.68x |  13.43x |
| GAE                  |    38400 |     128 |     0.0590 |     0.0269 |     0.0284 |        0.4147 |        0.3611 |   7.03x |  13.44x |
| GAE                  |    16384 |      16 |     0.0446 |     0.0110 |     0.0269 |        0.1515 |        0.0193 |   3.39x |   1.75x | ⚠️
| V-Trace              |     4096 |      80 |     0.0514 |     0.0042 |     0.0344 |        0.1875 |        0.0288 |   3.65x |   6.80x |
| V-Trace              |     8192 |      80 |     0.0521 |     0.0067 |     0.0344 |        0.1992 |        0.0513 |   3.82x |   7.62x |
| V-Trace              |    16384 |      80 |     0.0532 |     0.0120 |     0.0336 |        0.1976 |        0.1029 |   3.72x |   8.59x |
| V-Trace              |    32768 |      80 |     0.0660 |     0.0257 |     0.0339 |        0.2922 |        0.2212 |   4.42x |   8.62x |
| V-Trace              |    38400 |      80 |     0.0702 |     0.0299 |     0.0339 |        0.3303 |        0.2602 |   4.71x |   8.70x |
| V-Trace              |     4096 |     128 |     0.0527 |     0.0043 |     0.0343 |        0.1821 |        0.0435 |   3.46x |  10.11x |
| V-Trace              |     8192 |     128 |     0.0515 |     0.0070 |     0.0340 |        0.1834 |        0.0884 |   3.56x |  12.70x |
| V-Trace              |    16384 |     128 |     0.0600 |     0.0207 |     0.0340 |        0.2544 |        0.1853 |   4.24x |   8.97x |
| V-Trace              |    32768 |     128 |     0.0799 |     0.0403 |     0.0424 |        0.4325 |        0.3690 |   5.42x |   9.15x |
| V-Trace              |    38400 |     128 |     0.0869 |     0.0472 |     0.0493 |        0.4921 |        0.4299 |   5.66x |   9.10x |
| V-Trace              |    16384 |      16 |     0.0527 |     0.0112 |     0.0339 |        0.1880 |        0.0253 |   3.57x |   2.25x | ⚠️
| Retrace              |     4096 |      80 |     0.0645 |     0.0098 |     0.0453 |        0.1464 |        0.0277 |   2.27x |   2.83x |
| Retrace              |     8192 |      80 |     0.0741 |     0.0231 |     0.0454 |        0.1559 |        0.0581 |   2.10x |   2.51x |
| Retrace              |    16384 |      80 |     0.0924 |     0.0420 |     0.0455 |        0.1715 |        0.1213 |   1.86x |   2.89x |
| Retrace              |    32768 |      80 |     0.1307 |     0.0798 |     0.0812 |        0.2952 |        0.2450 |   2.26x |   3.07x |
| Retrace              |    38400 |      80 |     0.1435 |     0.0928 |     0.0942 |        0.3392 |        0.2881 |   2.36x |   3.10x |
| Retrace              |     4096 |     128 |     0.0646 |     0.0115 |     0.0457 |        0.1199 |        0.0271 |   1.86x |   2.36x |
| Retrace              |     8192 |     128 |     0.0796 |     0.0286 |     0.0447 |        0.1286 |        0.0695 |   1.62x |   2.43x |
| Retrace              |    16384 |     128 |     0.1035 |     0.0533 |     0.0551 |        0.1898 |        0.1378 |   1.83x |   2.58x |
| Retrace              |    32768 |     128 |     0.1529 |     0.1024 |     0.1042 |        0.3188 |        0.2673 |   2.08x |   2.61x |
| Retrace              |    38400 |     128 |     0.1694 |     0.1194 |     0.1211 |        0.3631 |        0.3104 |   2.14x |   2.60x |
| Retrace              |    16384 |      16 |     0.0639 |     0.0117 |     0.0449 |        0.1222 |        0.0158 |   1.91x |   1.35x | ⚠️
| lambda-returns       |     4096 |      80 |     0.0412 |     0.0039 |     0.0263 |        0.1323 |        0.0207 |   3.21x |   5.30x |
| lambda-returns       |     8192 |      80 |     0.0434 |     0.0064 |     0.0267 |        0.1431 |        0.0353 |   3.30x |   5.53x |
| lambda-returns       |    16384 |      80 |     0.0433 |     0.0113 |     0.0266 |        0.1413 |        0.0690 |   3.27x |   6.10x |
| lambda-returns       |    32768 |      80 |     0.0536 |     0.0214 |     0.0267 |        0.2113 |        0.1511 |   3.94x |   7.05x |
| lambda-returns       |    38400 |      80 |     0.0568 |     0.0250 |     0.0268 |        0.2381 |        0.1796 |   4.20x |   7.20x |
| lambda-returns       |     4096 |     128 |     0.0423 |     0.0039 |     0.0268 |        0.1357 |        0.0352 |   3.21x |   9.04x |
| lambda-returns       |     8192 |     128 |     0.0418 |     0.0064 |     0.0264 |        0.1410 |        0.0722 |   3.38x |  11.29x |
| lambda-returns       |    16384 |     128 |     0.0447 |     0.0114 |     0.0262 |        0.2126 |        0.1512 |   4.76x |  13.26x |
| lambda-returns       |    32768 |     128 |     0.0549 |     0.0231 |     0.0261 |        0.3532 |        0.2995 |   6.43x |  12.96x |
| lambda-returns       |    38400 |     128 |     0.0581 |     0.0268 |     0.0285 |        0.4046 |        0.3507 |   6.97x |  13.08x |
| lambda-returns       |    16384 |      16 |     0.0437 |     0.0110 |     0.0263 |        0.1464 |        0.0188 |   3.35x |   1.71x | ⚠️
| discounted-returns   |     4096 |      80 |     0.0420 |     0.0039 |     0.0247 |        0.1284 |        0.0183 |   3.06x |   4.72x |
| discounted-returns   |     8192 |      80 |     0.0419 |     0.0064 |     0.0246 |        0.1373 |        0.0314 |   3.28x |   4.94x |
| discounted-returns   |    16384 |      80 |     0.0425 |     0.0113 |     0.0244 |        0.1359 |        0.0582 |   3.20x |   5.16x |
| discounted-returns   |    32768 |      80 |     0.0521 |     0.0212 |     0.0246 |        0.1921 |        0.1345 |   3.69x |   6.36x |
| discounted-returns   |    38400 |      80 |     0.0561 |     0.0246 |     0.0261 |        0.2153 |        0.1588 |   3.84x |   6.45x |
| discounted-returns   |     4096 |     128 |     0.0419 |     0.0039 |     0.0244 |        0.1313 |        0.0328 |   3.14x |   8.44x |
| discounted-returns   |     8192 |     128 |     0.0415 |     0.0064 |     0.0242 |        0.1359 |        0.0649 |   3.27x |  10.21x |
| discounted-returns   |    16384 |     128 |     0.0428 |     0.0113 |     0.0243 |        0.1956 |        0.1380 |   4.57x |  12.21x |
| discounted-returns   |    32768 |     128 |     0.0521 |     0.0214 |     0.0243 |        0.3186 |        0.2674 |   6.12x |  12.47x |
| discounted-returns   |    38400 |     128 |     0.0552 |     0.0249 |     0.0263 |        0.3603 |        0.3115 |   6.53x |  12.50x |
| discounted-returns   |    16384 |      16 |     0.0424 |     0.0110 |     0.0243 |        0.1264 |        0.0150 |   2.98x |   1.37x | ⚠️
| eligibility-traces   |     4096 |      80 |     0.0389 |     0.0039 |     0.0235 |        0.1007 |        0.0135 |   2.59x |   3.46x |
| eligibility-traces   |     8192 |      80 |     0.0391 |     0.0064 |     0.0237 |        0.1071 |        0.0229 |   2.74x |   3.60x |
| eligibility-traces   |    16384 |      80 |     0.0403 |     0.0113 |     0.0234 |        0.1086 |        0.0441 |   2.69x |   3.91x |
| eligibility-traces   |    32768 |      80 |     0.0483 |     0.0212 |     0.0235 |        0.1471 |        0.1044 |   3.05x |   4.93x |
| eligibility-traces   |    38400 |      80 |     0.0520 |     0.0247 |     0.0263 |        0.1686 |        0.1264 |   3.24x |   5.11x |
| eligibility-traces   |     4096 |     128 |     0.0384 |     0.0039 |     0.0237 |        0.0917 |        0.0129 |   2.39x |   3.31x |
| eligibility-traces   |     8192 |     128 |     0.0377 |     0.0064 |     0.0234 |        0.0968 |        0.0217 |   2.56x |   3.41x |
| eligibility-traces   |    16384 |     128 |     0.0393 |     0.0113 |     0.0234 |        0.0966 |        0.0459 |   2.46x |   4.06x |
| eligibility-traces   |    32768 |     128 |     0.0492 |     0.0215 |     0.0231 |        0.1400 |        0.0998 |   2.85x |   4.64x |
| eligibility-traces   |    38400 |     128 |     0.0530 |     0.0249 |     0.0267 |        0.1581 |        0.1173 |   2.98x |   4.72x |
| eligibility-traces   |    16384 |      16 |     0.0394 |     0.0110 |     0.0235 |        0.0840 |        0.0074 |   2.13x |   0.67x | ⚠️
| prefix-sum           |     4096 |      80 |     0.0354 |     0.0039 |     0.0224 |        0.0944 |        0.0135 |   2.67x |   3.45x |
| prefix-sum           |     8192 |      80 |     0.0355 |     0.0064 |     0.0221 |        0.1016 |        0.0227 |   2.87x |   3.56x |
| prefix-sum           |    16384 |      80 |     0.0370 |     0.0113 |     0.0217 |        0.1023 |        0.0434 |   2.76x |   3.84x |
| prefix-sum           |    32768 |      80 |     0.0471 |     0.0211 |     0.0224 |        0.1451 |        0.1054 |   3.08x |   4.99x |
| prefix-sum           |    38400 |      80 |     0.0502 |     0.0246 |     0.0261 |        0.1656 |        0.1257 |   3.30x |   5.11x |
| prefix-sum           |     4096 |     128 |     0.0358 |     0.0039 |     0.0222 |        0.0865 |        0.0123 |   2.41x |   3.16x |
| prefix-sum           |     8192 |     128 |     0.0359 |     0.0064 |     0.0223 |        0.0918 |        0.0216 |   2.56x |   3.40x |
| prefix-sum           |    16384 |     128 |     0.0375 |     0.0113 |     0.0221 |        0.0920 |        0.0451 |   2.45x |   3.99x |
| prefix-sum           |    32768 |     128 |     0.0463 |     0.0215 |     0.0228 |        0.1386 |        0.0997 |   2.99x |   4.65x |
| prefix-sum           |    38400 |     128 |     0.0497 |     0.0249 |     0.0266 |        0.1562 |        0.1178 |   3.14x |   4.73x |
| prefix-sum           |    16384 |      16 |     0.0368 |     0.0110 |     0.0223 |        0.0796 |        0.0074 |   2.16x |   0.68x | ⚠️

*⚠️ marks the boundary-marker row (num_envs=16384, seq_len=16). **vs vec (full-call)** is the headline ratio — the complete `compute_*(tensors) -> tensors` call including launch/wrapper overhead, which a caller pays every invocation. **vs vec (device)** is a diagnostic showing the same ratio for CUDA-kernel-only time; where full-call and device speedups diverge, the gap is launch + wrapper overhead. **triton amortized** is N calls timed inside one region (separates harness per-call sync overhead from genuine per-call cost) — reported alongside, not used for any ratio.*
