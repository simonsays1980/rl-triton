# Benchmarks

Latest release only — see docs/benchmark-history/ for prior releases.

## unreleased – 2026-07-27 – NVIDIA H200

*Measured on NVIDIA H200 · 2026-07-27 · [`triton`](https://github.com/openai/triton) kernels vs `torch.compile` baselines and NumPy CPU.*

**Configuration.**  dtype float32 (all kernels require it; see NOTES.md on bf16 and autocast).  gamma=0.99, lambda=0.95 (lambda=0.9 for eligibility traces).  Termination probability ~5% per step; truncation-path tables additionally inject ~5% interior truncated steps (mutually exclusive with terminations) with populated `bootstrap_values`.

**Methodology.**  All GPU full-call timings use CUDA events (start/stop around the complete `compute_*(tensors) -> tensors` call, explicit sync immediately before start); reported value is the min-of-medians across 5 independent trials to filter clock-state noise. Every config is warmed up at its exact shape (20 untimed calls) before any timed call, so `torch.compile` JIT/autotuning and Triton kernel compilation never land in the timed region. A tolerance-based correctness gate (atol=rtol=1e-4 vs. a sequential reference implementation) runs before every timed config — not bit-identical, since `tl.associative_scan` reorders float ops depending on num_warps/block layout, so cross-config last-bit differences are legitimate. A monotonicity gate (2% band) then asserts a larger problem never measures faster than a smaller one along either swept axis. CPU timings are wall-clock (perf_counter), run until at least 0.5 s of samples.

**Two timing granularities.**  **triton** (headline) is full-call wall time — what a caller pays every invocation, including launch overhead and wrapper setup (HAS_TRUNCATIONS/HAS_BOOTSTRAP dispatch, allocation, layout). All speedup ratios are computed from this number. **dev** is device-only CUDA time (`torch.profiler` CUDA activity around steady-state calls, ncu/nsys being unavailable in typical containerized GPU environments) — a diagnostic showing pure kernel execution time; where dev is much smaller than the full-call number, the gap is launch + wrapper overhead the caller still pays. The production-regime table additionally reports an **amortized** variant (N calls in one timed region) for its short-seq_len rows, to separate harness per-call sync overhead from genuine per-call cost — the single-call full-call number remains the ratio basis throughout.

**Columns.**  **triton**: full-call wall time, headline (CUDA events).  **dev**: device-only kernel time, diagnostic (see above).  **compile(vec)**: `torch.compile` applied to the fastest correct vectorized PyTorch equivalent – a log2(T)-doubling associative scan (`parallel_suffix_scan`/`parallel_prefix_scan`, no Python loop, no log-space); this is the strongest GPU baseline a competent engineer would write, and the same implementation used for the with-truncations tables (called here with truncateds=0) – an earlier log-space cumsum version of this baseline silently underflowed to inf/nan at every size in this table and was replaced (see docs/benchmark-history/ for the investigation); there is no longer a separate specialized no-truncation baseline to compare against, so the prior compile(assoc) column has been dropped as redundant.  **compile(vec-trunc)**: `torch.compile` of the vectorized truncation baseline, used in the with-truncations tables (itself asserted correct against the sequential truncation reference before being trusted as a baseline).  **loop (gpu)**: uncompiled sequential Python loop dispatching GPU ops – the pattern used by CleanRL, RLlib, and most RL codebases today; no `torch.compile`, no vectorization; wall-clock timing.  **np→triton→np**: end-to-end wall-clock for the NumPy adoption path (CPU → GPU transfer, kernel, GPU → CPU transfer).  **numpy cpu**: sequential NumPy loop on CPU – same algorithm as the kernel, no GPU; establishes the CPU reference for each algorithm.  Headline tables below show 4 representative sizes per algorithm (small/parity, mid, main-grid-large, production-adjacent-large); the full CONFIGS grid is reproducible via `python tests/bench_release.py`.
#### GAE (`compute_gae`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.034 |     0.002 |         0.113 |         0.013 |    3.3x |    7.5x |     43.822 |  1285.9x |     17.454 |    512.1x |          0.147 |   119.0x |
|      128 |    1024 |      0.040 |     0.002 |         0.146 |         0.020 |    3.7x |    9.4x |     88.249 |  2211.5x |    140.023 |   3509.0x |          0.272 |   515.6x |
|      256 |    1024 |      0.039 |     0.003 |         0.146 |         0.027 |    3.7x |   10.1x |     87.114 |  2216.9x |    157.478 |   4007.5x |          0.478 |   329.2x |
|      512 |    2048 |      0.040 |     0.008 |         0.149 |         0.080 |    3.7x |    9.7x |    176.032 |  4369.3x |    199.251 |   4945.7x |          1.239 |   160.9x |
|      512 |    4096 |      0.047 |     0.015 |         0.231 |         0.180 |    5.0x |   11.8x |    350.514 |  7523.0x |    299.931 |   6437.4x |          2.192 |   136.9x |
|      512 |     128 |      0.041 |     0.002 |         0.131 |         0.013 |    3.2x |    7.2x |     11.005 |   271.2x |    156.962 |   3868.3x |          0.185 |   847.4x |
|      512 |     512 |      0.040 |     0.003 |         0.138 |         0.026 |    3.5x |   10.2x |     44.060 |  1105.0x |    140.874 |   3533.2x |          0.485 |   290.4x |
|     4096 |     128 |      0.039 |     0.004 |         0.128 |         0.038 |    3.2x |    9.0x |     11.087 |   281.4x |    138.249 |   3509.6x |          0.703 |   196.5x |
|     4096 |     512 |      0.041 |     0.009 |         0.193 |         0.144 |    4.8x |   16.9x |     43.850 |  1079.0x |    159.874 |   3933.9x |          2.096 |    76.3x |
|     4096 |    2048 |      0.090 |     0.061 |         0.621 |         0.581 |    6.9x |    9.5x |    175.192 |  1936.6x |    401.187 |   4434.8x |         24.416 |    16.4x |
|    16384 |     128 |      0.042 |     0.012 |         0.187 |         0.141 |    4.4x |   11.9x |     10.972 |   259.0x |    162.019 |   3824.1x |          2.060 |    78.6x |
|    16384 |     512 |      0.065 |     0.035 |         0.558 |         0.522 |    8.6x |   14.7x |     43.824 |   677.6x |    319.355 |   4938.1x |         23.464 |    13.6x |

#### GAE – with truncations (`compute_gae`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.060 |     0.004 |             0.139 |             0.013 |       2.3x |       3.6x |
|      128 |    1024 |      0.060 |     0.004 |             0.146 |             0.020 |       2.5x |       4.8x |
|      256 |    1024 |      0.058 |     0.005 |             0.147 |             0.027 |       2.5x |       5.3x |
|      512 |    2048 |      0.061 |     0.012 |             0.147 |             0.080 |       2.4x |       6.9x |
|      512 |    4096 |      0.078 |     0.031 |             0.231 |             0.179 |       2.9x |       5.8x |
|      512 |     128 |      0.060 |     0.004 |             0.129 |             0.014 |       2.1x |       3.6x |
|      512 |     512 |      0.059 |     0.005 |             0.141 |             0.027 |       2.4x |       5.4x |
|     4096 |     128 |      0.060 |     0.007 |             0.129 |             0.036 |       2.1x |       5.2x |
|     4096 |     512 |      0.065 |     0.017 |             0.194 |             0.148 |       3.0x |       8.6x |
|     4096 |    2048 |      0.117 |     0.073 |             0.620 |             0.581 |       5.3x |       7.9x |
|    16384 |     128 |      0.062 |     0.016 |             0.177 |             0.130 |       2.8x |       7.9x |
|    16384 |     512 |      0.098 |     0.053 |             0.578 |             0.540 |       5.9x |      10.1x |

#### V-Trace (`compute_vtrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.043 |     0.002 |         0.132 |         0.014 |    3.1x |    7.5x |     16.439 |   386.0x |    138.794 |   3258.7x |          0.247 |   562.6x |
|      128 |    1024 |      0.051 |     0.003 |         0.185 |         0.025 |    3.6x |    9.9x |     34.790 |   681.2x |   1299.201 |  25438.6x |          0.481 |  2700.2x |
|      256 |    1024 |      0.051 |     0.003 |         0.187 |         0.034 |    3.7x |   10.4x |     34.005 |   673.0x |   1319.949 |  26123.1x |          0.755 |  1747.3x |
|      512 |    2048 |      0.052 |     0.010 |         0.175 |         0.092 |    3.4x |    9.4x |     65.335 |  1254.1x |   1299.680 |  24947.8x |          6.004 |   216.5x |
|      512 |    4096 |      0.070 |     0.030 |         0.272 |         0.209 |    3.9x |    7.0x |    129.663 |  1851.1x |   1358.347 |  19391.7x |          6.043 |   224.8x |
|      512 |     128 |      0.051 |     0.002 |         0.161 |         0.015 |    3.2x |    7.4x |      4.336 |    85.2x |   1281.041 |  25177.7x |          0.323 |  3969.9x |
|      512 |     512 |      0.052 |     0.003 |         0.168 |         0.029 |    3.3x |    9.5x |     16.457 |   319.4x |   1300.053 |  25234.0x |          0.769 |  1690.3x |
|     4096 |     128 |      0.050 |     0.005 |         0.160 |         0.042 |    3.2x |    9.0x |      4.335 |    86.2x |   1283.884 |  25538.7x |          1.287 |   997.6x |
|     4096 |     512 |      0.055 |     0.016 |         0.223 |         0.162 |    4.1x |   10.3x |     16.511 |   301.2x |   1279.622 |  23344.0x |          5.990 |   213.6x |
|     4096 |    2048 |      0.105 |     0.069 |         0.713 |         0.666 |    6.8x |    9.7x |     65.132 |   620.0x |   1640.494 |  15615.4x |         32.484 |    50.5x |
|    16384 |     128 |      0.054 |     0.015 |         0.215 |         0.155 |    3.9x |   10.1x |      4.326 |    79.4x |   1317.060 |  24168.0x |          3.637 |   362.2x |
|    16384 |     512 |      0.095 |     0.056 |         0.640 |         0.589 |    6.7x |   10.5x |     16.513 |   174.0x |   1540.332 |  16229.1x |         49.368 |    31.2x |

#### V-Trace – with truncations (`compute_vtrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.053 |     0.002 |             0.168 |             0.015 |       3.2x |       6.7x |
|      128 |    1024 |      0.053 |     0.003 |             0.185 |             0.025 |       3.5x |       8.5x |
|      256 |    1024 |      0.052 |     0.004 |             0.186 |             0.034 |       3.5x |       9.0x |
|      512 |    2048 |      0.057 |     0.015 |             0.181 |             0.092 |       3.2x |       6.0x |
|      512 |    4096 |      0.077 |     0.035 |             0.272 |             0.209 |       3.5x |       5.9x |
|      512 |     128 |      0.052 |     0.002 |             0.154 |             0.015 |       2.9x |       6.6x |
|      512 |     512 |      0.052 |     0.003 |             0.168 |             0.029 |       3.2x |       8.5x |
|     4096 |     128 |      0.052 |     0.005 |             0.154 |             0.040 |       2.9x |       7.7x |
|     4096 |     512 |      0.062 |     0.021 |             0.224 |             0.163 |       3.6x |       7.9x |
|     4096 |    2048 |      0.169 |     0.131 |             0.712 |             0.665 |       4.2x |       5.1x |
|    16384 |     128 |      0.061 |     0.020 |             0.207 |             0.151 |       3.4x |       7.6x |
|    16384 |     512 |      0.112 |     0.072 |             0.640 |             0.594 |       5.7x |       8.3x |

#### Retrace(λ) (`compute_retrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.050 |     0.004 |         0.069 |         0.006 |    1.4x |    1.4x |     16.856 |   338.7x |     16.942 |    340.5x |          0.374 |    45.3x |
|      128 |    1024 |      0.059 |     0.005 |         0.089 |         0.010 |    1.5x |    1.8x |     31.822 |   542.2x |     68.853 |   1173.2x |          0.777 |    88.6x |
|      256 |    1024 |      0.058 |     0.007 |         0.088 |         0.012 |    1.5x |    1.8x |     31.853 |   544.8x |    135.959 |   2325.5x |          1.251 |   108.7x |
|      512 |    2048 |      0.089 |     0.042 |         0.089 |         0.037 |    1.0x |    0.9x |     63.558 |   715.2x |    544.392 |   6126.1x |          3.896 |   139.7x |
|      512 |    4096 |      0.325 |     0.258 |         0.113 |         0.072 |    0.3x |    0.3x |    126.743 |   390.5x |   1110.912 |   3422.3x |          8.882 |   125.1x |
|      512 |     128 |      0.059 |     0.003 |         0.090 |         0.011 |    1.5x |    3.4x |      4.324 |    73.4x |     34.279 |    581.9x |          0.577 |    59.4x |
|      512 |     512 |      0.059 |     0.007 |         0.090 |         0.014 |    1.5x |    2.0x |     16.714 |   284.3x |    135.430 |   2303.9x |          1.421 |    95.3x |
|     4096 |     128 |      0.060 |     0.011 |         0.104 |         0.052 |    1.7x |    4.7x |      4.322 |    72.5x |    274.393 |   4600.2x |          2.500 |   109.7x |
|     4096 |     512 |      0.113 |     0.067 |         0.130 |         0.088 |    1.1x |    1.3x |     16.726 |   147.4x |   1136.448 |  10018.1x |          9.307 |   122.1x |
|     4096 |    2048 |      0.348 |     0.304 |         0.308 |         0.265 |    0.9x |    0.9x |     66.490 |   191.1x |   4541.239 |  13054.3x |         60.701 |    74.8x |
|    16384 |     128 |      0.091 |     0.044 |         0.241 |         0.199 |    2.6x |    4.6x |      4.324 |    47.5x |   1113.778 |  12246.9x |          8.478 |   131.4x |
|    16384 |     512 |      0.291 |     0.246 |         0.354 |         0.314 |    1.2x |    1.3x |     16.761 |    57.7x |   4697.554 |  16160.1x |         61.580 |    76.3x |

#### λ-returns (`compute_lambda_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.035 |     0.002 |         0.102 |         0.011 |    2.9x |    6.5x |     40.679 |  1168.4x |      2.672 |     76.7x |
|      128 |    1024 |      0.039 |     0.002 |         0.125 |         0.019 |    3.2x |    8.9x |     81.274 |  2071.6x |      5.987 |    152.6x |
|      256 |    1024 |      0.039 |     0.003 |         0.126 |         0.026 |    3.2x |    9.7x |     81.039 |  2058.9x |      7.451 |    189.3x |
|      512 |    2048 |      0.041 |     0.008 |         0.127 |         0.076 |    3.1x |    9.4x |    162.375 |  3939.6x |     32.858 |    797.2x |
|      512 |    4096 |      0.045 |     0.015 |         0.215 |         0.172 |    4.8x |   11.5x |    324.702 |  7201.5x |     63.366 |   1405.4x |
|      512 |     128 |      0.041 |     0.002 |         0.106 |         0.011 |    2.6x |    6.3x |     10.107 |   248.3x |      1.272 |     31.3x |
|      512 |     512 |      0.040 |     0.002 |         0.117 |         0.023 |    2.9x |    9.5x |     40.410 |  1000.7x |      5.359 |    132.7x |
|     4096 |     128 |      0.040 |     0.004 |         0.107 |         0.034 |    2.7x |    8.0x |     10.133 |   256.2x |     10.760 |    272.1x |
|     4096 |     512 |      0.041 |     0.008 |         0.173 |         0.134 |    4.2x |   16.4x |     40.461 |   990.9x |     48.438 |   1186.3x |
|     4096 |    2048 |      0.089 |     0.060 |         0.595 |         0.556 |    6.7x |    9.3x |    161.729 |  1809.5x |    243.189 |   2721.0x |
|    16384 |     128 |      0.043 |     0.012 |         0.161 |         0.123 |    3.8x |   10.4x |     10.166 |   238.5x |     64.309 |   1508.8x |
|    16384 |     512 |      0.065 |     0.034 |         0.520 |         0.483 |    8.1x |   14.3x |     40.349 |   624.8x |    216.111 |   3346.6x |

#### λ-returns – with truncations (`compute_lambda_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.042 |     0.002 |             0.138 |             0.013 |       3.3x |       6.4x |
|      128 |    1024 |      0.043 |     0.003 |             0.149 |             0.020 |       3.4x |       7.6x |
|      256 |    1024 |      0.043 |     0.003 |             0.150 |             0.027 |       3.5x |       8.3x |
|      512 |    2048 |      0.044 |     0.009 |             0.149 |             0.077 |       3.4x |       8.4x |
|      512 |    4096 |      0.059 |     0.026 |             0.229 |             0.176 |       3.9x |       6.7x |
|      512 |     128 |      0.042 |     0.002 |             0.115 |             0.013 |       2.7x |       6.5x |
|      512 |     512 |      0.042 |     0.003 |             0.138 |             0.027 |       3.3x |       8.8x |
|     4096 |     128 |      0.043 |     0.005 |             0.114 |             0.034 |       2.7x |       7.4x |
|     4096 |     512 |      0.048 |     0.015 |             0.189 |             0.145 |       4.0x |       9.6x |
|     4096 |    2048 |      0.099 |     0.068 |             0.610 |             0.571 |       6.2x |       8.4x |
|    16384 |     128 |      0.045 |     0.013 |             0.168 |             0.125 |       3.7x |       9.6x |
|    16384 |     512 |      0.084 |     0.052 |             0.567 |             0.530 |       6.8x |      10.1x |

#### Discounted returns (`compute_discounted_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.033 |     0.002 |         0.093 |         0.010 |    2.8x |    6.0x |     24.336 |   741.2x |      1.698 |     51.7x |
|      128 |    1024 |      0.037 |     0.002 |         0.114 |         0.016 |    3.1x |    7.5x |     48.511 |  1310.3x |      3.890 |    105.1x |
|      256 |    1024 |      0.037 |     0.003 |         0.116 |         0.024 |    3.1x |    8.2x |     49.211 |  1318.9x |      5.015 |    134.4x |
|      512 |    2048 |      0.038 |     0.008 |         0.116 |         0.067 |    3.1x |    8.7x |     97.713 |  2605.4x |     22.484 |    599.5x |
|      512 |    4096 |      0.043 |     0.014 |         0.201 |         0.160 |    4.7x |   11.7x |    195.480 |  4575.9x |     44.187 |   1034.3x |
|      512 |     128 |      0.037 |     0.002 |         0.097 |         0.009 |    2.6x |    5.4x |      6.110 |   164.4x |      0.841 |     22.6x |
|      512 |     512 |      0.037 |     0.002 |         0.107 |         0.021 |    2.9x |    8.3x |     24.341 |   662.0x |      3.967 |    107.9x |
|     4096 |     128 |      0.036 |     0.004 |         0.099 |         0.031 |    2.7x |    7.4x |      6.090 |   167.1x |     10.097 |    277.0x |
|     4096 |     512 |      0.038 |     0.010 |         0.159 |         0.121 |    4.2x |   12.7x |     24.207 |   636.2x |     36.817 |    967.6x |
|     4096 |    2048 |      0.085 |     0.057 |         0.546 |         0.509 |    6.5x |    8.9x |     96.858 |  1145.2x |    194.196 |   2296.1x |
|    16384 |     128 |      0.040 |     0.012 |         0.148 |         0.111 |    3.7x |    9.6x |      6.096 |   154.0x |     48.930 |   1236.1x |
|    16384 |     512 |      0.065 |     0.039 |         0.469 |         0.435 |    7.2x |   11.2x |     24.341 |   375.3x |    151.423 |   2334.5x |

#### Discounted returns – with truncations (`compute_discounted_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | compile(vec-trunc) device (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.039 |     0.002 |             0.116 |             0.010 |       3.0x |       5.6x |
|      128 |    1024 |      0.038 |     0.003 |             0.125 |             0.018 |       3.2x |       6.8x |
|      256 |    1024 |      0.039 |     0.003 |             0.124 |             0.025 |       3.1x |       7.1x |
|      512 |    2048 |      0.040 |     0.009 |             0.124 |             0.069 |       3.1x |       7.7x |
|      512 |    4096 |      0.051 |     0.021 |             0.208 |             0.164 |       4.1x |       7.9x |
|      512 |     128 |      0.039 |     0.002 |             0.104 |             0.010 |       2.7x |       5.3x |
|      512 |     512 |      0.039 |     0.003 |             0.116 |             0.022 |       3.0x |       7.4x |
|     4096 |     128 |      0.038 |     0.005 |             0.105 |             0.031 |       2.7x |       6.5x |
|     4096 |     512 |      0.043 |     0.014 |             0.169 |             0.126 |       3.9x |       9.1x |
|     4096 |    2048 |      0.093 |     0.066 |             0.562 |             0.525 |       6.1x |       8.0x |
|    16384 |     128 |      0.042 |     0.013 |             0.153 |             0.112 |       3.6x |       8.9x |
|    16384 |     512 |      0.079 |     0.051 |             0.495 |             0.460 |       6.3x |       8.9x |

#### Eligibility traces (`compute_eligibility_traces`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.028 |     0.002 |         0.076 |         0.007 |    2.7x |    4.8x |     24.395 |   869.3x |      1.698 |     60.5x |
|      128 |    1024 |      0.033 |     0.002 |         0.096 |         0.010 |    2.9x |    5.4x |     48.383 |  1472.2x |      3.924 |    119.4x |
|      256 |    1024 |      0.033 |     0.002 |         0.095 |         0.013 |    2.9x |    5.5x |     48.613 |  1469.2x |      5.112 |    154.5x |
|      512 |    2048 |      0.034 |     0.004 |         0.096 |         0.028 |    2.9x |    7.4x |     96.616 |  2878.2x |     22.274 |    663.5x |
|      512 |    4096 |      0.033 |     0.007 |         0.103 |         0.057 |    3.1x |    8.7x |    193.744 |  5804.9x |     45.992 |   1378.0x |
|      512 |     128 |      0.031 |     0.002 |         0.079 |         0.006 |    2.5x |    3.6x |      6.066 |   193.0x |      0.823 |     26.2x |
|      512 |     512 |      0.031 |     0.002 |         0.088 |         0.011 |    2.8x |    5.1x |     24.251 |   774.1x |      3.611 |    115.3x |
|     4096 |     128 |      0.032 |     0.004 |         0.079 |         0.014 |    2.5x |    3.3x |      6.058 |   189.3x |      6.756 |    211.1x |
|     4096 |     512 |      0.033 |     0.007 |         0.088 |         0.042 |    2.7x |    6.1x |     24.220 |   735.5x |     38.858 |   1180.1x |
|     4096 |    2048 |      0.050 |     0.026 |         0.254 |         0.221 |    5.1x |    8.6x |     97.387 |  1957.1x |    180.819 |   3633.8x |
|    16384 |     128 |      0.036 |     0.012 |         0.079 |         0.040 |    2.2x |    3.4x |      6.069 |   168.1x |     47.489 |   1315.6x |
|    16384 |     512 |      0.053 |     0.029 |         0.217 |         0.183 |    4.1x |    6.3x |     24.385 |   459.3x |    153.260 |   2886.9x |

#### Episodic prefix sum (`compute_episodic_prefix_sum`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------------:|:-------------------:|:---------------:|
|       64 |     512 |      0.029 |     0.002 |         0.078 |         0.007 |    2.7x |    4.8x |
|      128 |    1024 |      0.034 |     0.002 |         0.106 |         0.014 |    3.1x |    7.4x |
|      256 |    1024 |      0.034 |     0.002 |         0.106 |         0.019 |    3.1x |    8.0x |
|      512 |    2048 |      0.035 |     0.004 |         0.106 |         0.050 |    3.0x |   13.5x |
|      512 |    4096 |      0.034 |     0.007 |         0.142 |         0.100 |    4.1x |   15.1x |
|      512 |     128 |      0.034 |     0.002 |         0.088 |         0.007 |    2.6x |    4.2x |
|      512 |     512 |      0.033 |     0.002 |         0.098 |         0.016 |    2.9x |    7.4x |
|     4096 |     128 |      0.034 |     0.004 |         0.088 |         0.019 |    2.6x |    4.5x |
|     4096 |     512 |      0.034 |     0.007 |         0.113 |         0.074 |    3.3x |   11.0x |
|     4096 |    2048 |      0.050 |     0.026 |         0.370 |         0.336 |    7.4x |   13.1x |
|    16384 |     128 |      0.036 |     0.012 |         0.095 |         0.057 |    2.6x |    4.9x |
|    16384 |     512 |      0.053 |     0.029 |         0.309 |         0.274 |    5.8x |    9.3x |

#### Production regime — seq_len [80,128] × num_envs [4096..38400], all algorithms (plus one boundary-marker row, num_envs=16384/seq_len=16)

| algo | num_envs | seq_len | triton full-call (ms) | triton device (ms) | triton amortized (ms) | compile(vec) full-call (ms) | compile(vec) device (ms) | vs vec (full-call) | vs vec (device) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GAE                  |     4096 |      80 |     0.0401 |     0.0042 |     0.0249 |        0.1404 |        0.0236 |   3.50x |   5.55x |
| GAE                  |     8192 |      80 |     0.0410 |     0.0067 |     0.0248 |        0.1468 |        0.0358 |   3.58x |   5.34x |
| GAE                  |    16384 |      80 |     0.0424 |     0.0117 |     0.0241 |        0.1482 |        0.0637 |   3.49x |   5.43x |
| GAE                  |    32768 |      80 |     0.0524 |     0.0218 |     0.0250 |        0.1848 |        0.1294 |   3.52x |   5.95x |
| GAE                  |    38400 |      80 |     0.0557 |     0.0253 |     0.0265 |        0.2047 |        0.1509 |   3.67x |   5.97x |
| GAE                  |     4096 |     128 |     0.0399 |     0.0043 |     0.0244 |        0.1410 |        0.0384 |   3.53x |   8.94x |
| GAE                  |     8192 |     128 |     0.0405 |     0.0068 |     0.0247 |        0.1478 |        0.0687 |   3.64x |  10.16x |
| GAE                  |    16384 |     128 |     0.0424 |     0.0119 |     0.0247 |        0.1959 |        0.1363 |   4.62x |  11.42x |
| GAE                  |    32768 |     128 |     0.0522 |     0.0220 |     0.0249 |        0.3092 |        0.2589 |   5.93x |  11.78x |
| GAE                  |    38400 |     128 |     0.0555 |     0.0254 |     0.0266 |        0.3489 |        0.3004 |   6.28x |  11.84x |
| GAE                  |    16384 |      16 |     0.0421 |     0.0113 |     0.0247 |        0.1425 |        0.0210 |   3.38x |   1.86x | ⚠️
| V-Trace              |     4096 |      80 |     0.0512 |     0.0046 |     0.0319 |        0.1803 |        0.0305 |   3.52x |   6.70x |
| V-Trace              |     8192 |      80 |     0.0503 |     0.0070 |     0.0320 |        0.1917 |        0.0488 |   3.81x |   6.95x |
| V-Trace              |    16384 |      80 |     0.0519 |     0.0122 |     0.0317 |        0.1901 |        0.0918 |   3.66x |   7.50x |
| V-Trace              |    32768 |      80 |     0.0618 |     0.0227 |     0.0315 |        0.2459 |        0.1813 |   3.98x |   7.98x |
| V-Trace              |    38400 |      80 |     0.0646 |     0.0262 |     0.0317 |        0.2740 |        0.2106 |   4.24x |   8.04x |
| V-Trace              |     4096 |     128 |     0.0509 |     0.0046 |     0.0322 |        0.1716 |        0.0432 |   3.37x |   9.33x |
| V-Trace              |     8192 |     128 |     0.0506 |     0.0071 |     0.0316 |        0.1768 |        0.0809 |   3.49x |  11.37x |
| V-Trace              |    16384 |     128 |     0.0543 |     0.0148 |     0.0313 |        0.2266 |        0.1578 |   4.17x |  10.67x |
| V-Trace              |    32768 |     128 |     0.0679 |     0.0292 |     0.0317 |        0.3616 |        0.3022 |   5.33x |  10.34x |
| V-Trace              |    38400 |     128 |     0.0728 |     0.0338 |     0.0352 |        0.4073 |        0.3506 |   5.60x |  10.39x |
| V-Trace              |    16384 |      16 |     0.0523 |     0.0115 |     0.0317 |        0.1832 |        0.0276 |   3.51x |   2.40x | ⚠️
| Retrace              |     4096 |      80 |     0.0604 |     0.0102 |     0.0410 |        0.0796 |        0.0136 |   1.32x |   1.33x |
| Retrace              |     8192 |      80 |     0.0681 |     0.0201 |     0.0413 |        0.0871 |        0.0232 |   1.28x |   1.15x |
| Retrace              |    16384 |      80 |     0.0858 |     0.0384 |     0.0410 |        0.0885 |        0.0447 |   1.03x |   1.16x |
| Retrace              |    32768 |      80 |     0.1192 |     0.0723 |     0.0733 |        0.1246 |        0.0852 |   1.05x |   1.18x |
| Retrace              |    38400 |      80 |     0.1314 |     0.0841 |     0.0849 |        0.1387 |        0.0993 |   1.06x |   1.18x |
| Retrace              |     4096 |     128 |     0.0601 |     0.0110 |     0.0411 |        0.0802 |        0.0163 |   1.34x |   1.48x |
| Retrace              |     8192 |     128 |     0.0720 |     0.0239 |     0.0410 |        0.0871 |        0.0317 |   1.21x |   1.33x |
| Retrace              |    16384 |     128 |     0.0905 |     0.0438 |     0.0451 |        0.1032 |        0.0641 |   1.14x |   1.46x |
| Retrace              |    32768 |     128 |     0.1303 |     0.0832 |     0.0844 |        0.1588 |        0.1195 |   1.22x |   1.44x |
| Retrace              |    38400 |     128 |     0.1434 |     0.0963 |     0.0975 |        0.1765 |        0.1380 |   1.23x |   1.43x |
| Retrace              |    16384 |      16 |     0.0615 |     0.0120 |     0.0414 |        0.0901 |        0.0120 |   1.46x |   1.00x | ⚠️
| lambda-returns       |     4096 |      80 |     0.0398 |     0.0042 |     0.0252 |        0.1285 |        0.0224 |   3.23x |   5.28x |
| lambda-returns       |     8192 |      80 |     0.0402 |     0.0067 |     0.0250 |        0.1375 |        0.0356 |   3.42x |   5.31x |
| lambda-returns       |    16384 |      80 |     0.0430 |     0.0117 |     0.0251 |        0.1363 |        0.0625 |   3.17x |   5.34x |
| lambda-returns       |    32768 |      80 |     0.0521 |     0.0218 |     0.0254 |        0.1824 |        0.1274 |   3.50x |   5.86x |
| lambda-returns       |    38400 |      80 |     0.0562 |     0.0252 |     0.0265 |        0.2029 |        0.1488 |   3.61x |   5.90x |
| lambda-returns       |     4096 |     128 |     0.0407 |     0.0043 |     0.0254 |        0.1298 |        0.0371 |   3.19x |   8.71x |
| lambda-returns       |     8192 |     128 |     0.0407 |     0.0067 |     0.0254 |        0.1364 |        0.0658 |   3.35x |   9.78x |
| lambda-returns       |    16384 |     128 |     0.0428 |     0.0118 |     0.0253 |        0.1876 |        0.1321 |   4.38x |  11.17x |
| lambda-returns       |    32768 |     128 |     0.0529 |     0.0219 |     0.0252 |        0.3006 |        0.2486 |   5.68x |  11.35x |
| lambda-returns       |    38400 |     128 |     0.0554 |     0.0253 |     0.0267 |        0.3356 |        0.2865 |   6.06x |  11.32x |
| lambda-returns       |    16384 |      16 |     0.0424 |     0.0113 |     0.0252 |        0.1425 |        0.0206 |   3.36x |   1.82x | ⚠️
| discounted-returns   |     4096 |      80 |     0.0364 |     0.0042 |     0.0227 |        0.1171 |        0.0202 |   3.21x |   4.82x |
| discounted-returns   |     8192 |      80 |     0.0382 |     0.0067 |     0.0228 |        0.1253 |        0.0318 |   3.28x |   4.78x |
| discounted-returns   |    16384 |      80 |     0.0392 |     0.0116 |     0.0224 |        0.1234 |        0.0538 |   3.15x |   4.63x |
| discounted-returns   |    32768 |      80 |     0.0492 |     0.0216 |     0.0232 |        0.1622 |        0.1114 |   3.30x |   5.14x |
| discounted-returns   |    38400 |      80 |     0.0523 |     0.0250 |     0.0263 |        0.1795 |        0.1298 |   3.43x |   5.19x |
| discounted-returns   |     4096 |     128 |     0.0373 |     0.0042 |     0.0230 |        0.1180 |        0.0341 |   3.16x |   8.14x |
| discounted-returns   |     8192 |     128 |     0.0376 |     0.0067 |     0.0225 |        0.1236 |        0.0590 |   3.28x |   8.84x |
| discounted-returns   |    16384 |     128 |     0.0397 |     0.0116 |     0.0225 |        0.1710 |        0.1189 |   4.31x |  10.26x |
| discounted-returns   |    32768 |     128 |     0.0497 |     0.0219 |     0.0231 |        0.2663 |        0.2209 |   5.36x |  10.11x |
| discounted-returns   |    38400 |     128 |     0.0529 |     0.0251 |     0.0266 |        0.2999 |        0.2550 |   5.67x |  10.14x |
| discounted-returns   |    16384 |      16 |     0.0396 |     0.0113 |     0.0226 |        0.1166 |        0.0164 |   2.94x |   1.46x | ⚠️
| eligibility-traces   |     4096 |      80 |     0.0320 |     0.0042 |     0.0212 |        0.0865 |        0.0150 |   2.70x |   3.55x |
| eligibility-traces   |     8192 |      80 |     0.0334 |     0.0067 |     0.0211 |        0.0942 |        0.0241 |   2.82x |   3.60x |
| eligibility-traces   |    16384 |      80 |     0.0358 |     0.0117 |     0.0217 |        0.0940 |        0.0421 |   2.63x |   3.61x |
| eligibility-traces   |    32768 |      80 |     0.0459 |     0.0216 |     0.0226 |        0.1220 |        0.0871 |   2.66x |   4.04x |
| eligibility-traces   |    38400 |      80 |     0.0493 |     0.0249 |     0.0261 |        0.1380 |        0.1032 |   2.80x |   4.14x |
| eligibility-traces   |     4096 |     128 |     0.0330 |     0.0042 |     0.0211 |        0.0786 |        0.0138 |   2.38x |   3.28x |
| eligibility-traces   |     8192 |     128 |     0.0327 |     0.0067 |     0.0213 |        0.0854 |        0.0229 |   2.61x |   3.42x |
| eligibility-traces   |    16384 |     128 |     0.0356 |     0.0116 |     0.0207 |        0.0849 |        0.0414 |   2.38x |   3.56x |
| eligibility-traces   |    32768 |     128 |     0.0450 |     0.0217 |     0.0229 |        0.1187 |        0.0850 |   2.64x |   3.92x |
| eligibility-traces   |    38400 |     128 |     0.0492 |     0.0253 |     0.0266 |        0.1323 |        0.0985 |   2.69x |   3.89x |
| eligibility-traces   |    16384 |      16 |     0.0352 |     0.0113 |     0.0210 |        0.0737 |        0.0081 |   2.10x |   0.72x | ⚠️
| prefix-sum           |     4096 |      80 |     0.0343 |     0.0042 |     0.0209 |        0.0906 |        0.0149 |   2.64x |   3.51x |
| prefix-sum           |     8192 |      80 |     0.0353 |     0.0067 |     0.0210 |        0.0979 |        0.0241 |   2.78x |   3.59x |
| prefix-sum           |    16384 |      80 |     0.0367 |     0.0116 |     0.0212 |        0.0992 |        0.0419 |   2.70x |   3.60x |
| prefix-sum           |    32768 |      80 |     0.0473 |     0.0216 |     0.0227 |        0.1238 |        0.0867 |   2.62x |   4.01x |
| prefix-sum           |    38400 |      80 |     0.0507 |     0.0250 |     0.0263 |        0.1393 |        0.1022 |   2.75x |   4.09x |
| prefix-sum           |     4096 |     128 |     0.0353 |     0.0042 |     0.0211 |        0.0824 |        0.0131 |   2.34x |   3.10x |
| prefix-sum           |     8192 |     128 |     0.0342 |     0.0067 |     0.0206 |        0.0887 |        0.0229 |   2.59x |   3.42x |
| prefix-sum           |    16384 |     128 |     0.0367 |     0.0116 |     0.0209 |        0.0883 |        0.0416 |   2.40x |   3.58x |
| prefix-sum           |    32768 |     128 |     0.0476 |     0.0218 |     0.0232 |        0.1211 |        0.0853 |   2.54x |   3.92x |
| prefix-sum           |    38400 |     128 |     0.0505 |     0.0252 |     0.0267 |        0.1345 |        0.0990 |   2.66x |   3.93x |
| prefix-sum           |    16384 |      16 |     0.0364 |     0.0113 |     0.0210 |        0.0778 |        0.0080 |   2.14x |   0.71x | ⚠️

*⚠️ marks the boundary-marker row (num_envs=16384, seq_len=16). **vs vec (full-call)** is the headline ratio — the complete `compute_*(tensors) -> tensors` call including launch/wrapper overhead, which a caller pays every invocation. **vs vec (device)** is a diagnostic showing the same ratio for CUDA-kernel-only time; where full-call and device speedups diverge, the gap is launch + wrapper overhead. **triton amortized** is N calls timed inside one region (separates harness per-call sync overhead from genuine per-call cost) — reported alongside, not used for any ratio.*
