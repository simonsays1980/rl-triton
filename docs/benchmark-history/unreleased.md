# Benchmarks archive: unreleased

## unreleased – 2026-07-25 – NVIDIA H100 80GB HBM3

*Measured on NVIDIA H100 80GB HBM3 · 2026-07-25 · [`triton`](https://github.com/openai/triton) kernels vs `torch.compile` baselines and NumPy CPU.*

**Configuration.**  dtype float32 (all kernels require it; see NOTES.md on bf16 and autocast).  gamma=0.99, lambda=0.95 (lambda=0.9 for eligibility traces).  Termination probability ~5% per step; truncation-path tables additionally inject ~5% interior truncated steps (mutually exclusive with terminations) with populated `bootstrap_values`.

**Methodology.**  All GPU full-call timings use CUDA events (start/stop around the complete `compute_*(tensors) -> tensors` call, explicit sync immediately before start); reported value is the min-of-medians across 5 independent trials to filter clock-state noise. Every config is warmed up at its exact shape (20 untimed calls) before any timed call, so `torch.compile` JIT/autotuning and Triton kernel compilation never land in the timed region. A tolerance-based correctness gate (atol=rtol=1e-4 vs. a sequential reference implementation) runs before every timed config — not bit-identical, since `tl.associative_scan` reorders float ops depending on num_warps/block layout, so cross-config last-bit differences are legitimate. A monotonicity gate (2% band) then asserts a larger problem never measures faster than a smaller one along either swept axis. CPU timings are wall-clock (perf_counter), run until at least 0.5 s of samples.

**Two timing granularities.**  **triton** (headline) is full-call wall time — what a caller pays every invocation, including launch overhead and wrapper setup (HAS_TRUNCATIONS/HAS_BOOTSTRAP dispatch, allocation, layout). All speedup ratios are computed from this number. **dev** is device-only CUDA time (`torch.profiler` CUDA activity around steady-state calls, ncu/nsys being unavailable in typical containerized GPU environments) — a diagnostic showing pure kernel execution time; where dev is much smaller than the full-call number, the gap is launch + wrapper overhead the caller still pays. The production-regime table additionally reports an **amortized** variant (N calls in one timed region) for its short-seq_len rows, to separate harness per-call sync overhead from genuine per-call cost — the single-call full-call number remains the ratio basis throughout.

**Columns.**  **triton**: full-call wall time, headline (CUDA events).  **dev**: device-only kernel time, diagnostic (see above).  **compile(vec)**: `torch.compile` applied to the fastest hand-vectorized PyTorch equivalent – cumsum / suffix-product implementations with no Python loop; this is the strongest GPU baseline a competent engineer would write.  **compile(assoc)**: `torch.compile` applied to the unified associative-scan implementation that also handles truncations – called here with zero truncations to show the cost of a single regime-agnostic implementation vs the specialized no-truncation path (GAE, V-Trace, λ-returns, discounted returns only).  **compile(vec-trunc)**: `torch.compile` of the vectorized truncation baseline, used in the with-truncations tables (itself asserted correct against the sequential truncation reference before being trusted as a baseline).  **loop (gpu)**: uncompiled sequential Python loop dispatching GPU ops – the pattern used by CleanRL, RLlib, and most RL codebases today; no `torch.compile`, no vectorization; wall-clock timing.  **np→triton→np**: end-to-end wall-clock for the NumPy adoption path (CPU → GPU transfer, kernel, GPU → CPU transfer).  **numpy cpu**: sequential NumPy loop on CPU – same algorithm as the kernel, no GPU; establishes the CPU reference for each algorithm.  Headline tables below show 4 representative sizes per algorithm (small/parity, mid, main-grid-large, production-adjacent-large); the full CONFIGS grid is reproducible via `python tests/bench_release.py`.

#### GAE (`compute_gae`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | vs vec (full-call) | vs vec (device) | compile(assoc) (ms) | vs assoc | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|:-------------------:|:--------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.053 |     0.002 |         0.083 |    1.6x |    2.4x |          0.174 |      3.3x |     65.667 |  1241.4x |     25.956 |    490.7x |          0.214 |   121.1x |
|      256 |    1024 |      0.061 |     0.002 |         0.109 |    1.8x |    2.6x |          0.231 |      3.8x |     84.018 |  1368.2x |     86.812 |   1413.7x |          0.521 |   166.5x |
|      512 |    4096 |      0.064 |     0.016 |         0.108 |    1.7x |    2.0x |          0.305 |      4.8x |    522.260 |  8205.4x |    199.062 |   3127.6x |          2.290 |    86.9x |
|    16384 |     512 |      0.070 |     0.045 |         0.215 |    3.1x |    4.0x |          0.672 |      9.6x |     38.003 |   543.8x |    729.286 |  10435.1x |         26.663 |    27.4x |


#### GAE – with truncations (`compute_gae`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.052 |     0.003 |             0.137 |       2.6x |       4.3x |
|      256 |    1024 |      0.052 |     0.004 |             0.130 |       2.5x |       5.6x |
|      512 |    4096 |      0.098 |     0.035 |             0.266 |       2.7x |       6.2x |
|    16384 |     512 |      0.110 |     0.070 |             0.675 |       6.1x |       9.1x |


#### V-Trace (`compute_vtrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | vs vec (full-call) | vs vec (device) | compile(assoc) (ms) | vs assoc | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|:-------------------:|:--------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.075 |     0.002 |         0.152 |    2.0x |    4.1x |          0.232 |      3.1x |     23.611 |   316.7x |     65.125 |    873.5x |          0.332 |   196.1x |
|      256 |    1024 |      0.078 |     0.003 |         0.176 |    2.3x |    4.1x |          0.302 |      3.9x |     52.662 |   677.5x |     84.654 |   1089.1x |          1.298 |    65.2x |
|      512 |    4096 |      0.091 |     0.033 |         0.166 |    1.8x |    2.4x |          0.360 |      4.0x |    186.531 |  2060.5x |    302.565 |   3342.2x |          5.699 |    53.1x |
|    16384 |     512 |      0.109 |     0.078 |         0.419 |    3.8x |    4.9x |          0.805 |      7.4x |     15.214 |   139.4x |    819.737 |   7510.1x |         53.986 |    15.2x |


#### V-Trace – with truncations (`compute_vtrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.043 |     0.002 |             0.147 |       3.4x |       6.8x |
|      256 |    1024 |      0.077 |     0.004 |             0.280 |       3.6x |       8.9x |
|      512 |    4096 |      0.139 |     0.041 |             0.361 |       2.6x |       6.3x |
|    16384 |     512 |      0.158 |     0.098 |             0.843 |       5.3x |       7.7x |


#### Retrace(λ) (`compute_retrace`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy | np→triton→np (ms) | e2e vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|:-----------------:|:------------:|
|       64 |     512 |      0.092 |     0.004 |         0.124 |    1.3x |    1.3x |     26.703 |   289.6x |     31.087 |    337.2x |          0.611 |    50.8x |
|      256 |    1024 |      0.090 |     0.007 |         0.138 |    1.5x |    1.7x |     47.076 |   520.9x |    218.643 |   2419.5x |          2.189 |    99.9x |
|      512 |    4096 |      0.653 |     0.284 |         0.163 |    0.3x |    0.3x |    243.830 |   373.2x |   1757.538 |   2689.8x |         12.918 |   136.1x |
|    16384 |     512 |      0.311 |     0.271 |         0.417 |    1.3x |    1.4x |     14.614 |    47.0x |   4310.713 |  13856.2x |         50.497 |    85.4x |


#### λ-returns (`compute_lambda_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | vs vec (full-call) | vs vec (device) | compile(assoc) (ms) | vs assoc | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|:-------------------:|:--------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.061 |     0.001 |         0.095 |    1.5x |    2.3x |          0.177 |      2.9x |     55.985 |   916.9x |      4.351 |     71.3x |
|      256 |    1024 |      0.036 |     0.002 |         0.060 |    1.7x |    2.4x |          0.125 |      3.5x |     72.599 |  2025.6x |      7.661 |    213.8x |
|      512 |    4096 |      0.044 |     0.015 |         0.067 |    1.5x |    2.0x |          0.261 |      5.9x |    288.782 |  6568.0x |     70.238 |   1597.5x |
|    16384 |     512 |      0.074 |     0.046 |         0.205 |    2.8x |    3.7x |          0.662 |      9.0x |     34.577 |   468.4x |    332.724 |   4507.0x |


#### λ-returns – with truncations (`compute_lambda_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.042 |     0.002 |             0.134 |       3.2x |       7.1x |
|      256 |    1024 |      0.041 |     0.003 |             0.141 |       3.4x |       8.1x |
|      512 |    4096 |      0.063 |     0.031 |             0.261 |       4.1x |       6.9x |
|    16384 |     512 |      0.098 |     0.067 |             0.664 |       6.8x |       9.2x |


#### Discounted returns (`compute_discounted_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | vs vec (full-call) | vs vec (device) | compile(assoc) (ms) | vs assoc | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|:-------------------:|:--------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.059 |     0.001 |         0.092 |    1.6x |    1.9x |          0.169 |      2.9x |     36.247 |   613.9x |      2.844 |     48.2x |
|      256 |    1024 |      0.033 |     0.003 |         0.058 |    1.8x |    1.8x |          0.111 |      3.4x |     46.038 |  1409.1x |      5.205 |    159.3x |
|      512 |    4096 |      0.042 |     0.014 |         0.064 |    1.5x |    1.6x |          0.240 |      5.7x |    192.229 |  4557.8x |     52.891 |   1254.0x |
|    16384 |     512 |      0.066 |     0.041 |         0.158 |    2.4x |    3.0x |          0.597 |      9.1x |     22.876 |   347.2x |    228.800 |   3472.6x |


#### Discounted returns – with truncations (`compute_discounted_returns`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec-trunc) (ms) | vs vec-trunc (full-call) | vs vec-trunc (device) |
|:--------:|:-------:|:----------------------:|:-------------------:|:-----------------------:|:-------------------------:|:----------------------:|
|       64 |     512 |      0.038 |     0.002 |             0.114 |       3.0x |       5.2x |
|      256 |    1024 |      0.040 |     0.003 |             0.125 |       3.2x |       6.6x |
|      512 |    4096 |      0.064 |     0.024 |             0.254 |       4.0x |       8.0x |
|    16384 |     512 |      0.087 |     0.057 |             0.604 |       7.0x |       9.8x |


#### Eligibility traces (`compute_eligibility_traces`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | vs vec (full-call) | vs vec (device) | loop gpu (ms) | vs loop | numpy cpu (ms) | vs numpy |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|:-------------:|:-------:|:--------------:|:--------:|
|       64 |     512 |      0.054 |     0.001 |         0.066 |    1.2x |    1.7x |     36.505 |   672.6x |      2.782 |     51.3x |
|      256 |    1024 |      0.030 |     0.002 |         0.041 |    1.4x |    2.1x |     46.802 |  1552.6x |      5.264 |    174.6x |
|      512 |    4096 |      0.043 |     0.006 |         0.057 |    1.3x |    3.9x |    188.369 |  4419.3x |     46.858 |   1099.3x |
|    16384 |     512 |      0.059 |     0.035 |         0.189 |    3.2x |    4.6x |     23.365 |   393.0x |    229.522 |   3860.4x |


#### Episodic prefix sum (`compute_episodic_prefix_sum`)

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | compile(vec) (ms) | vs vec (full-call) | vs vec (device) |
|:--------:|:-------:|:---------------------:|:-------------------:|:-----------------:|:-------------------:|:---------------:|
|       64 |     512 |      0.033 |     0.001 |         0.041 |    1.3x |    1.0x |
|      256 |    1024 |      0.033 |     0.002 |         0.047 |    1.4x |    1.2x |
|      512 |    4096 |      0.035 |     0.006 |         0.047 |    1.3x |    1.8x |
|    16384 |     512 |      0.059 |     0.035 |         0.120 |    2.0x |    2.6x |


#### Production regime — seq_len [80,128] × num_envs [4096..38400], all algorithms (plus one boundary-marker row, num_envs=16384/seq_len=16)

| algo | num_envs | seq_len | triton full-call (ms) | triton device (ms) | triton amortized (ms) | compile(vec) full-call (ms) | vs vec (full-call) | vs vec (device) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GAE                  |     4096 |      80 |     0.0349 |     0.0039 |     0.0229 |        0.0780 |   2.23x |  10.14x |
| GAE                  |     8192 |      80 |     0.0375 |     0.0064 |     0.0229 |        0.1123 |   3.00x |  11.78x |
| GAE                  |    16384 |      80 |     0.0388 |     0.0113 |     0.0214 |        0.1822 |   4.69x |  12.97x |
| GAE                  |    32768 |      80 |     0.0507 |     0.0214 |     0.0229 |        0.3267 |   6.45x |  13.68x |
| GAE                  |    38400 |      80 |     0.0522 |     0.0248 |     0.0265 |        0.3795 |   7.27x |  13.86x |
| GAE                  |     4096 |     128 |     0.0349 |     0.0040 |     0.0218 |        0.0764 |   2.19x |  10.20x |
| GAE                  |     8192 |     128 |     0.0365 |     0.0064 |     0.0228 |        0.1133 |   3.11x |  11.87x |
| GAE                  |    16384 |     128 |     0.0387 |     0.0114 |     0.0219 |        0.1858 |   4.80x |  13.17x |
| GAE                  |    32768 |     128 |     0.0489 |     0.0229 |     0.0246 |        0.3348 |   6.85x |  13.13x |
| GAE                  |    38400 |     128 |     0.0548 |     0.0267 |     0.0286 |        0.3874 |   7.06x |  13.21x |
| GAE                  |    16384 |      16 |     0.0412 |     0.0110 |     0.0221 |        0.1786 |   4.33x |  13.06x | ⚠️
| V-Trace              |     4096 |      80 |     0.0411 |     0.0043 |     0.0276 |        0.0942 |   2.29x |  11.12x |
| V-Trace              |     8192 |      80 |     0.0425 |     0.0067 |     0.0276 |        0.1252 |   2.95x |  12.80x |
| V-Trace              |    16384 |      80 |     0.0448 |     0.0119 |     0.0301 |        0.2157 |   4.82x |  14.76x |
| V-Trace              |    32768 |      80 |     0.0607 |     0.0256 |     0.0278 |        0.3976 |   6.55x |  14.28x |
| V-Trace              |    38400 |      80 |     0.0620 |     0.0295 |     0.0313 |        0.4583 |   7.39x |  14.47x |
| V-Trace              |     4096 |     128 |     0.0414 |     0.0043 |     0.0293 |        0.0941 |   2.27x |  11.46x |
| V-Trace              |     8192 |     128 |     0.0397 |     0.0068 |     0.0293 |        0.1358 |   3.42x |  14.00x |
| V-Trace              |    16384 |     128 |     0.0532 |     0.0202 |     0.0292 |        0.2412 |   4.53x |  10.11x |
| V-Trace              |    32768 |     128 |     0.0771 |     0.0404 |     0.0423 |        0.4413 |   5.72x |  10.11x |
| V-Trace              |    38400 |     128 |     0.0798 |     0.0468 |     0.0487 |        0.5100 |   6.39x |  10.22x |
| V-Trace              |    16384 |      16 |     0.0506 |     0.0112 |     0.0278 |        0.1895 |   3.75x |  13.63x | ⚠️
| Retrace              |     4096 |      80 |     0.0546 |     0.0098 |     0.0373 |        0.1020 |   1.87x |   4.84x |
| Retrace              |     8192 |      80 |     0.0694 |     0.0228 |     0.0384 |        0.1421 |   2.05x |   4.15x |
| Retrace              |    16384 |      80 |     0.0852 |     0.0416 |     0.0431 |        0.2333 |   2.74x |   4.63x |
| Retrace              |    32768 |      80 |     0.1211 |     0.0792 |     0.0806 |        0.4179 |   3.45x |   4.80x |
| Retrace              |    38400 |      80 |     0.1393 |     0.0913 |     0.0928 |        0.4831 |   3.47x |   4.87x |
| Retrace              |     4096 |     128 |     0.0565 |     0.0119 |     0.0389 |        0.0993 |   1.76x |   4.33x |
| Retrace              |     8192 |     128 |     0.0723 |     0.0282 |     0.0407 |        0.1500 |   2.07x |   3.77x |
| Retrace              |    16384 |     128 |     0.1020 |     0.0525 |     0.0544 |        0.2452 |   2.40x |   3.96x |
| Retrace              |    32768 |     128 |     0.1451 |     0.1009 |     0.1029 |        0.4449 |   3.07x |   4.05x |
| Retrace              |    38400 |     128 |     0.1587 |     0.1173 |     0.1193 |        0.5123 |   3.23x |   4.07x |
| Retrace              |    16384 |      16 |     0.0757 |     0.0117 |     0.0367 |        0.2095 |   2.77x |  13.72x | ⚠️
| lambda-returns       |     4096 |      80 |     0.0369 |     0.0039 |     0.0241 |        0.0743 |   2.01x |   9.41x |
| lambda-returns       |     8192 |      80 |     0.0363 |     0.0064 |     0.0227 |        0.1064 |   2.93x |  10.88x |
| lambda-returns       |    16384 |      80 |     0.0393 |     0.0113 |     0.0238 |        0.1714 |   4.37x |  11.92x |
| lambda-returns       |    32768 |      80 |     0.0510 |     0.0214 |     0.0239 |        0.3052 |   5.99x |  12.56x |
| lambda-returns       |    38400 |      80 |     0.0542 |     0.0248 |     0.0265 |        0.3500 |   6.45x |  12.72x |
| lambda-returns       |     4096 |     128 |     0.0386 |     0.0039 |     0.0241 |        0.0751 |   1.95x |   9.57x |
| lambda-returns       |     8192 |     128 |     0.0367 |     0.0064 |     0.0228 |        0.1081 |   2.94x |  11.02x |
| lambda-returns       |    16384 |     128 |     0.0423 |     0.0114 |     0.0232 |        0.1755 |   4.14x |  12.14x |
| lambda-returns       |    32768 |     128 |     0.0512 |     0.0230 |     0.0245 |        0.3111 |   6.08x |  12.05x |
| lambda-returns       |    38400 |     128 |     0.0545 |     0.0267 |     0.0285 |        0.3574 |   6.56x |  12.18x |
| lambda-returns       |    16384 |      16 |     0.0604 |     0.0110 |     0.0229 |        0.1692 |   2.80x |  11.98x | ⚠️
| discounted-returns   |     4096 |      80 |     0.0359 |     0.0039 |     0.0222 |        0.0581 |   1.62x |   6.29x |
| discounted-returns   |     8192 |      80 |     0.0340 |     0.0063 |     0.0203 |        0.0763 |   2.24x |   7.12x |
| discounted-returns   |    16384 |      80 |     0.0360 |     0.0113 |     0.0211 |        0.1180 |   3.27x |   7.74x |
| discounted-returns   |    32768 |      80 |     0.0480 |     0.0211 |     0.0226 |        0.2025 |   4.22x |   8.37x |
| discounted-returns   |    38400 |      80 |     0.0524 |     0.0247 |     0.0264 |        0.2349 |   4.48x |   8.46x |
| discounted-returns   |     4096 |     128 |     0.0369 |     0.0039 |     0.0211 |        0.0611 |   1.66x |   6.43x |
| discounted-returns   |     8192 |     128 |     0.0348 |     0.0064 |     0.0207 |        0.0789 |   2.27x |   7.28x |
| discounted-returns   |    16384 |     128 |     0.0381 |     0.0113 |     0.0219 |        0.1241 |   3.26x |   8.11x |
| discounted-returns   |    32768 |     128 |     0.0493 |     0.0214 |     0.0230 |        0.2136 |   4.33x |   8.70x |
| discounted-returns   |    38400 |     128 |     0.0515 |     0.0248 |     0.0266 |        0.2436 |   4.73x |   8.82x |
| discounted-returns   |    16384 |      16 |     0.0585 |     0.0110 |     0.0218 |        0.1178 |   2.01x |   7.68x | ⚠️
| eligibility-traces   |     4096 |      80 |     0.0324 |     0.0039 |     0.0208 |        0.0727 |   2.24x |  10.72x |
| eligibility-traces   |     8192 |      80 |     0.0335 |     0.0064 |     0.0196 |        0.1088 |   3.25x |  12.69x |
| eligibility-traces   |    16384 |      80 |     0.0367 |     0.0113 |     0.0210 |        0.1885 |   5.14x |  14.02x |
| eligibility-traces   |    32768 |      80 |     0.0466 |     0.0211 |     0.0226 |        0.3411 |   7.32x |  14.85x |
| eligibility-traces   |    38400 |      80 |     0.0502 |     0.0247 |     0.0264 |        0.3939 |   7.84x |  14.94x |
| eligibility-traces   |     4096 |     128 |     0.0330 |     0.0039 |     0.0194 |        0.0718 |   2.18x |  10.73x |
| eligibility-traces   |     8192 |     128 |     0.0329 |     0.0064 |     0.0199 |        0.1120 |   3.40x |  12.68x |
| eligibility-traces   |    16384 |     128 |     0.0369 |     0.0113 |     0.0202 |        0.1882 |   5.10x |  14.01x |
| eligibility-traces   |    32768 |     128 |     0.0470 |     0.0215 |     0.0233 |        0.3425 |   7.28x |  14.59x |
| eligibility-traces   |    38400 |     128 |     0.0493 |     0.0249 |     0.0267 |        0.3956 |   8.02x |  14.77x |
| eligibility-traces   |    16384 |      16 |     0.0353 |     0.0110 |     0.0203 |        0.1891 |   5.35x |  14.36x | ⚠️
| prefix-sum           |     4096 |      80 |     0.0338 |     0.0039 |     0.0203 |        0.0525 |   1.55x |   5.52x |
| prefix-sum           |     8192 |      80 |     0.0334 |     0.0064 |     0.0200 |        0.0713 |   2.13x |   6.47x |
| prefix-sum           |    16384 |      80 |     0.0369 |     0.0113 |     0.0200 |        0.1091 |   2.96x |   7.12x |
| prefix-sum           |    32768 |      80 |     0.0458 |     0.0211 |     0.0226 |        0.1851 |   4.04x |   7.55x |
| prefix-sum           |    38400 |      80 |     0.0500 |     0.0246 |     0.0263 |        0.2152 |   4.30x |   7.72x |
| prefix-sum           |     4096 |     128 |     0.0334 |     0.0039 |     0.0202 |        0.0507 |   1.52x |   5.55x |
| prefix-sum           |     8192 |     128 |     0.0343 |     0.0063 |     0.0196 |        0.0708 |   2.06x |   6.48x |
| prefix-sum           |    16384 |     128 |     0.0376 |     0.0113 |     0.0206 |        0.1078 |   2.87x |   7.13x |
| prefix-sum           |    32768 |     128 |     0.0468 |     0.0215 |     0.0232 |        0.1914 |   4.09x |   7.69x |
| prefix-sum           |    38400 |     128 |     0.0487 |     0.0249 |     0.0267 |        0.2170 |   4.46x |   7.78x |
| prefix-sum           |    16384 |      16 |     0.0372 |     0.0110 |     0.0199 |        0.1066 |   2.86x |   7.32x | ⚠️

*⚠️ marks the boundary-marker row (num_envs=16384, seq_len=16). **vs vec (full-call)** is the headline ratio — the complete `compute_*(tensors) -> tensors` call including launch/wrapper overhead, which a caller pays every invocation. **vs vec (device)** is a diagnostic showing the same ratio for CUDA-kernel-only time; where full-call and device speedups diverge, the gap is launch + wrapper overhead. **triton amortized** is N calls timed inside one region (separates harness per-call sync overhead from genuine per-call cost) — reported alongside, not used for any ratio.*
