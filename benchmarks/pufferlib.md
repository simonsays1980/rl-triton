# rl-triton vs. PufferLib — GAE advantage kernel

*Measured on NVIDIA H100 80GB HBM3 · 2026-07-25 · PufferLib 3.0.0.*

**One-time study, not a recurring benchmark.** This is a single, manually-run comparison —
it is not wired into `bench_release.py`, CI, or any regenerated-on-every-release table.
`benchmarks/compare_pufferlib.py` is a standalone, run-on-demand script.

**Method used to produce the numbers below**: PufferLib's pip package does not always ship
a prebuilt CUDA extension, and was not installed in the environment this study ran in. For
this one-time run only, `puff_advantage_row_cuda` was obtained by JIT-compiling a
sha256-pinned, verbatim copy of PufferLib 3.0.0's own `pufferlib.cpp`/`pufferlib.cu` source
(vendored separately at `tests/pufferlib_ext/`, predating this study, used there by the
separate, non-pytest-collected `tests/benchmark_gae_vs_pufferlib.py`) — the exact code
PufferLib ships, just compiled locally instead of via pip. **This is not how
`compare_pufferlib.py` behaves going forward**: the script now imports the real
`pufferlib` pip package only and has no vendored/JIT-build fallback, so it will skip
cleanly rather than reproduce this path unless a future run has `pufferlib` actually
pip-installed with its prebuilt extension available. PufferLib's source (used here only
to compile its own published kernel, unmodified) is available at its own repository under
its own license terms: https://github.com/PufferAI/PufferLib.

Capability + design comparison, not a scoreboard. PufferLib's `puff_advantage_row_cuda` is a real hand-written CUDA kernel: one thread per environment row, sequential O(T) scan within the thread. rl-triton's `compute_gae` is a Triton kernel: one program per environment row, O(log T) in-SRAM tree reduction via `tl.associative_scan`. Different mechanisms, different tradeoffs — PufferLib's flat per-thread cost tends to win at very short horizons; rl-triton's parallel scan tends to win as horizon grows.

**Capability gap (not benchmarked, because there is nothing on the other side to benchmark against):** PufferLib takes a single `dones` flag per step, with no distinction between true episode termination and a time-limit truncation, and no `bootstrap_values` mechanism. rl-triton's interior-truncation path (`terminateds`/`truncateds`/`bootstrap_values`) has no PufferLib equivalent at all.

Both full-call wall time (headline — includes launch/wrapper overhead, what a caller pays every invocation) and device-only kernel time (diagnostic) are reported, since per-call overhead is central to the short-horizon comparison.

#### Production regime (seq_len [80,128] x num_envs [4096..38400])

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 80 | 0.0495 | 0.0039 | 0.0716 | 0.0545 | 1.45x | 13.80x |
| 4096 | 128 | 0.0574 | 0.0040 | 0.1471 | 0.1236 | 2.56x | 31.13x |
| 8192 | 80 | 0.0571 | 0.0064 | 0.0773 | 0.0540 | 1.35x | 8.43x |
| 8192 | 128 | 0.0590 | 0.0064 | 0.1429 | 0.1194 | 2.42x | 18.52x |
| 16384 | 80 | 0.0578 | 0.0114 | 0.0716 | 0.0560 | 1.24x | 4.94x |
| 16384 | 128 | 0.0396 | 0.0114 | 0.1345 | 0.1204 | 3.40x | 10.58x |
| 32768 | 80 | 0.0492 | 0.0215 | 0.0821 | 0.0695 | 1.67x | 3.23x |
| 32768 | 128 | 0.0496 | 0.0230 | 0.1407 | 0.1280 | 2.84x | 5.57x |
| 38400 | 80 | 0.0521 | 0.0249 | 0.1575 | 0.1452 | 3.02x | 5.84x |
| 38400 | 128 | 0.0541 | 0.0266 | 0.2810 | 0.2694 | 5.19x | 10.12x |

#### Boundary marker (seq_len [8,16,32]) — PufferLib's best-case regime

| num_envs | seq_len | triton full-call (ms) | triton device (ms) | puffer full-call (ms) | puffer device (ms) | speedup (full-call) | speedup (device) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4096 | 8 | 0.0400 | 0.0036 | 0.0237 | 0.0038 | 0.59x | 1.05x |
| 4096 | 16 | 0.0460 | 0.0037 | 0.0253 | 0.0093 | 0.55x | 2.54x |
| 4096 | 32 | 0.0492 | 0.0037 | 0.0530 | 0.0295 | 1.08x | 8.01x |
| 8192 | 8 | 0.0498 | 0.0061 | 0.0250 | 0.0039 | 0.50x | 0.65x |
| 8192 | 16 | 0.0356 | 0.0061 | 0.0235 | 0.0094 | 0.66x | 1.54x |
| 8192 | 32 | 0.0347 | 0.0061 | 0.0499 | 0.0351 | 1.44x | 5.73x |
| 16384 | 8 | 0.0383 | 0.0110 | 0.0231 | 0.0041 | 0.60x | 0.37x |
| 16384 | 16 | 0.0388 | 0.0110 | 0.0248 | 0.0096 | 0.64x | 0.87x |
| 16384 | 32 | 0.0395 | 0.0111 | 0.0452 | 0.0301 | 1.15x | 2.72x |
| 32768 | 8 | 0.0631 | 0.0209 | 0.0358 | 0.0050 | 0.57x | 0.24x |
| 32768 | 16 | 0.0647 | 0.0209 | 0.0361 | 0.0102 | 0.56x | 0.49x |
| 32768 | 32 | 0.0649 | 0.0209 | 0.0539 | 0.0320 | 0.83x | 1.53x |
| 38400 | 8 | 0.0674 | 0.0243 | 0.0357 | 0.0067 | 0.53x | 0.28x |
| 38400 | 16 | 0.0673 | 0.0243 | 0.0391 | 0.0170 | 0.58x | 0.70x |
| 38400 | 32 | 0.0672 | 0.0243 | 0.0896 | 0.0679 | 1.33x | 2.79x |

Reference: [PufferLib](https://github.com/PufferAI/PufferLib).
