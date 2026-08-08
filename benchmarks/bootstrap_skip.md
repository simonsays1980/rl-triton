# HAS_BOOTSTRAP scalar-allocation skip: isolated measurement

*Measured on NVIDIA RTX 2000 Ada Generation · 2026-08-08 · isolates `gae_fused_kernel`'s `HAS_BOOTSTRAP` compile-time branch directly, bypassing `compute_gae`'s dispatch, so no other cost (truncation handling, wrapper overhead) is mixed in.*

Run via `python benchmarks/measure_bootstrap_skip.py`. Not part of the release sweep (`bench_release.py`) -- GPU/Triton/driver-version specific by design; re-run to check whether a cited percentage (e.g. the `gae_fused_kernel` docstring's "~33% at 128x1024") is still current rather than trusting it indefinitely.

Both configurations are asserted bit-for-bit identical in output before any timing is trusted (an all-zeros bootstrap buffer is mathematically a no-op, so `HAS_BOOTSTRAP=True` with zeros and `HAS_BOOTSTRAP=False` must agree exactly).

| num_envs | seq_len | HAS_BOOTSTRAP=True (ms) | HAS_BOOTSTRAP=False (ms) | saved (ms) | saved (% of True-path total) |
|:--------:|:-------:|:------------------------:|:--------------------------:|:----------:|:-----------------------------:|
|      128 |    1024 |                   0.0294 |                     0.0220 |     0.0074 |                          25.2 |
|       64 |     512 |                   0.0291 |                     0.0216 |     0.0076 |                          25.9 |
|      512 |     512 |                   0.0312 |                     0.0238 |     0.0075 |                          23.9 |
|     4096 |     128 |                   0.0386 |                     0.0310 |     0.0075 |                          19.5 |
