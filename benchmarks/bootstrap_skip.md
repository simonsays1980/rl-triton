# HAS_BOOTSTRAP scalar-allocation skip: isolated measurement

Isolates `gae_fused_kernel`'s `HAS_BOOTSTRAP` compile-time branch directly, bypassing `compute_gae`'s dispatch, so no other cost (truncation handling, wrapper overhead) is mixed in.

Run via `python benchmarks/measure_bootstrap_skip.py [--gpu LABEL]`. Not part of the release sweep (`bench_release.py`) -- GPU/Triton/driver-version specific by design; re-run to check whether a cited percentage (e.g. the `gae_fused_kernel` docstring's "~33% at 128x1024") is still current rather than trusting it indefinitely.

Both configurations are asserted bit-for-bit identical in output before any timing is trusted (an all-zeros bootstrap buffer is mathematically a no-op, so `HAS_BOOTSTRAP=True` with zeros and `HAS_BOOTSTRAP=False` must agree exactly).

This file is an UPSERT KEYED BY GPU, like `docs/benchmark-history/unreleased.md` -- re-running on the same GPU replaces that GPU's own section in place; running on a different GPU appends a new section, leaving others untouched. Never a wholesale overwrite.

## NVIDIA H100 80GB HBM3 · 2026-08-08

| num_envs | seq_len | HAS_BOOTSTRAP=True (ms) | HAS_BOOTSTRAP=False (ms) | saved (ms) | saved (% of True-path total) |
|:--------:|:-------:|:------------------------:|:--------------------------:|:----------:|:-----------------------------:|
|      128 |    1024 |                   0.0385 |                     0.0295 |     0.0091 |                          23.5 |
|       64 |     512 |                   0.0382 |                     0.0289 |     0.0093 |                          24.4 |
|      512 |     512 |                   0.0387 |                     0.0289 |     0.0098 |                          25.4 |
|     4096 |     128 |                   0.0382 |                     0.0301 |     0.0081 |                          21.2 |

## NVIDIA RTX 2000 Ada Generation · 2026-08-08

| num_envs | seq_len | HAS_BOOTSTRAP=True (ms) | HAS_BOOTSTRAP=False (ms) | saved (ms) | saved (% of True-path total) |
|:--------:|:-------:|:------------------------:|:--------------------------:|:----------:|:-----------------------------:|
|      128 |    1024 |                   0.0307 |                     0.0223 |     0.0084 |                          27.3 |
|       64 |     512 |                   0.0308 |                     0.0224 |     0.0084 |                          27.4 |
|      512 |     512 |                   0.0316 |                     0.0239 |     0.0077 |                          24.5 |
|     4096 |     128 |                   0.0392 |                     0.0312 |     0.0080 |                          20.5 |
