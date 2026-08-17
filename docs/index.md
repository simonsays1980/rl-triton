---
title: rl-triton
hide:
  - toc
---

<style>
.md-content__inner > h1 { display: none; }
</style>

<!-- docs/index.md -->
<p align="center">
  <img src="assets/banner_dark.svg" alt="rl-triton" width="100%"/>
</p>

High-performance [Triton](https://github.com/openai/triton) GPU kernels for common reinforcement learning computations.

## Kernels

{%
  include-markdown "../README.md"
  start="<!-- KERNELS_START -->"
  end="<!-- KERNELS_END -->"
%}

For the `seq_len > 131072` fallback behavior of each kernel, see
[GPU Concepts](kernels/gpu-concepts.md) and the per-kernel pages under
[Kernels](kernels/index.md).

## Installation

{%
  include-markdown "../README.md"
  start="<!-- INSTALL_START -->"
  end="<!-- INSTALL_END -->"
%}

## Usage

{%
  include-markdown "../README.md"
  start="<!-- USAGE_START -->"
  end="<!-- USAGE_END -->"
%}

## Testing

{%
  include-markdown "../README.md"
  start="<!-- TESTING_START -->"
  end="<!-- TESTING_END -->"
%}

## Benchmarking

The release benchmark runs all algorithms across multiple (num_envs, seq_len) configs
and stages the results to `docs/benchmark-history/unreleased.md` for review -- it never
writes `benchmarks.md` directly:

```bash
python tests/bench_release.py --parent-sweep --gpu "RTX 2000 Ada"
```

Use `--no-update` to print results without staging anything. A staged candidate becomes
the published `benchmarks.md` only when explicitly promoted with a version tag:

```bash
python tests/bench_release.py --promote --version <new-release-tag>   # e.g. v0.1.4
```

## Performance

Full sweep, methodology, and truncation-path results: [Benchmarks](benchmarks.md).

{%
  include-markdown "../README.md"
  start="benchmarks.md).*"
  end="<!-- BENCH_END -->"
%}
