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

High-performance [Triton](https://github.com/openai/triton) GPU kernels for reinforcement-learning credit assignment and return estimation.

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

Run the full release benchmark suite with:

```bash
python tests/bench_release.py --parent-sweep --gpu "RTX 2000 Ada"
```

See [Benchmarks](benchmarks.md) for methodology, full results, and benchmark details.

## Performance

Full sweep, methodology, and truncation-path results: [Benchmarks](benchmarks.md).

{%
  include-markdown "../README.md"
  start="num_envs=4096, seq_len=128"
  end="<!-- BENCH_END -->"
%}
