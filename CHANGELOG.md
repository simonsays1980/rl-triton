# Changelog

All notable changes to rl-triton are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

## [Unreleased]

### Changed
- Removed the `torch.zeros(num_envs)` bootstrap/seed-default allocation from
  the no-bootstrap/no-seed kernel path across all kernels (GAE, V-Trace,
  Lambda Returns, Discounted Returns, Eligibility Traces, Prefix Sum, and the
  shared scan fallback), via a `HAS_BOOTSTRAP`/`HAS_SEED` compile-time flag
  that substitutes a literal `0.0` instead. Eliminates an extra CUDA kernel
  launch that previously cost 28-40% of total op time at small sizes.
  Bit-identical output verified for every kernel. `bench_safeguard.py` floors
  recalibrated accordingly (e.g. GAE 1.4x → 1.9x, Prefix Sum flips from a
  0.75x non-regression guard to a genuine 1.1x speedup floor).

## [0.1.0] - 2026-06-08

### Added
- GAE kernel: fused backward associative scan, 1.6x over torch.compile at 128×1024
- V-Trace kernel: fused IS-weighted scan, 1.8x over torch.compile at 128×1024
- Retrace kernel: 2.2x over torch.compile at 128×1024
- Lambda Returns kernel: 1.6x over torch.compile at 128×1024
- Discounted Returns kernel: 1.3x over torch.compile at 128×1024
- Eligibility Traces kernel: 1.6x over torch.compile at 128×1024
- Episodic Prefix Sum kernel: cumulative sum with done-mask episode resets
- Safeguard benchmark suite enforcing minimum speedup thresholds
- PyTorch wrappers for all kernels with full docstrings
