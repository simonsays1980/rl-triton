# Changelog

All notable changes to rl-triton are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

## [Unreleased]

## [0.1.0] - 2026-06-08

### Added
- GAE kernel: fused backward associative scan, 1.6x over torch.compile at 128×1024
- V-Trace kernel: fused IS-weighted scan, 1.8x over torch.compile at 128×1024
- Retrace kernel: 2.2x over torch.compile at 128×1024
- Lambda Returns kernel: 1.6x over torch.compile at 128×1024
- Discounted Returns kernel: 1.3x over torch.compile at 128×1024
- Eligibility Traces kernel: 1.6x over torch.compile at 128×1024
- Safeguard benchmark suite enforcing minimum speedup thresholds
- PyTorch wrappers for all kernels with full docstrings
- Example scripts for all kernels in examples/
