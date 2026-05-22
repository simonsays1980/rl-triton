# rl-triton

High-performance [Triton](https://github.com/openai/triton) GPU kernels for common reinforcement learning computations.

## Kernels

| Kernel | Op | Description |
|---|---|---|
| GAE | `compute_gae_triton` | Generalized Advantage Estimation — backward scan over `δ + γλ · A` |

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
import torch
from rl_triton import compute_gae_triton

deltas = torch.randn(64, 512, device="cuda")
decays = torch.rand(64, 512, device="cuda") * 0.99
advantages = compute_gae_triton(deltas, decays)
```

## Testing

```bash
pytest tests/ -v
```
