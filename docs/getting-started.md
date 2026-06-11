---
title: Getting Started
---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.2
- Triton ≥ 2.3
- A CUDA-capable GPU

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/simonsays1980/rl-triton.git
cd rl-triton
pip install -e ".[dev]"
```

The `[dev]` extra pulls in `pytest`, `pytest-benchmark`, and `numpy` for running tests and benchmarks.

## Quick Start

All kernels follow the same convention: inputs are `(num_envs, seq_len)` tensors on CUDA, and outputs are tensors of the same shape.

### Generalized Advantage Estimation (GAE)

```python
import torch
from rl_triton import compute_gae

num_envs, seq_len = 64, 512
device = "cuda"

rewards          = torch.randn(num_envs, seq_len, device=device)
values           = torch.randn(num_envs, seq_len + 1, device=device)
dones            = torch.zeros(num_envs, seq_len, device=device)
bootstrap_values = values[:, -1]  # V(s_T) for truncated episodes

advantages = compute_gae(
    rewards=rewards,
    values=values,
    dones=dones,
    gamma=0.99,
    lambda_=0.95,
    bootstrap_values=bootstrap_values,
)  # → (num_envs, seq_len)
```

### V-Trace (off-policy)

```python
from rl_triton import compute_vtrace

log_pi_target   = torch.randn(num_envs, seq_len, device=device)
log_pi_behavior = torch.randn(num_envs, seq_len, device=device)

vs, advantages = compute_vtrace(
    log_pi_target=log_pi_target,
    log_pi_behavior=log_pi_behavior,
    values=values,
    rewards=rewards,
    dones=dones,
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
    bootstrap_values=bootstrap_values,
)  # → (num_envs, seq_len), (num_envs, seq_len)
```

### Discounted Returns

```python
from rl_triton import compute_discounted_returns

returns = compute_discounted_returns(
    rewards=rewards,
    dones=dones,
    gamma=0.99,
    bootstrap_values=bootstrap_values,
)  # → (num_envs, seq_len)
```

### TD(λ) Returns

```python
from rl_triton import compute_lambda_returns

next_values = values[:, 1:]  # V(s_{t+1}) for each step

returns = compute_lambda_returns(
    rewards=rewards,
    next_values=next_values,
    dones=dones,
    gamma=0.99,
    lambda_=0.95,
    bootstrap_values=bootstrap_values,
)  # → (num_envs, seq_len)
```

### Eligibility Traces

```python
from rl_triton import compute_eligibility_traces

gradients = torch.randn(num_envs, seq_len, device=device)  # ∇_w V̂(s_t)

traces = compute_eligibility_traces(
    gradients=gradients,
    dones=dones,
    gamma=0.99,
    lambda_=0.95,
)  # → (num_envs, seq_len)
```

### Episodic Prefix Sum

```python
from rl_triton import compute_episodic_prefix_sum

inputs = torch.randn(num_envs, seq_len, device=device)

prefix_sums = compute_episodic_prefix_sum(
    inputs=inputs,
    dones=dones,
)  # → (num_envs, seq_len)
```

The accumulation resets to zero whenever `dones[t] == 1`, so each episode's cumulative sum is independent of the previous one. An optional `seed_values` tensor of shape `(num_envs,)` sets the initial carry $C[-1]$ per environment (defaults to zero). Limited to `seq_len ≤ 131072`.

## Tensor Layout

All kernels expect `float32` tensors with shape `(num_envs, seq_len)`. Episode boundaries are encoded in `dones`: a value of `1.0` at timestep $t$ means the episode ended at $t$, zeroing out the carry into the next step.

`bootstrap_values` is a `(num_envs,)` tensor. Pass `V(s_T)` for truncated episodes and `0` for terminated ones. When omitted, it defaults to zero.

## Correctness Tests

```bash
pytest tests/ -v
```

## Performance Tests

A fast safeguard suite runs one configuration per algorithm and asserts that the Triton kernel meets its performance target versus `torch.compile`:

```bash
pytest -m perf -v   # requires CUDA
```

The full sweep across all `(num_envs, seq_len)` configurations:

```bash
pytest -m slow -v   # requires CUDA, takes several minutes
```

## Benchmarking

To reproduce the release benchmark numbers on your own GPU:

```bash
python tests/bench_release.py --gpu "RTX 4090"
```

Omit `--gpu` to skip the header label. Add `--no-update` to print results without writing them back to the README.
