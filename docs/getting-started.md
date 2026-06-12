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
values           = torch.randn(num_envs, seq_len, device=device)
terminateds      = torch.zeros(num_envs, seq_len, device=device)
bootstrap_values = torch.zeros(num_envs, device=device)  # V(s_T) for truncated episodes

advantages = compute_gae(
    rewards=rewards,
    values=values,
    terminateds=terminateds,
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
    terminateds=terminateds,
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

All kernels expect `float32` tensors with shape `(num_envs, seq_len)`.

**`terminateds`** (used by `compute_gae` and `compute_vtrace`): pass only true episode terminations (`1.0`). Truncated episodes — where the rollout window ended but the environment continues — must be `0.0` here; supply `V(s_T)` via `bootstrap_values` instead.

**`dones`** (used by `compute_retrace`, `compute_lambda_returns`, `compute_discounted_returns`, `compute_eligibility_traces`, `compute_episodic_prefix_sum`): pass `terminated | truncated`. `compute_retrace` additionally accepts an optional `truncateds` tensor to distinguish the two cases for correct bootstrap gating.

`bootstrap_values` is a `(num_envs,)` tensor. Pass `V(s_T)` for truncated episodes and `0` for terminated ones. When omitted, it defaults to zero.

## Environment Variables

Both variables are read **once at import time**, so they must be set before
`import rl_triton` (or before the first kernel call in a fresh process).

### `RL_TRITON_PERF_WARNINGS`

```bash
RL_TRITON_PERF_WARNINGS=1 python train.py
```

Emits a `warnings.warn` whenever a non-contiguous input tensor triggers an
implicit `.contiguous()` copy inside the scan dispatcher. Off by default to
avoid noise in production training loops.

**When to enable:** profiling or debugging unexpectedly high memory allocation
rates. The warning points to the tensor (`u` or `v`) and recommends calling
`.contiguous()` once before the hot loop rather than paying the copy cost on
every step.

### `RL_TRITON_CORRECTNESS_WARNINGS`

```bash
RL_TRITON_CORRECTNESS_WARNINGS=1 python train.py
```

Emits a `warnings.warn` when `compute_retrace` detects a step where
`truncateds=1` but `dones=0` — which is always a caller error (a step cannot
be truncated without also being marked done). Off by default.

**When to enable:** integrating `compute_retrace` with a new environment or
rollout buffer, especially when adapting from a single-done-flag API.

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
