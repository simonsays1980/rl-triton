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

rewards     = torch.randn(num_envs, seq_len, device=device)
values      = torch.randn(num_envs, seq_len, device=device)
terminateds = torch.zeros(num_envs, seq_len, device=device)
last_value  = torch.zeros(num_envs, device=device)  # V(s_T); 0 if the window ends at a true termination

advantages = compute_gae(
    rewards=rewards,
    values=values,
    terminateds=terminateds,
    gamma=0.99,
    lambda_=0.95,
    last_value=last_value,
)  # → (num_envs, seq_len)
```

`last_value` is the convenience form for the common case of no interior truncations -- see [Tensor Layout](#tensor-layout) below. For per-step truncations, pass a full `(num_envs, seq_len)` `bootstrap_values` tensor instead (mutually exclusive with `last_value`).

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
    last_value=last_value,
)  # → (num_envs, seq_len), (num_envs, seq_len)
```

### Discounted Returns

```python
from rl_triton import compute_discounted_returns

returns = compute_discounted_returns(
    rewards=rewards,
    terminateds=terminateds,
    gamma=0.99,
)  # → (num_envs, seq_len)
```

### TD(λ) Returns

```python
from rl_triton import compute_lambda_returns

next_values = values[:, 1:]  # V(s_{t+1}) for each step

returns = compute_lambda_returns(
    rewards=rewards,
    next_values=next_values,
    terminateds=terminateds,
    gamma=0.99,
    lambda_=0.95,
)  # → (num_envs, seq_len)
```

### Eligibility Traces

```python
from rl_triton import compute_eligibility_traces

gradients = torch.randn(num_envs, seq_len, device=device)  # ∇_w V̂(s_t)
dones     = torch.zeros(num_envs, seq_len, device=device)  # terminated | truncated

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
dones  = torch.zeros(num_envs, seq_len, device=device)  # terminated | truncated

prefix_sums = compute_episodic_prefix_sum(
    inputs=inputs,
    dones=dones,
)  # → (num_envs, seq_len)
```

The accumulation resets to zero whenever `dones[t] == 1`, so each episode's cumulative sum is independent of the previous one. An optional `seed_values` tensor of shape `(num_envs,)` sets the initial carry $C[-1]$ per environment (defaults to zero). Limited to `seq_len ≤ 131072`.

## Tensor Layout

All kernels expect `float32` tensors with shape `(num_envs, seq_len)`.

**`terminateds`** (used by `compute_gae`, `compute_vtrace`, `compute_retrace`, `compute_lambda_returns`, `compute_discounted_returns`): pass only true episode terminations (`1.0`). Truncated episodes - where the rollout window ended but the environment continues - must be `0.0` here. All five of these functions additionally accept an optional `truncateds` tensor (`1.0` at time-limit boundaries) to distinguish the two cases for correct bootstrap gating; `compute_retrace` requires it, the other four treat it as optional (defaulting every boundary to a termination if omitted).

**`dones`** (used by `compute_eligibility_traces` and `compute_episodic_prefix_sum`): pass `terminated | truncated`. Neither of these forward-scan kernels distinguishes termination from truncation - both simply reset the trace/sum at the boundary - so there is no separate `truncateds` parameter for either.

**`bootstrap_values`** (used by `compute_gae`, `compute_vtrace`, `compute_lambda_returns`, `compute_discounted_returns`) is a `(num_envs, seq_len)` tensor: the true continuation value `V(s_{t+1})` at truncated steps and at the final column (`t = T-1`) if the window ends mid-episode, zero everywhere else. `compute_gae` and `compute_vtrace` additionally accept **`last_value`** - a `(num_envs,)` convenience tensor for the common case of no interior truncations, mutually exclusive with `bootstrap_values` - which populates only the final column automatically. `compute_lambda_returns` and `compute_discounted_returns` have no `last_value` shortcut; pass a full `bootstrap_values` tensor even for a window-only boundary. `compute_retrace` has neither parameter: its continuation value is embedded directly in `next_q_values_all`, which the caller must supply for every step including the final column.

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
`truncateds=1` but `dones=0` - which is always a caller error (a step cannot
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
python tests/bench_release.py --parent-sweep --gpu "RTX 2000 Ada"
```

`--gpu` sets the label recorded in the output; omit it to auto-detect from `torch.cuda.get_device_name(0)`. This stages the results to `docs/benchmark-history/unreleased.md` for review - it never writes `benchmarks.md` directly. Add `--no-update` to print results to the console without staging anything. See [Contributing](contributing.md#adding-a-new-kernel) and the release workflow in `.github/workflows/gpu-tests.yml` for the full staging → promotion process.
