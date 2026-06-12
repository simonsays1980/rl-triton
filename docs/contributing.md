---
title: Contributing
---

Thank you for your interest in contributing to **rl-triton**. This guide covers
everything you need to add a new kernel, write tests, and get a PR merged.

---

## Setup

```bash
git clone https://github.com/simonsays1980/rl-triton.git
cd rl-triton
pip install -e ".[dev]"
```

You need a CUDA-capable GPU for kernel development. Correctness tests can be
written on CPU first (mocking the Triton path), but all timing and integration
tests require CUDA.

---

## Repository Layout

```
src/rl_triton/
├── kernels/        # @triton.jit kernel definitions — no Python logic
│   ├── gae.py
│   ├── vtrace_fused.py
│   └── ...
└── ops/            # PyTorch wrappers — shape checks, dispatch, grid launch
    ├── gae.py
    ├── vtrace.py
    └── _scan.py    # shared chunked scan infrastructure

tests/
├── test_<name>.py          # correctness + vectorized baseline per kernel
├── bench_safeguard.py      # PR perf gate — one config per kernel, ≥1.5× vs torch.compile
├── bench_release.py        # full sweep across all (num_envs, seq_len) configs
└── bench_utils.py          # shared timing helpers
```

The split between `kernels/` and `ops/` is strict: `kernels/` files contain
only `@triton.jit` functions, `ops/` files contain only Python orchestration.
Never put Python control flow in a kernel file, and never put `@triton.jit`
code in an ops file.

---

## Adding a New Kernel

Follow these five steps in order.

### 1. Kernel file — `src/rl_triton/kernels/<name>_fused.py`

One `@triton.jit` function per file. Use `tl.constexpr` for `BLOCK_SIZE` and
any other compile-time constants. Mask out-of-bounds accesses with range
comparisons — never assume `seq_len` is a power of two at the call site.

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    input_ptr, dones_ptr, out_ptr,
    seq_len: int,
    row_stride: int,
    gamma: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row   = tl.program_id(0)
    offs  = tl.arange(0, BLOCK_SIZE)
    mask  = offs < seq_len
    base  = row * row_stride + offs
    x     = tl.load(input_ptr + base, mask=mask, other=0.0)
    # ... kernel body ...
    tl.store(out_ptr + base, result, mask=mask)
```

Add a short docstring explaining the algorithm, scan direction, and any
non-obvious indexing decisions.

### 2. Ops wrapper — `src/rl_triton/ops/<name>.py`

One public function per file. The wrapper is responsible for:

- Validating shapes, dtypes, and device placement
- Calling `.contiguous()` on inputs that the kernel requires to be contiguous
- Selecting `BLOCK_SIZE` via `triton.next_power_of_2(seq_len)`
- Dispatching to the fused kernel for `seq_len ≤ 131072`, falling back to the
  chunked scan path for longer sequences

```python
import torch
import triton
from rl_triton.kernels.my_kernel_fused import my_kernel_fused
from rl_triton.ops._scan import _run_scan, _FLAT_MAX_SEQ_LEN

def compute_my_kernel(
    inputs: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Docstring: what is computed, recurrence, shape contract, dtype, dispatch.

    Args:
        inputs: ..., [num_envs, seq_len], float32, CUDA.
        dones:  Episode termination flags (1.0=done), same shape, float32.
        gamma:  Discount factor.

    Returns:
        out: ..., shape [num_envs, seq_len], float32.
    """
    assert inputs.is_cuda and dones.is_cuda
    assert inputs.dtype == torch.float32
    assert inputs.shape == dones.shape

    num_envs, seq_len = inputs.shape
    inputs = inputs.contiguous()
    dones  = dones.contiguous()
    out    = torch.empty_like(inputs)

    if seq_len <= _FLAT_MAX_SEQ_LEN:
        BLOCK_SIZE = triton.next_power_of_2(seq_len)
        my_kernel_fused[(num_envs,)](
            inputs, dones, out,
            seq_len, inputs.stride(0),
            gamma=gamma,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out

    # Chunked fallback for seq_len > 131072
    u = inputs
    v = gamma * (1.0 - dones)
    return _run_scan(u, v)
```

### 3. Export — `src/rl_triton/__init__.py`

Add the new function to the imports and `__all__` list.

### 4. Tests — `tests/test_<name>.py`

Every test file must contain two things:

**A reference implementation** — a plain PyTorch sequential loop with no
tricks, used as the correctness ground truth:

```python
def reference_my_kernel(inputs, dones, gamma):
    """Sequential ground truth — never optimise this."""
    out   = torch.zeros_like(inputs)
    carry = torch.zeros(inputs.shape[0], device=inputs.device)
    for t in range(inputs.shape[1]):
        carry     = inputs[:, t] + gamma * (1 - dones[:, t]) * carry
        out[:, t] = carry
    return out
```

**A vectorized baseline** used by the PR performance gate — this is the
strongest `torch.compile` implementation, not the sequential loop:

```python
@torch.compile
def vectorized_my_kernel(inputs, dones, gamma):
    # Vectorized PyTorch equivalent — the bar the Triton kernel must clear
    ...
```

**Correctness tests** with `torch.testing.assert_close`:

```python
@cuda_only
@pytest.mark.parametrize("num_envs,seq_len", [(1, 16), (64, 512), (4, 131072)])
def test_correctness(num_envs, seq_len):
    torch.manual_seed(0)
    inputs = torch.randn(num_envs, seq_len, device="cuda")
    dones  = (torch.rand(num_envs, seq_len, device="cuda") < 0.05).float()
    expected = reference_my_kernel(inputs, dones, gamma=0.99)
    actual   = compute_my_kernel(inputs, dones, gamma=0.99)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
```

Test at least: a single environment, a typical batch size, the full fused
range (`seq_len = 131072`), and episode boundaries (non-zero `dones`).

### 5. Performance gate — `tests/bench_safeguard.py`

Add one entry to `bench_safeguard.py` for the PR perf gate. It runs a single
representative config (`128 envs × 1024 steps`) and asserts the Triton kernel
is **≥ 1.5× faster** than `torch.compile` on the vectorized baseline:

```python
@cuda_only
@pytest.mark.perf
def test_my_kernel_performance():
    inputs = torch.randn(_NUM_ENVS, _SEQ_LEN, device="cuda")
    dones  = torch.zeros(_NUM_ENVS, _SEQ_LEN, device="cuda")

    triton_ms    = _bench_gpu(lambda: compute_my_kernel(inputs, dones, gamma=0.99))
    pt_compile_ms = _bench_gpu(lambda: vectorized_my_kernel(inputs, dones, gamma=0.99))

    speedup = pt_compile_ms / triton_ms
    assert speedup >= _SPEEDUP_FLOOR, (
        f"compute_my_kernel speedup {speedup:.2f}× < {_SPEEDUP_FLOOR}× floor"
    )
```

---

## Running Tests

```bash
# Correctness tests (all algorithms, no CUDA required for CPU-only paths)
pytest tests/ -v

# PR performance gate (one config per algorithm, requires CUDA)
pytest -m perf -v

# Full benchmark sweep (all configs, requires CUDA, takes several minutes)
pytest -m slow -v
```

---

## Numerical Precision

All kernels require `float32` and will raise on any other dtype. This is
intentional: the associative scan accumulates over thousands of timesteps and
`bfloat16`'s 7 mantissa bits cause measurable drift at those sequence lengths.
Unlike matrix multiplications — where bf16 errors average out across large
inner products — a sequential scan chains rounding errors multiplicatively,
producing up to 6× relative error on individual advantage estimates at
`T=1024, gamma=0.99`.

When using `torch.autocast`, cast inputs back to `float32` before calling any
kernel:

```python
with torch.autocast("cuda"):
    # ... policy forward pass in bf16 ...
    rewards = rewards.float()
    values  = values.float()
    dones   = dones.float()
    advantages = compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95)
```

---

## Done Flag Convention

All kernels use the **start-of-episode** convention: `done[t] = 1` means
timestep `t` is the **first step of a new episode** and the carry from the
previous step is zeroed at `t`.

Most gym-compatible environments (including Gymnasium) use the opposite
**end-of-episode** convention where `done[t] = 1` marks the last step of the
ending episode. Passing these flags directly is silently wrong — shift them
before use:

```python
# gym_dones: end-of-episode convention
kernel_dones = torch.roll(gym_dones, shifts=1, dims=1)
kernel_dones[:, 0] = 0
```

See [NOTES.md](https://github.com/simonsays1980/rl-triton/blob/main/NOTES.md)
for the full explanation.

---

## Common Pitfalls

**Non-power-of-two `seq_len`** — always use `triton.next_power_of_2(seq_len)`
for `BLOCK_SIZE` and guard out-of-bounds loads with a range mask inside the
kernel.

**Non-contiguous inputs** — Triton pointer arithmetic assumes row-major
contiguous layout. Call `.contiguous()` in the wrapper before launching the
kernel. Set `RL_TRITON_PERF_WARNINGS=1` at runtime to surface cases where the
`.contiguous()` call is copying data inside a hot loop. Both environment
variables are read at import time — see
[Environment Variables](getting-started.md#environment-variables) for full
details.

**Retrace and discrete actions** — `compute_retrace` requires the full
action-probability vector over all actions to compute $\mathbb{E}_\pi[Q]$. It
is not applicable to continuous action spaces; use `compute_vtrace` instead.

**Terminated vs truncated episodes** — all value-estimating kernels require
the caller to handle this distinction, but they do so through different
mechanisms:

*GAE, V-Trace, lambda returns, discounted returns* — these kernels have no
explicit `truncateds` parameter. The distinction is encoded entirely in
`bootstrap_values`: pass `V(s_T)` for a truncated episode (the episode
continues beyond the window, so the next state has real value) and `0` for a
terminated episode (the episode ended, no future value). A mixed batch — some
environments truncated, others terminated — is handled naturally by
constructing `bootstrap_values` per-environment:

```python
# bootstrap_values[i] = V(s_T) if truncated[i] else 0
bootstrap_values = torch.where(truncated, value_at_boundary, torch.zeros_like(value_at_boundary))
advantages = compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95,
                         bootstrap_values=bootstrap_values)
```

*Retrace* — because the one-step Q-bootstrap is folded into each TD error
$\delta[t]$ via `next_q_values_all`, the distinction must be resolved
*inside* the kernel rather than at the boundary. Pass `terminated` (true
episode ends only) as `dones` and the separate `truncated` flag as
`truncateds`. When `truncateds=None`, every boundary is treated as a
termination — correct for purely episodic data but suboptimal when truncation
is common.

*Eligibility traces, episodic prefix sum* — forward scans with no value
bootstrapping; the terminated/truncated distinction is irrelevant.

---

## Pull Request Checklist

- [ ] `src/rl_triton/kernels/<name>_fused.py` — kernel only, no Python logic
- [ ] `src/rl_triton/ops/<name>.py` — wrapper with shape checks, dispatch, docstring
- [ ] Exported from `src/rl_triton/__init__.py`
- [ ] `tests/test_<name>.py` — reference impl, vectorized baseline, correctness tests
- [ ] Entry added to `tests/bench_safeguard.py` — PR perf gate passes (`pytest -m perf`)
- [ ] `pytest tests/ -v` passes (all correctness tests green)
- [ ] Kernel documented in `docs/kernels/`
