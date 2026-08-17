# Contributing to rl-triton

The full contributing guide -- repository layout, the five steps for adding a
kernel, test requirements, and the PR checklist -- lives in the documentation:

**[simonsays1980.github.io/rl-triton/contributing](https://simonsays1980.github.io/rl-triton/contributing/)**

(Source: [`docs/contributing.md`](docs/contributing.md).)

## Quick start

```bash
git clone https://github.com/simonsays1980/rl-triton
cd rl-triton
pip install -e ".[dev]"
```

Kernel development needs a Linux machine with a CUDA-capable GPU -- Triton
compiles and runs GPU kernels, and there is no CPU fallback.

```bash
pytest tests/ -v      # correctness tests
pytest -m perf -v     # PR performance gate (requires CUDA)
```

## Before opening a PR

- `kernels/` holds `@triton.jit` functions only; `ops/` holds Python
  orchestration only. The split is strict.
- Every new kernel needs a sequential reference implementation, a vectorized
  `torch.compile` baseline, correctness tests, and an entry in
  `tests/bench_safeguard.py`.
- All kernels are float32-only, and `done[t] = 1` means *start* of a new
  episode -- the opposite of the Gymnasium convention. See the full guide for
  both.

If a kernel you need does not exist yet, open an issue. Requests backed by a
concrete use case and a reference implementation get prioritised -- see
[ROADMAP.md](ROADMAP.md).
