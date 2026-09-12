"""Loads the real PufferLib advantage-calculation CUDA kernel.

PufferLib's advantage kernel does NOT live at `pufferlib/torch_pufferl.py` /
`puff_advantage_row` (those names do not exist in the published package as of
PufferLib 3.0.0, verified against the PyPI sdist on 2026-07-23). The actual
entry points are:

  - Python wrapper: `compute_puff_advantage()` in `pufferlib/pufferl.py`,
    which dispatches to `torch.ops.pufferlib.compute_puff_advantage`.
  - CUDA kernel:     `puff_advantage_row_cuda()` in
    `pufferlib/extensions/cuda/pufferlib.cu` (one CUDA thread per
    environment row; sequential O(T) scan within each thread -- this is a
    genuine hand-written CUDA kernel, not a Python loop).
  - CPU fallback:    `puff_advantage_row()` in `pufferlib/extensions/pufferlib.cpp`.
  - Torch schema:     registered under the `pufferlib::compute_puff_advantage`
    namespace via `TORCH_LIBRARY(pufferlib, m)` in `pufferlib.cpp`.

`pufferlib.cpp` and `pufferlib.cu` in this directory are vendored **verbatim**
from the official `pufferlib==3.0.0` sdist on PyPI (sha256 of pufferlib.cpp:
3efbfd62a5b2875451cc77bbb5cdb70a48001400e43cb3a4bbb724a926ee9a53; pufferlib.cu:
8154ba37ea6a722a13aa115734680d90c350699c7e6a7b7d8b811611f7f0fdc3). We do not
`pip install pufferlib` in full: the package's `setup.py` also builds raylib,
Box2D, and dozens of C environment bindings unrelated to the advantage
kernel, and pulls in a large, version-pinned RL-environment dependency set
(gym==0.23, exact gymnasium/pettingzoo pins, wandb, neptune, ...) purely to
get one CUDA kernel. Importing the full `pufferlib.pufferl` module also has
process-wide side effects we do not want in a benchmark harness (it installs
a SIGINT handler that calls `os._exit(0)` and sets `warnings.filterwarnings
('error', category=RuntimeWarning)` globally). Compiling only the two
extension source files via `torch.utils.cpp_extension.load` gives the exact
same compiled kernel -- same source, same `TORCH_LIBRARY` registration -- none
of that.

If PufferLib happens to be fully installed already, we prefer its prebuilt
`_C` extension instead of rebuilding, since it is the same code either way.

`torch.utils.cpp_extension.load` caches its build under `TORCH_EXTENSIONS_DIR`
(or `~/.cache/torch_extensions/` if unset), keyed by a hash of the source
files and build flags -- so the JIT compile below only actually runs on a
cold cache (first run, a changed source/flag, or a cleared cache directory);
every subsequent call in the same environment returns the cached `.so`
almost immediately.

================================================================================
PufferLib 5.0 addendum (load_puff_advantage_5_0, below)
================================================================================

PufferLib does not publish version releases to PyPI beyond 3.0.0, and does
not tag versions in git -- each version lives as a same-named branch
(`refs/heads/4.0`, `refs/heads/5.0`, ...) on
https://github.com/PufferAI/PufferLib. `pip install pufferlib==5.0.0` does
not work; there is no such artifact. As of the `5.0` branch, PufferLib is
also "0 python" (its own author's description) -- the advantage kernel
(`puff_advantage` in `src/algo.cu`) is called only from a C/CUDA training
loop (`src/pufferl.cu`) with no `TORCH_LIBRARY`/Python binding at all.

`pufferlib_5_0.cu` therefore vendors just the kernel body (verbatim, marked
inline) plus a hand-written torch::Tensor binding modeled on this file's own
3.0.0 binding pattern -- see that file's module comment for the exact
provenance (commit, line range, sha256) and for what changed vs. 3.0.0
(vectorized 16B load/store, extra `returns` output, `ADV_THREADS=64` launch
config). Float32 only, matching PufferLib's `-DPRECISION_FLOAT` build option
and this repo's own benchmarking regime -- the bf16 path is not vendored.
"""
import os

import torch

_EXT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_puff_advantage():
    """Returns torch.ops.pufferlib.compute_puff_advantage, building it if needed.

    Returns:
        (op, source_desc): op is the callable
        `torch.ops.pufferlib.compute_puff_advantage(values, rewards, dones,
        importance, advantages, gamma, lambda, rho_clip, c_clip) -> None`
        (mutates `advantages` in place). source_desc is a short string noting
        whether the real installed package or the vendored JIT build was used.
    """
    try:
        from pufferlib import _C  # noqa: F401
        return torch.ops.pufferlib.compute_puff_advantage, "installed pufferlib._C"
    except ImportError:
        pass

    from torch.utils.cpp_extension import load

    print(
        "[pufferlib_ext] Building PufferLib's vendored CUDA extension "
        "(first run only -- cached under TORCH_EXTENSIONS_DIR after this)..."
    )
    load(
        name="_C",
        sources=[
            os.path.join(_EXT_DIR, "pufferlib.cpp"),
            os.path.join(_EXT_DIR, "pufferlib.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return (
        torch.ops.pufferlib.compute_puff_advantage,
        "vendored pufferlib==3.0.0 pufferlib.cpp/pufferlib.cu, JIT-compiled",
    )


def load_puff_advantage_5_0():
    """Returns torch.ops.pufferlib_5_0.compute_puff_advantage_5_0, building it if
    needed. PufferLib 5.0 has no Python bindings at all (it is "0 python" -- see
    pufferlib_5_0.cu's module docstring), so there is no installed-package
    fast path here (unlike load_puff_advantage() above): this always JIT-builds
    the vendored kernel plus hand-written torch binding glue.

    Returns:
        (op, source_desc): op is the callable
        `torch.ops.pufferlib_5_0.compute_puff_advantage_5_0(values, rewards,
        dones, importance, advantages, returns, gamma, lambda, rho_clip,
        c_clip) -> None` (mutates `advantages` and `returns` in place, float32
        only -- see pufferlib_5_0.cu). source_desc is a short provenance string.
    """
    from torch.utils.cpp_extension import load

    print(
        "[pufferlib_ext] Building PufferLib 5.0's vendored CUDA extension "
        "(first run only -- cached under TORCH_EXTENSIONS_DIR after this)..."
    )
    load(
        name="_C_5_0",
        sources=[
            os.path.join(_EXT_DIR, "pufferlib_5_0.cpp"),
            os.path.join(_EXT_DIR, "pufferlib_5_0.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return (
        torch.ops.pufferlib_5_0.compute_puff_advantage_5_0,
        "vendored pufferlib branch 5.0 @ 355e0be1 (2026-08-17) src/algo.cu "
        "puff_advantage, JIT-compiled, float32 only",
    )
