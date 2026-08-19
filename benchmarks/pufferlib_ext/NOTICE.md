# Third-party code notice

`pufferlib.cpp` and `pufferlib.cu` in this directory are vendored **verbatim** from the
official `pufferlib==3.0.0` sdist on PyPI:

- Source project: [PufferLib](https://github.com/PufferAI/PufferLib)
- Version vendored: 3.0.0
- Files: `pufferlib/extensions/pufferlib.cpp`, `pufferlib/extensions/cuda/pufferlib.cu`
  (upstream paths)
- sha256(`pufferlib.cpp`) = `3efbfd62a5b2875451cc77bbb5cdb70a48001400e43cb3a4bbb724a926ee9a53`
- sha256(`pufferlib.cu`)  = `8154ba37ea6a722a13aa115734680d90c350699c7e6a7b7d8b811611f7f0fdc3`

This code is **not** authored by, or licensed to third parties by, this repository -- it is
PufferLib's own source, included here unmodified solely so that
`benchmarks/benchmark_gae_vs_pufferlib.py` can JIT-compile PufferLib's real advantage-kernel
CUDA implementation for a one-time, standalone hardware comparison, without requiring the
full `pip install pufferlib` dependency set (see `build.py`'s docstring for why). It is
governed by PufferLib's own license terms -- see [`LICENSE-PufferLib`](LICENSE-PufferLib)
in this directory for the verbatim text. This repository makes no license claim over these
two files beyond what is necessary to compile and benchmark them locally.

This directory and `benchmarks/benchmark_gae_vs_pufferlib.py` are standalone, run-on-demand
comparison scripts -- neither is part of the test suite, `bench_release.py`, or CI, and
neither is installed or built by `.runpod/start.sh`. (An earlier, superseded comparison
script, `compare_pufferlib.py`, is no longer tracked -- see `.gitignore` -- since it depends
on the real pip `pufferlib` package rather than this directory's vendored JIT build and its
output is not what the paper cites.)

## PufferLib 5.0 addendum

`pufferlib_5_0.cu` vendors PufferLib's advantage kernel (`puff_advantage`) as it
exists on the `5.0` branch of https://github.com/PufferAI/PufferLib -- **not**
a PyPI release. PufferLib does not publish versions past 3.0.0 to PyPI and
does not tag releases in git; each version instead lives as a same-named
branch (`refs/heads/4.0`, `refs/heads/5.0`, ...). `pip install
pufferlib==5.0.0` does not exist and cannot be installed.

- Source project: [PufferLib](https://github.com/PufferAI/PufferLib)
- Branch vendored: `5.0`
- Commit: `355e0be1fa7198b6ad7c82df90d0b9d487b4ac59` (2026-08-17, the branch tip
  as of this vendoring on 2026-08-19)
- File: `src/algo.cu` (upstream path), lines 1524-1602 (the `puff_advantage`
  kernel and its `adv_ld`/`adv_st` helpers) -- copied verbatim into
  `pufferlib_5_0.cu` and marked inline as VENDORED
- sha256(upstream `src/algo.cu`, full file) =
  `1f5bad1778a3efa01085c37de74a26eada19290689a84575138b09a17d7c2bc1`

Unlike the 3.0.0 files above, this is **not** a verbatim copy of an entire
upstream file: PufferLib 5.0 is "0 python" (no `TORCH_LIBRARY`/Python binding
exists upstream at all -- the kernel is called only from a C/CUDA training
loop in `src/pufferl.cu`), and the kernel itself is embedded in a ~1600-line
file (`algo.cu`) that also defines unrelated model/training machinery with
heavy dependencies (cuBLAS, NCCL, NVML, an env header selected via
`-DENV_HEADER`). `pufferlib_5_0.cu` extracts only the kernel + its two small
inline helpers (verbatim, marked) and adds a hand-written torch::Tensor
binding (not from PufferLib) modeled on this directory's existing 3.0.0
binding pattern, so it can be JIT-compiled standalone the same way. Float32
only (`-DPRECISION_FLOAT` in PufferLib's own build), matching the regime this
repo actually benchmarks -- the bf16 precision_t path is not vendored.

Same license terms and scope-of-inclusion apply as the 3.0.0 files above --
see [`LICENSE-PufferLib`](LICENSE-PufferLib). This repository makes no
license claim over the vendored kernel body beyond what is necessary to
compile and benchmark it locally.
