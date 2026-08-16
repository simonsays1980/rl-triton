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
