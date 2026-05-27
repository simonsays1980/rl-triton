"""Shared timing helpers for all benchmarks."""
import time

import torch


def _bench_gpu(fn, *args, n_warmup: int = 25, n_iter: int = 100, **kwargs) -> float:
    """Time a GPU kernel with CUDA events. Returns milliseconds per call."""
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def _bench_cpu(fn, *args, n_warmup: int = 3, target_s: float = 0.5, **kwargs) -> float:
    """Time a CPU-dispatched function with wall-clock. Returns milliseconds per call."""
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    elapsed = 0.0
    n = 0
    t0 = time.perf_counter()
    while elapsed < target_s or n < 5:
        fn(*args, **kwargs)
        n += 1
        elapsed = time.perf_counter() - t0
    return elapsed / n * 1000.0


def _n_iter_gpu(seq_len: int, num_envs: int) -> tuple[int, int]:
    """Scale warmup/iter counts so we don't over-benchmark small configs."""
    elements = seq_len * num_envs
    n_warmup = max(5,  min(25,  5_000_000 // elements))
    n_iter   = max(20, min(200, 20_000_000 // elements))
    return n_warmup, n_iter
