"""Shared timing helpers for all benchmarks."""
import time

import torch


def parallel_suffix_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Log-depth parallel suffix associative scan for benchmarking truncation baselines.

    Combines pairs (a[t], b[t]) under the operator
      (a1, b1) ∘ (a2, b2) = (a1 + b1*a2, b1*b2)
    where the LEFT operand is the earlier timestep (smaller t).

    After log2(T) doubling passes, a[t] = suffix scan result G[t] for the
    recurrence G[t] = a[t] + b[t]*G[t+1].  b[t]=0 at boundary steps severs
    the carry naturally (b1=0 → a1+b1*a2 = a1) — no special handling needed.

    This is the same associative operator used by tl.associative_scan in the
    Triton kernels.  Fully vectorized (no Python loop), compiles cleanly with
    torch.compile.

    Shape: a, b are [N, T].  Returns a_out [N, T].
    """
    N, T = a.shape
    # Pad T to next power of 2 for clean doubling.
    T_pad = 1
    while T_pad < T:
        T_pad *= 2
    if T_pad > T:
        pad = T_pad - T
        a = torch.cat([a, torch.zeros(N, pad, device=a.device, dtype=a.dtype)], dim=1)
        b = torch.cat([b, torch.zeros(N, pad, device=b.device, dtype=b.dtype)], dim=1)

    # Suffix scan: at each step, position t absorbs the aggregated result of t+stride.
    stride = 1
    while stride < T_pad:
        a_right = torch.roll(a, -stride, dims=1)
        b_right = torch.roll(b, -stride, dims=1)
        # Zero out positions that wrapped around (out of [0, T_pad) on the right).
        a_right[:, T_pad - stride:] = 0.0
        b_right[:, T_pad - stride:] = 0.0
        # G[t] = a[t] + b[t]*G[t+stride]
        a = a + b * a_right
        b = b * b_right
        stride *= 2

    return a[:, :T]


def _warmup_gpu(fn, *args, n_warmup: int = 20, **kwargs) -> None:
    """Run n_warmup untimed iterations and synchronize.

    Must be called once per (fn, config) pair before _bench_gpu — absorbs
    Triton JIT compilation, autotuning, cuBLAS init, and first-touch
    allocation so none of these land in the timed region.
    """
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()


def _bench_gpu(fn, *args, n_iter: int = 50, **kwargs) -> float:
    """Time a GPU kernel with CUDA events. Returns MEDIAN milliseconds per call.

    Caller must call _warmup_gpu(fn, *args, **kwargs) before this so that
    Triton compilation and autotuning are already done.  Measures each
    iteration individually with a pair of CUDA events and returns the median
    — robust against occasional scheduler jitter.
    """
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _bench_gpu_spread(
    fn_a, fn_b, args_a, args_b, kwargs_a, kwargs_b,
    n_warmup: int = 20, n_iter: int = 50, n_trials: int = 5,
) -> tuple[list[float], list[float], list[float]]:
    """Run two GPU functions n_trials times each and return per-trial speedups.

    Returns (speedups, ms_a_list, ms_b_list) where speedup[i] = ms_b[i] / ms_a[i].
    Used to measure run-to-run variance before setting performance floors.
    """
    _warmup_gpu(fn_a, *args_a, n_warmup=n_warmup, **kwargs_a)
    _warmup_gpu(fn_b, *args_b, n_warmup=n_warmup, **kwargs_b)
    speedups, ms_a_list, ms_b_list = [], [], []
    for _ in range(n_trials):
        ms_a = _bench_gpu(fn_a, *args_a, n_iter=n_iter, **kwargs_a)
        ms_b = _bench_gpu(fn_b, *args_b, n_iter=n_iter, **kwargs_b)
        speedups.append(ms_b / ms_a)
        ms_a_list.append(ms_a)
        ms_b_list.append(ms_b)
    return speedups, ms_a_list, ms_b_list


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
    """Scale iter count so we don't over-benchmark small configs.

    Warmup is always handled by explicit _warmup_gpu calls — this only
    controls the number of timed iterations passed to _bench_gpu.
    """
    elements = seq_len * num_envs
    n_iter = max(20, min(200, 20_000_000 // elements))
    return n_iter
