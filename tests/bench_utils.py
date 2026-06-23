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


def _bench_gpu(fn, *args, n_iter: int = 50, n_trials: int = 5, **kwargs) -> float:
    """Time a GPU kernel with CUDA events. Returns the MIN-of-medians across n_trials.

    Caller must call _warmup_gpu(fn, *args, **kwargs) before this so that
    Triton compilation and autotuning are already done.  Each trial measures
    n_iter iterations individually with a pair of CUDA events (explicit sync
    immediately before start AND stop) and takes that trial's MEDIAN — robust
    against occasional scheduler jitter within a trial.

    Repeating across n_trials and taking the MIN guards against a rarer but
    confirmed failure mode on this hardware: intermittent multi-millisecond
    interference episodes (GPU power-state/clock transitions, OS scheduling
    stalls) that last long enough to corrupt MORE than half of a single
    trial's iterations, inflating even that trial's median. Such an episode
    can only ever slow a trial down, never make it faster than true steady
    state, so the min across independently-warmed trials is a sound estimator
    of the kernel's real cost and is what eliminates the "smaller config
    measures slower than a larger one" artifacts this harness was built to
    catch — set n_trials=1 to fall back to a single plain median.
    """
    trial_medians = []
    for _ in range(n_trials):
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
        trial_medians.append(times[len(times) // 2])
    return min(trial_medians)


def _bench_gpu_spread(
    fn_a, fn_b, args_a, args_b, kwargs_a, kwargs_b,
    n_warmup: int = 20, n_iter: int = 50, n_trials: int = 5,
) -> tuple[list[float], list[float], list[float]]:
    """Run two GPU functions n_trials times each and return per-trial speedups.

    Returns (speedups, ms_a_list, ms_b_list) where speedup[i] = ms_b[i] / ms_a[i].
    Used to measure run-to-run variance before setting performance floors.

    Each of these n_trials calls into _bench_gpu uses _bench_gpu's own default
    (min-of-5-medians) rather than a single plain median. Confirmed empirically
    (repeated bench_safeguard.py runs) that without that inner robustness, a
    single outer trial can land entirely inside one of this GPU's intermittent
    multi-ms interference episodes and report a wildly low one-off speedup
    (observed down to ~0.6x on kernels that are reliably >1.5x otherwise) —
    exactly the artifact this harness exists to filter out. Costs 5x more
    measurement work per outer trial; worth it so every value in the returned
    list — including min(speedups) — is trustworthy enough to gate a floor on.
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


def _n_iter_gpu(seq_len: int, num_envs: int) -> int:
    """Scale iter count so we don't over-benchmark small configs.

    Warmup is always handled by explicit _warmup_gpu calls — this only
    controls the number of timed iterations passed to _bench_gpu.
    """
    elements = seq_len * num_envs
    n_iter = max(20, min(200, 20_000_000 // elements))
    return n_iter
