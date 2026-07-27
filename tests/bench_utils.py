"""Shared timing helpers for all benchmarks."""
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile


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


def parallel_prefix_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Log-depth parallel PREFIX associative scan for FORWARD recurrences
    (e.g. eligibility traces): z[t] = a[t] + b[t]*z[t-1], z[-1] implicitly 0
    (inject a seed via a[:, 0] += b[:, 0] * seed before calling).

    Native forward (left-to-right) Hillis-Steele doubling scan — the direct
    mirror of parallel_suffix_scan's own doubling loop, written directly
    against the forward recurrence rather than via flip -> suffix -> flip.
    The prior flip-based construction was mathematically correct but its
    torch.flip/roll-heavy fused Inductor kernel triggered an autotuning
    illegal-memory-access crash; this formulation contains no torch.flip.

    At each doubling step, position t absorbs the aggregated segment ending
    at t-stride (its LEFT neighbor, vs. suffix scan's right neighbor):
      a[t] <- a[t] + b[t]*a[t-stride]
      b[t] <- b[t]*b[t-stride]
    for t >= stride; positions t < stride have no left neighbor within range
    (b[t-stride] treated as the identity's absorbing 0, same convention as
    parallel_suffix_scan's out-of-range zeroing) so they pass through with
    b[t] zeroed post-update, since nothing further can carry through them.

    Shape: a, b are [N, T].  Returns a_out [N, T].
    """
    N, T = a.shape
    # Pad T to next power of 2 for clean doubling. Padding at the tail is
    # inert for a forward/prefix recurrence: z[t] for t < T depends only on
    # positions <= t, never on the padded tail, so no special handling is
    # needed at the shape transition (unlike the suffix scan's tail padding,
    # which sits ahead of every real position instead of behind it).
    T_pad = 1
    while T_pad < T:
        T_pad *= 2
    if T_pad > T:
        pad = T_pad - T
        a = torch.cat([a, torch.zeros(N, pad, device=a.device, dtype=a.dtype)], dim=1)
        b = torch.cat([b, torch.zeros(N, pad, device=b.device, dtype=b.dtype)], dim=1)

    stride = 1
    while stride < T_pad:
        a_left = torch.roll(a, stride, dims=1)
        b_left = torch.roll(b, stride, dims=1)
        # Zero out positions that wrapped around (out of [0, T_pad) on the left).
        a_left[:, :stride] = 0.0
        b_left[:, :stride] = 0.0
        # z[t] = a[t] + b[t]*z[t-stride]
        a = a + b * a_left
        b = b * b_left
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


# ---------------------------------------------------------------------------
# Two timing granularities: full-call wall time (headline) vs device-only
# kernel time (diagnostic), plus the amortized (N-calls-per-region) variant.
# ---------------------------------------------------------------------------

def _bench_gpu_amortized(fn, *args, n_calls: int = 100, n_trials: int = 5, **kwargs) -> float:
    """N calls inside ONE timed region (one sync before, one after); ms/call.

    Separates harness per-call sync overhead from genuine per-call cost —
    the full-call single-shot number in _bench_gpu pays one cudaDeviceSynchronize
    per iteration, which at very short seq_len can be a large fraction of the
    measured time. This does not replace _bench_gpu as the speedup basis
    (single-call full-call wall remains that); it is reported alongside it,
    primarily for the short-seq_len production rows. Returns the min-of-medians
    across n_trials, same robustness rationale as _bench_gpu.
    """
    trial_vals = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_calls):
            fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        trial_vals.append(start.elapsed_time(end) / n_calls)
    trial_vals.sort()
    return trial_vals[len(trial_vals) // 2] if n_trials == 1 else min(trial_vals)


def _device_profile(fn, *args, n_iter: int = 20, n_warmup: int = 5, **kwargs) -> tuple[float, float]:
    """Device-only CUDA time (ms/call) and kernel-launch count (launches/call).

    Uses torch.profiler CUDA activity around steady-state calls (ncu/nsys are
    unusable in typical containerized GPU environments — RmProfilingAdminOnly
    restrictions). This captures ONLY time the GPU spends executing kernels,
    excluding Python dispatch / wrapper setup / launch overhead — the gap
    between this and _bench_gpu's full-call wall time IS that overhead, which
    callers still pay every invocation and which this function deliberately
    does not hide.
    """
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iter):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total_device_us = sum(e.self_device_time_total for e in events)
    launches = sum(e.count for e in events)
    return total_device_us / n_iter / 1000.0, launches / n_iter


# ---------------------------------------------------------------------------
# Correctness / determinism gates
# ---------------------------------------------------------------------------

def assert_correctness(triton_out, ref_out, label: str, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    """Tolerance-based correctness gate vs the sequential reference.

    NOT bit-identical: the associative-scan kernels reorder float ops
    depending on num_warps/block layout, so cross-config last-bit
    differences are legitimate and expected. This is the gate that must
    hold at every swept config, before any timing is trusted.
    """
    if isinstance(triton_out, tuple):
        for i, (t, r) in enumerate(zip(triton_out, ref_out)):
            torch.testing.assert_close(
                t, r, atol=atol, rtol=rtol,
                msg=lambda m, i=i: f"[{label}] output[{i}] mismatch vs reference: {m}",
            )
    else:
        torch.testing.assert_close(
            triton_out, ref_out, atol=atol, rtol=rtol,
            msg=lambda m: f"[{label}] mismatch vs reference: {m}",
        )


def assert_deterministic(fn, *args, label: str, n_repeats: int = 3, **kwargs) -> None:
    """Same-shape, same-num_warps repeat-run determinism check (bit-identical).

    Narrow scope on purpose: bit-identical is valid ONLY as a same-configuration
    check. It is NOT valid across configs (see assert_correctness docstring).
    """
    first = fn(*args, **kwargs)
    for i in range(1, n_repeats):
        again = fn(*args, **kwargs)
        if isinstance(first, tuple):
            for j, (a, b) in enumerate(zip(first, again)):
                if not torch.equal(a, b):
                    raise AssertionError(
                        f"[{label}] run {i} output[{j}] not bit-identical to run 0 "
                        f"at the same shape/config — kernel is non-deterministic."
                    )
        else:
            if not torch.equal(first, again):
                raise AssertionError(
                    f"[{label}] run {i} not bit-identical to run 0 at the same "
                    f"shape/config — kernel is non-deterministic."
                )


# ---------------------------------------------------------------------------
# Monotonicity gate (2% band): a larger problem must not measure faster than
# a smaller one along a single swept axis.
# ---------------------------------------------------------------------------

def _check_monotonic_series(labeled_ms: list[tuple], tol: float = 0.02) -> list[str]:
    """labeled_ms: [(size_label, ms), ...] already sorted ascending by size.

    Returns a list of human-readable violation strings (empty if none).
    """
    violations = []
    for (l1, v1), (l2, v2) in zip(labeled_ms, labeled_ms[1:]):
        if v2 < v1 * (1 - tol):
            violations.append(
                f"{l1} -> {l2}: {v1:.4f}ms -> {v2:.4f}ms "
                f"(decreased by {(1 - v2 / v1) * 100:.1f}%, beyond {tol * 100:.0f}% tolerance)"
            )
    return violations


def check_monotonic_grid(rows: list[dict], ms_key: str = "triton_ms", tol: float = 0.02) -> list[str]:
    """Monotonicity gate over a list of {'num_envs', 'seq_len', ms_key} rows.

    Checks two axes independently: for each fixed num_envs, ms should be
    non-decreasing (within tol) as seq_len grows; for each fixed seq_len, ms
    should be non-decreasing (within tol) as num_envs grows. Only uses pairs
    that actually exist in `rows` (grids need not be a full cross product).
    """
    violations = []
    by_num_envs: dict = {}
    by_seq_len: dict = {}
    for r in rows:
        by_num_envs.setdefault(r["num_envs"], []).append((r["seq_len"], r[ms_key]))
        by_seq_len.setdefault(r["seq_len"], []).append((r["num_envs"], r[ms_key]))

    for num_envs, series in by_num_envs.items():
        series = sorted(series)
        labeled = [(f"num_envs={num_envs},seq_len={sl}", ms) for sl, ms in series]
        violations += _check_monotonic_series(labeled, tol=tol)
    for seq_len, series in by_seq_len.items():
        series = sorted(series)
        labeled = [(f"seq_len={seq_len},num_envs={ne}", ms) for ne, ms in series]
        violations += _check_monotonic_series(labeled, tol=tol)
    return violations


# ---------------------------------------------------------------------------
# Startup environment print (GPU, torch/triton versions, cache warmth)
# ---------------------------------------------------------------------------

def print_environment_header(script_name: str) -> None:
    """Print GPU name, torch/triton versions, and TRITON_CACHE_DIR, flushed
    immediately, and warn if the cache dir looks cold/ephemeral — a cold
    cache silently spends minutes recompiling every kernel variant, which
    looks like a hang to anyone watching a long unattended sweep."""
    print(f"=== {script_name} ===", flush=True)
    if torch.cuda.is_available():
        print(f"GPU:              {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("GPU:              none (CUDA not available)", flush=True)
    print(f"torch:            {torch.__version__}", flush=True)
    try:
        import triton
        print(f"triton:           {triton.__version__}", flush=True)
    except ImportError:
        print("triton:           not installed", flush=True)

    cache_env = os.environ.get("TRITON_CACHE_DIR")
    cache_dir = cache_env or os.path.expanduser("~/.triton/cache")
    print(f"TRITON_CACHE_DIR: {cache_env or '(unset, default: ~/.triton/cache)'} -> {cache_dir}", flush=True)
    if os.path.isdir(cache_dir):
        n_entries = len(os.listdir(cache_dir))
        print(f"cache state:      exists, {n_entries} entries", flush=True)
        if n_entries == 0:
            print("WARNING: cache dir exists but is EMPTY — first kernel launches "
                  "will trigger LLVM compilation (can take minutes, looks like a hang).",
                  flush=True)
    else:
        print(f"WARNING: cache dir does not exist yet ({cache_dir}) — cold cache, "
              f"first run will compile every kernel variant from scratch.", flush=True)
    print(flush=True)
