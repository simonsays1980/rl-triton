"""Shared metadata capture and timing helpers for the unilab_rl V-trace study.

Run under the isolated venv created for this study (unilab_rl requires
torch>=2.7; the rest of this repo pins torch>=2.4.1, so this suite is kept
out of the system environment on purpose -- see the run notes in
unilab_vtrace_correctness-<date>.json's "setup" block for the exact venv
path and install commands).
"""
import datetime
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton


def _git_commit(repo_dir: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except Exception:
        return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def capture_metadata(unilab_repo_dir: str) -> dict:
    """One-time hardware/software metadata block, included in every output file."""
    assert torch.cuda.is_available(), "CUDA required for this study"
    props = torch.cuda.get_device_properties(0)
    try:
        unilab_version = importlib.metadata.version("unilab-rl")
    except importlib.metadata.PackageNotFoundError:
        unilab_version = None
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": int(props.total_memory),
        "gpu_memory_gb": round(props.total_memory / (1024**3), 1),
        "cuda_version": torch.version.cuda,
        "driver_version": _driver_version(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "python_version": sys.version.split()[0],
        "unilab_rl_version": unilab_version,
        "unilab_rl_git_commit": _git_commit(unilab_repo_dir),
        "cpu_model": _cpu_model(),
        "os": platform.platform(),
        "captured_at": datetime.datetime.now().isoformat(),
    }


def today() -> str:
    return datetime.date.today().isoformat()


def unique_path(base: Path) -> Path:
    """Append -run2, -run3, ... if base already exists today (don't silently overwrite)."""
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    n = 2
    while True:
        candidate = base.with_name(f"{stem}-run{n}{suffix}")
        if not candidate.exists():
            return candidate
        n += 1


# --- CUDA-event timing (matches tests/bench_utils.py's _bench_gpu / _warmup_gpu pattern) ---

def warmup_gpu(fn, *args, n_warmup: int = 20, **kwargs) -> None:
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()


def bench_gpu_cuda_events(fn, *args, n_iter: int = 100, n_trials: int = 5, **kwargs):
    """Min-of-trial-medians CUDA-event timing (ms). Returns (headline_ms, trial_medians_ms)."""
    trial_medians = []
    for _ in range(n_trials):
        times = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        trial_medians.append(times[len(times) // 2])
    return min(trial_medians), trial_medians


def bench_gpu_wall_clock(fn, *args, n_iter: int = 100, n_trials: int = 5, **kwargs):
    """Min-of-trial-medians synchronized wall-clock timing (ms) via perf_counter.

    Secondary check alongside the CUDA-event number -- reported separately,
    never blended into one speedup number, since a CPU round-trip arm may
    behave differently under host-clock vs. device-clock measurement.
    """
    trial_medians = []
    for _ in range(n_trials):
        times = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        times.sort()
        trial_medians.append(times[len(times) // 2])
    return min(trial_medians), trial_medians


def device_profile(fn, *args, n_iter: int = 20, n_warmup: int = 5, **kwargs):
    """Device-only CUDA time (ms/call), excluding Python dispatch/launch overhead.

    Mirrors tests/bench_utils.py's _device_profile: torch.profiler CUDA
    activity around steady-state calls (ncu/nsys are typically unusable in
    containerized GPU environments).
    """
    from torch.profiler import ProfilerActivity, profile
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iter):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total_device_us = sum(e.self_device_time_total for e in events)
    return total_device_us / n_iter / 1000.0  # ms/call


def assert_finite_and_close(t, r, label, atol=1e-4, rtol=1e-4):
    """Return (passed, max_abs_err, mean_abs_err, rmse) without raising."""
    finite_t = torch.isfinite(t)
    finite_r = torch.isfinite(r)
    if not (finite_t.all() and finite_r.all()):
        return False, float("inf"), float("inf"), float("inf")
    diff = (t - r).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rmse = float(torch.sqrt((diff**2).mean()))
    try:
        torch.testing.assert_close(t, r, atol=atol, rtol=rtol)
        passed = True
    except AssertionError:
        passed = False
    return passed, max_abs, mean_abs, rmse
