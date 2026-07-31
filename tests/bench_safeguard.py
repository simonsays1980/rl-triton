"""
PR safeguard performance tests.

Each test covers one algorithm at a single representative config (128 envs x 1024
steps) and asserts the Triton kernel is >=1.5x faster than torch.compile on a
fully-vectorized PyTorch reference.  Both sides are timed with CUDA events so the
comparison is apples-to-apples (pure GPU time, no Python-loop overhead).

They are fast (~5-10s total on GPU) and are intended to run on every PR to catch
regressions before they ship.

Run with:
    pytest tests/bench_safeguard.py -v
or select in CI:
    pytest -m perf
"""
import os

# Silence torch.compile/dynamo symbolic-shapes warnings (e.g. "q1 is not in
# var_ranges, defaulting to unknown range") -- see tests/bench_release.py's
# same setdefault for the full rationale. Only effective if this module's
# `import torch` below is the first import to touch torch in the process
# (pytest collection order can beat us to it if another test module already
# imported torch first).
os.environ.setdefault("TORCH_LOGS", "-dynamic")

import pytest
import torch

triton = pytest.importorskip("triton")

from bench_utils import _bench_gpu, _bench_gpu_spread, _n_iter_gpu, _warmup_gpu, check_monotonic_grid
from rl_triton.ops.gae import compute_gae
from rl_triton.ops.prefix_sum import compute_episodic_prefix_sum
from rl_triton.ops.retrace import compute_retrace
from rl_triton.ops.returns import (
    compute_discounted_returns,
    compute_eligibility_traces,
    compute_lambda_returns,
)
from rl_triton.ops.vtrace_fused import compute_vtrace_fused

# Import vectorized baselines from each algorithm's test module.
# These are the strongest compiled PyTorch baselines (no Python loop per timestep).
#
# GAE and discounted-returns import their _with_truncations siblings instead
# of the plain wrapper: torch.compile(vectorized_gae) / torch.compile(
# vectorized_discounted_returns) was found to silently produce WRONG numeric
# output (not NaN -- real wrong numbers) even at a single fixed shape,
# because the wrapper allocates torch.zeros_like(...) INSIDE the traced
# region before handing it to parallel_suffix_scan (likely an Inductor
# buffer-reuse bug). Compiling the *_with_truncations function directly and
# building the zero truncateds/bootstrap tensors in eager code (below, in
# each test) avoids it -- see tests/bench_release.py's bench_gae() for the
# full bisection. vectorized_vtrace/vectorized_lambda_returns/
# vectorized_eligibility_traces/vectorized_retrace were checked against this
# same failure mode directly and are safe to compile as-is.
from test_gae import vectorized_gae_with_truncations
from test_prefix_sum import vectorized_episodic_prefix_sum
from test_retrace import vectorized_retrace
from test_returns import (
    vectorized_discounted_returns_with_truncations,
    vectorized_eligibility_traces,
    vectorized_lambda_returns,
)
from test_vtrace import vectorized_vtrace

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

_NUM_ENVS      = 128
_SEQ_LEN       = 1024
_SPEEDUP_FLOOR = 1.8


# ---------------------------------------------------------------------------
# Per-GPU performance floors.
#
# Achievable Triton-vs-torch.compile speedup is hardware-dependent, not a
# fixed property of the kernel: at this suite's small 128x1024 config, each
# side's *device* kernel time is only a few microseconds, dwarfed by ~25-30us
# of fixed CUDA dispatch/sync overhead that both sides pay identically
# (confirmed via bench_utils._device_profile: e.g. GAE's own device-only
# ratio is 2.14x -- above the RTX floor that was in effect at the time (1.9x,
# since retired and recalibrated -- see below) -- even though its full-call
# wall-clock ratio, what this suite actually gates on, is ~1.5x). A faster
# GPU shrinks the device time on both sides while the fixed overhead stays
# the same, compressing the wall-clock ratio toward 1x. A floor calibrated on
# one card is therefore not transferable to a faster one; keying by device
# name is a correctness requirement here, not an optimization.
#
# "rtx_2000_ada" entries below were RETIRED and are gone from this table.
# The original values (gae 1.9, vtrace 1.8, retrace 1.35, lambda_returns 1.6,
# discounted_returns 1.25, eligibility_traces 1.85, prefix_sum 1.1) were never
# a genuine first-hand calibration to begin with: investigated 2026-07-26
# (three test_perf_gae/lambda_returns/eligibility_traces failures on a CI run
# believed at the time to be H100, later found to actually be H200 -- see
# below), commits 72b7299/b236ce9 that set these floors are both dated
# 2026-06-23, a full month before this repo's first H100-branded commit
# (dc246a4, 2026-07-22), and b236ce9's own prefix_sum comment from that same
# day describes "this card idles at 210MHz and boosts to 3105MHz, 70W TDP" --
# an RTX-2000-Ada-class clock/power profile (the H200 card measured then,
# mislabelled H100 at the time: ~345MHz idle / 1980MHz boost / 700W TDP via
# nvidia-smi). No commit message from that date names the GPU explicitly, so
# "RTX 2000 Ada" was inference from converging evidence (matches
# docs/benchmark-history/v0.1.0.md's "NVIDIA RTX 2000 Ada Generation" release,
# dated in between), not a confirmed first-hand record -- hence "existing/
# legacy, not asserted" in this file's history. Also measured against the
# same broken log-space-underflow baseline described in (2) below, same as
# the original h100_sxm entries -- a second, independent reason those values
# never carried real evidential weight.
#
# "rtx_2000_ada" entries now in the table below are the replacement: a
# genuine first-hand calibration, measured 2026-07-30 on real RTX 2000 Ada
# Generation hardware -- confirmed via torch.cuda.get_device_name(0) returning
# "NVIDIA RTX 2000 Ada Generation". Same methodology and margin convention as
# h200_sxm/h100_sxm below (3 independent process runs, lowest-of-3
# min-of-5-trials x0.9, median-of-5-trials for prefix_sum).
#
# "h100_sxm" entries measured 2026-07-26 were RETIRED and are gone from this
# table, on two independent grounds, either alone sufficient:
#   (1) wrong GPU. torch.cuda.get_device_name(0) on the card those numbers
#       came from reads "NVIDIA H200", not H100 -- despite this repo's
#       branch/commit naming assuming H100, nobody had checked the device
#       name string against the actual hardware until it was needed for this
#       table.
#   (2) wrong baseline. That measurement was taken against
#       torch.compile(vectorized_gae) / torch.compile(vectorized_discounted_returns)
#       baselines later found to silently produce wrong numeric output (an
#       Inductor buffer-reuse bug triggered by allocating torch.zeros_like(...)
#       inside the compiled region -- see the import comment above), plus a
#       separate Dynamo cross-shape-compilation state bug. Both were fixed
#       afterwards; any floor derived from the broken versions does not carry
#       over to the fixed ones.
#
# "h200_sxm" entries below are the replacement: remeasured against the fixed
# baselines, on the same physical H200 card, 3 independent full
# pytest-process runs (5 trials each) via a standalone script mirroring this
# file's exact methodology (single 128x1024 shape, torch.compile'd vec
# baseline, _bench_gpu_spread 5-trial spread). Floor = lowest of the 3 runs'
# min-of-5-trials, x0.9 (~10% margin), matching this file's existing
# floor-setting convention -- except prefix_sum, whose test gates on the
# *median* of 5 trials (see test_perf_prefix_sum): each of its 3 stored run
# values below is itself a median-of-5-trials, not a min, so taking
# min-of-those-three-medians x0.9 applies the same "lowest of 3 runs"
# convention at the granularity the test actually gates on, rather than
# silently reusing the min-of-5-trials formula the other six algorithms use.
# "h100_sxm" entries below are a genuine H100 calibration, measured
# 2026-07-28 on real H100 hardware -- confirmed via torch.cuda.get_device_name(0)
# returning "NVIDIA H100 80GB HBM3", not inferred or assumed this time. Same
# methodology and margin convention as h200_sxm above (3 independent runs,
# lowest-of-3 min-of-5-trials x0.9, median-of-5-trials for prefix_sum). This
# is additive, not a replacement: both h100_sxm and h200_sxm are now real,
# independently-measured entries for their respective cards.
_RTX_2000_ADA_MEASURED_2026_07_30 = {
    # algo: (run1, run2, run3) wall-clock speedup, 128x1024 -- min-of-5-trials
    # per run, except prefix_sum which is median-of-5-trials per run (see above)
    "gae":                 (3.110, 3.100, 3.052),
    "vtrace":              (2.758, 2.995, 3.005),
    "retrace":             (1.829, 1.814, 1.794),
    "lambda_returns":      (2.685, 2.671, 2.733),
    "discounted_returns":  (2.723, 2.805, 2.858),
    "eligibility_traces":  (2.489, 2.371, 2.515),
    "prefix_sum":          (2.530, 2.533, 2.584),  # median-gated, see test_perf_prefix_sum
}

_H200_MEASURED_2026_07_27 = {
    # algo: (run1, run2, run3) wall-clock speedup, 128x1024 -- min-of-5-trials
    # per run, except prefix_sum which is median-of-5-trials per run (see above)
    "gae":                 (3.474, 3.396, 3.389),
    "vtrace":              (3.237, 3.159, 3.165),
    "retrace":             (2.127, 2.145, 2.129),
    "lambda_returns":      (2.960, 3.021, 2.892),
    "discounted_returns":  (3.131, 3.146, 3.071),
    "eligibility_traces":  (2.877, 2.936, 2.836),
    "prefix_sum":          (3.061, 3.126, 2.924),  # median-gated, see test_perf_prefix_sum
}

_H100_MEASURED_2026_07_28 = {
    # algo: (run1, run2, run3) wall-clock speedup, 128x1024 -- min-of-5-trials
    # per run, except prefix_sum which is median-of-5-trials per run (see above)
    "gae":                 (3.555, 3.511, 3.500),
    "vtrace":              (3.223, 3.208, 3.167),
    "retrace":             (2.167, 2.164, 2.170),
    "lambda_returns":      (2.938, 3.072, 2.962),
    "discounted_returns":  (3.139, 3.194, 3.071),
    "eligibility_traces":  (2.911, 3.119, 2.923),
    "prefix_sum":          (3.043, 3.131, 3.047),  # median-gated, see test_perf_prefix_sum
}

_FLOOR_TABLE = {
    "rtx_2000_ada": {
        algo: round(min(runs) * 0.9, 2)
        for algo, runs in _RTX_2000_ADA_MEASURED_2026_07_30.items()
    },
    "h200_sxm": {
        algo: round(min(runs) * 0.9, 2)
        for algo, runs in _H200_MEASURED_2026_07_27.items()
    },
    "h100_sxm": {
        algo: round(min(runs) * 0.9, 2)
        for algo, runs in _H100_MEASURED_2026_07_28.items()
    },
}

_DEVICE_SUBSTRING_MAP = [
    ("RTX 2000 Ada", "rtx_2000_ada"),
    ("H200", "h200_sxm"),
    ("H100", "h100_sxm"),
]


def _floor_for(algo: str):
    """Return this algorithm's calibrated floor for the running GPU, or None
    if no entry matches -- callers must pytest.skip() on None rather than
    fall back to another GPU's floor (see module-level comment above)."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for substr, key in _DEVICE_SUBSTRING_MAP:
        if substr in name:
            return _FLOOR_TABLE[key][algo]
    return None


def _skip_if_uncalibrated(floor):
    if floor is None:
        pytest.skip(
            f"no calibrated perf floor for device "
            f"{torch.cuda.get_device_name(0)!r} -- add an entry to "
            f"_FLOOR_TABLE before trusting this gate on this hardware"
        )


# ---------------------------------------------------------------------------
# Input factories
# ---------------------------------------------------------------------------

def _gae_inputs():
    torch.manual_seed(0)
    d = "cuda"
    rewards = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    values  = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    dones   = (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float()
    return rewards, values, dones


def _vtrace_inputs():
    torch.manual_seed(0)
    d = "cuda"
    return (
        -torch.rand(_NUM_ENVS, _SEQ_LEN, device=d),   # log_pi_target
        -torch.rand(_NUM_ENVS, _SEQ_LEN, device=d),   # log_pi_behavior
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),   # values
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),   # rewards
        (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float(),  # dones
    )


def _retrace_inputs(num_actions=4):
    torch.manual_seed(0)
    d = "cuda"
    terminateds = (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float()
    # truncateds are mutually exclusive with terminateds
    truncateds  = ((torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05) & ~terminateds.bool()).float()
    return (
        torch.softmax(torch.randn(_NUM_ENVS, _SEQ_LEN, num_actions, device=d), dim=-1),
        torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) * 0.8 + 0.1,
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),
        torch.randn(_NUM_ENVS, _SEQ_LEN, num_actions, device=d),
        torch.randint(0, num_actions, (_NUM_ENVS, _SEQ_LEN), device=d),
        torch.randn(_NUM_ENVS, _SEQ_LEN, device=d),
        terminateds,
        truncateds,
    )


def _returns_inputs():
    torch.manual_seed(0)
    d = "cuda"
    rewards     = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    next_values = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    dones       = (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float()
    return rewards, next_values, dones


# ---------------------------------------------------------------------------
# Safeguard tests -- one per algorithm
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.perf
def test_perf_gae():
    # GAE and V-Trace shared one floor (_SPEEDUP_FLOOR=1.5) pre-harness-fix.
    # Re-calibrated after removing the torch.zeros(num_envs) bootstrap-default
    # allocation from the no-bootstrap kernel path (HAS_BOOTSTRAP constexpr).
    # floor is now per-GPU -- see _FLOOR_TABLE above. RTX 2000 Ada: 2.75
    # (3-run min 3.052x, ~10% margin, measured 2026-07-30). H200 SXM: 3.05
    # (3-run min 3.389x, ~10% margin). H100 SXM: 3.15 (3-run min 3.500x,
    # ~10% margin).
    _GAE_FLOOR = _floor_for("gae")
    _skip_if_uncalibrated(_GAE_FLOOR)
    args = _gae_inputs()
    # Compile vectorized_gae_with_truncations directly, not the plain wrapper
    # -- see the import comment above for why. truncateds/bootstrap built
    # here in eager code, never inside the compiled region.
    compiled = torch.compile(vectorized_gae_with_truncations)
    trunc0 = torch.zeros(_NUM_ENVS, _SEQ_LEN, device="cuda")
    bsv0   = torch.zeros(_NUM_ENVS, _SEQ_LEN, device="cuda")
    compiled_args = (*args, trunc0, bsv0)

    ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    speedups, tri_ms_list, vec_ms_list = _bench_gpu_spread(
        compute_gae, compiled, args, compiled_args,
        {"gamma": 0.99, "lambda_": 0.95}, {"gamma": 0.99, "lambda_": 0.95},
        n_iter=ni, n_trials=5,
    )
    speedup = min(speedups)

    print(
        f"\nGAE  {_NUM_ENVS}x{_SEQ_LEN}  (5-trial spread):"
        f"\n  triton ms : {[f'{x:.3f}' for x in tri_ms_list]}"
        f"\n  vec ms    : {[f'{x:.3f}' for x in vec_ms_list]}"
        f"\n  speedups  : {[f'{x:.2f}x' for x in speedups]}"
        f"\n  min={min(speedups):.2f}x  median={sorted(speedups)[2]:.2f}x  max={max(speedups):.2f}x"
    )
    assert speedup >= _GAE_FLOOR, (
        f"GAE Triton {speedup:.2f}x vs torch.compile(vec) -- below {_GAE_FLOOR}x floor"
        f" (5-run spread: min={min(speedups):.2f}x max={max(speedups):.2f}x)"
    )


@cuda_only
@pytest.mark.perf
def test_perf_vtrace():
    # Re-calibrated after removing the torch.zeros(num_envs) bootstrap-default
    # allocation from the no-bootstrap kernel path (HAS_BOOTSTRAP constexpr).
    # floor is now per-GPU -- see _FLOOR_TABLE above. RTX 2000 Ada: 2.48
    # (3-run min 2.758x, ~10% margin, measured 2026-07-30). H200 SXM: 2.84
    # (3-run min 3.159x, ~10% margin). H100 SXM: 2.85 (3-run min 3.167x,
    # ~10% margin).
    _VTRACE_FLOOR = _floor_for("vtrace")
    _skip_if_uncalibrated(_VTRACE_FLOOR)
    args = _vtrace_inputs()
    compiled = torch.compile(vectorized_vtrace)

    ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    speedups, tri_ms_list, vec_ms_list = _bench_gpu_spread(
        compute_vtrace_fused, compiled, args, args,
        {"gamma": 0.99}, {"gamma": 0.99},
        n_iter=ni, n_trials=5,
    )
    speedup = min(speedups)

    print(
        f"\nV-Trace  {_NUM_ENVS}x{_SEQ_LEN}  (5-trial spread):"
        f"\n  triton ms : {[f'{x:.3f}' for x in tri_ms_list]}"
        f"\n  vec ms    : {[f'{x:.3f}' for x in vec_ms_list]}"
        f"\n  speedups  : {[f'{x:.2f}x' for x in speedups]}"
        f"\n  min={min(speedups):.2f}x  median={sorted(speedups)[2]:.2f}x  max={max(speedups):.2f}x"
    )
    assert speedup >= _VTRACE_FLOOR, (
        f"V-Trace Triton {speedup:.2f}x vs torch.compile(vec) -- below {_VTRACE_FLOOR}x floor"
        f" (5-run spread: min={min(speedups):.2f}x max={max(speedups):.2f}x)"
    )


@cuda_only
@pytest.mark.perf
def test_perf_retrace():
    # Retrace reads a 3D action-prob tensor per timestep and computes advantages
    # in a second pass (store→debug_barrier→reload targets).  vectorized_retrace
    # does the same two-output work.  At 128x1024 this dispatches to the fully
    # fused single-kernel path (compute_retrace_fused), which has no bootstrap-
    # default allocation to remove; the floor below is a fresh re-calibration.
    # floor is now per-GPU -- see _FLOOR_TABLE above. RTX 2000 Ada: 1.61
    # (3-run min 1.794x, ~10% margin, measured 2026-07-30). H200 SXM: 1.91
    # (3-run min 2.127x, ~10% margin). H100 SXM: 1.95 (3-run min 2.164x,
    # ~10% margin).
    _RETRACE_FLOOR = _floor_for("retrace")
    _skip_if_uncalibrated(_RETRACE_FLOOR)
    args = _retrace_inputs()
    compiled = torch.compile(vectorized_retrace)

    ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    speedups, tri_ms_list, vec_ms_list = _bench_gpu_spread(
        compute_retrace, compiled,
        args, args,
        {"gamma": 0.99}, {"gamma": 0.99},
        n_iter=ni, n_trials=5,
    )
    speedup = min(speedups)  # min across trials is the trustworthy, interference-filtered value

    print(
        f"\nRetrace  {_NUM_ENVS}x{_SEQ_LEN}  (5-trial spread):"
        f"\n  triton ms : {[f'{x:.3f}' for x in tri_ms_list]}"
        f"\n  vec ms    : {[f'{x:.3f}' for x in vec_ms_list]}"
        f"\n  speedups  : {[f'{x:.2f}x' for x in speedups]}"
        f"\n  min={min(speedups):.2f}x  median={sorted(speedups)[2]:.2f}x  max={max(speedups):.2f}x"
    )
    assert speedup >= _RETRACE_FLOOR, (
        f"Retrace Triton {speedup:.2f}x vs torch.compile(vec) -- below {_RETRACE_FLOOR}x floor"
        f" (5-run spread: min={min(speedups):.2f}x max={max(speedups):.2f}x)"
    )


@cuda_only
@pytest.mark.perf
def test_perf_lambda_returns():
    # λ-returns is a near-pure scan: 3 inputs (rewards, next_values, terminateds),
    # trivial per-step arithmetic.  Both kernel and vectorized_lambda_returns are
    # fully memory-bound at 128x1024.  Re-calibrated after removing the
    # torch.zeros(num_envs) bootstrap-default allocation from the no-bootstrap
    # kernel path (HAS_BOOTSTRAP constexpr).
    # floor is now per-GPU -- see _FLOOR_TABLE above. RTX 2000 Ada: 2.4
    # (3-run min 2.671x, ~10% margin, measured 2026-07-30). H200 SXM: 2.60
    # (3-run min 2.892x, ~10% margin). H100 SXM: 2.64 (3-run min 2.938x,
    # ~10% margin).
    _LAMBDA_FLOOR = _floor_for("lambda_returns")
    _skip_if_uncalibrated(_LAMBDA_FLOOR)
    rewards, next_values, dones = _returns_inputs()
    compiled = torch.compile(vectorized_lambda_returns)

    ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    kw = {"gamma": 0.99, "lambda_": 0.95}
    speedups, tri_ms_list, vec_ms_list = _bench_gpu_spread(
        compute_lambda_returns, compiled,
        (rewards, next_values, dones), (rewards, next_values, dones),
        kw, kw,
        n_iter=ni, n_trials=5,
    )
    speedup = min(speedups)  # min across trials is the trustworthy, interference-filtered value

    print(
        f"\nλ-returns  {_NUM_ENVS}x{_SEQ_LEN}  (5-trial spread):"
        f"\n  triton ms : {[f'{x:.3f}' for x in tri_ms_list]}"
        f"\n  vec ms    : {[f'{x:.3f}' for x in vec_ms_list]}"
        f"\n  speedups  : {[f'{x:.2f}x' for x in speedups]}"
        f"\n  min={min(speedups):.2f}x  median={sorted(speedups)[2]:.2f}x  max={max(speedups):.2f}x"
    )
    assert speedup >= _LAMBDA_FLOOR, (
        f"λ-returns Triton {speedup:.2f}x vs torch.compile(vec) -- below {_LAMBDA_FLOOR}x floor"
        f" (5-run spread: min={min(speedups):.2f}x max={max(speedups):.2f}x)"
    )


@cuda_only
@pytest.mark.perf
def test_perf_discounted_returns():
    # Discounted returns is the lightest kernel (2 inputs, u[t]=r[t]) so the
    # associative-scan tree overhead is a larger fraction of total runtime than
    # for heavier kernels.  torch.compile(vectorized) avoids the scan entirely
    # via a log-space cumsum that is fully parallel across envs, leaving less
    # room for the Triton kernel to win on bandwidth.  Re-calibrated after
    # removing the torch.zeros(num_envs) bootstrap-default allocation from the
    # no-bootstrap kernel path (HAS_BOOTSTRAP constexpr); this is still the
    # noisiest, lightest kernel in the library (pre-fix runs dipped as low as
    # 1.046x), so its margin below the observed min is wider than the others.
    # floor is now per-GPU -- see _FLOOR_TABLE above. RTX 2000 Ada: 2.45
    # (3-run min 2.723x, ~10% margin, measured 2026-07-30).
    # H200 SXM: 2.76 (3-run min 3.071x, ~10% margin). H100 SXM: 2.76
    # (3-run min 3.071x, ~10% margin -- coincidentally identical to H200 here).
    _DISC_FLOOR = _floor_for("discounted_returns")
    _skip_if_uncalibrated(_DISC_FLOOR)
    rewards, _, dones = _returns_inputs()
    # Compile vectorized_discounted_returns_with_truncations directly, not
    # the plain wrapper -- see the import comment above for why.
    compiled = torch.compile(vectorized_discounted_returns_with_truncations)
    trunc0 = torch.zeros(_NUM_ENVS, _SEQ_LEN, device="cuda")
    bsv0   = torch.zeros(_NUM_ENVS, _SEQ_LEN, device="cuda")

    ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    speedups, tri_ms_list, vec_ms_list = _bench_gpu_spread(
        compute_discounted_returns, compiled,
        (rewards, dones), (rewards, dones, trunc0, bsv0),
        {"gamma": 0.99}, {"gamma": 0.99},
        n_iter=ni, n_trials=5,
    )
    speedup = min(speedups)  # min across trials is the trustworthy, interference-filtered value

    print(
        f"\nDisc-returns  {_NUM_ENVS}x{_SEQ_LEN}  (5-trial spread):"
        f"\n  triton ms : {[f'{x:.3f}' for x in tri_ms_list]}"
        f"\n  vec ms    : {[f'{x:.3f}' for x in vec_ms_list]}"
        f"\n  speedups  : {[f'{x:.2f}x' for x in speedups]}"
        f"\n  min={min(speedups):.2f}x  median={sorted(speedups)[2]:.2f}x  max={max(speedups):.2f}x"
    )
    assert speedup >= _DISC_FLOOR, (
        f"Discounted-returns Triton {speedup:.2f}x vs torch.compile(vec) -- below {_DISC_FLOOR}x floor"
        f" (5-run spread: min={min(speedups):.2f}x max={max(speedups):.2f}x)"
    )



@cuda_only
@pytest.mark.perf
def test_perf_eligibility_traces():
    # Eligibility traces is a forward scan: 2 inputs (gradients, dones), trivial
    # per-step arithmetic.  Both sides are memory-bound at 128x1024.
    # Re-calibrated after removing the torch.zeros(num_envs) seed-default
    # allocation from the no-seed kernel path (HAS_SEED constexpr).
    # floor is now per-GPU -- see _FLOOR_TABLE above. RTX 2000 Ada: 2.13
    # (3-run min 2.371x, ~10% margin, measured 2026-07-30). H200 SXM: 2.55
    # (3-run min 2.836x, ~10% margin). H100 SXM: 2.62 (3-run min 2.911x,
    # ~10% margin).
    _ELIG_FLOOR = _floor_for("eligibility_traces")
    _skip_if_uncalibrated(_ELIG_FLOOR)
    rewards, _, dones = _returns_inputs()
    compiled = torch.compile(vectorized_eligibility_traces)

    ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    kw = {"gamma": 0.99, "lambda_": 0.9}
    speedups, tri_ms_list, vec_ms_list = _bench_gpu_spread(
        compute_eligibility_traces, compiled,
        (rewards, dones), (rewards, dones),
        kw, kw,
        n_iter=ni, n_trials=5,
    )
    speedup = min(speedups)  # min across trials is the trustworthy, interference-filtered value

    print(
        f"\nElig-traces  {_NUM_ENVS}x{_SEQ_LEN}  (5-trial spread):"
        f"\n  triton ms : {[f'{x:.3f}' for x in tri_ms_list]}"
        f"\n  vec ms    : {[f'{x:.3f}' for x in vec_ms_list]}"
        f"\n  speedups  : {[f'{x:.2f}x' for x in speedups]}"
        f"\n  min={min(speedups):.2f}x  median={sorted(speedups)[2]:.2f}x  max={max(speedups):.2f}x"
    )
    assert speedup >= _ELIG_FLOOR, (
        f"Eligibility-traces Triton {speedup:.2f}x vs torch.compile(vec) -- below {_ELIG_FLOOR}x floor"
        f" (5-run spread: min={min(speedups):.2f}x max={max(speedups):.2f}x)"
    )


def _prefix_sum_inputs():
    torch.manual_seed(0)
    d = "cuda"
    inputs = torch.randn(_NUM_ENVS, _SEQ_LEN, device=d)
    dones  = (torch.rand(_NUM_ENVS, _SEQ_LEN, device=d) < 0.05).float()
    return inputs, dones


@cuda_only
@pytest.mark.perf
def test_perf_prefix_sum():
    # Prefix sum is the lightest kernel in the library: u[t]=x[t], v[t]=1-done[t],
    # no derived quantities, no next-values.  The baseline is the fair segmented
    # PyTorch equivalent (cumsum + episodic reset), not bare cumsum.  Removing
    # the torch.zeros(num_envs) seed-default allocation from the no-seed kernel
    # path (HAS_SEED constexpr) flipped this from a launch-overhead-bound loss
    # to a genuine win -- median speedup is ~1.24x across 30+ independent runs.
    #
    # This is also the shortest-duration kernel in the library, so unlike every
    # other test here, it gates on the MEDIAN rather than the min. Its min is
    # exposed to single-trial GPU clock-ramp transients (this card idles at
    # 210MHz and boosts to 3105MHz, 70W TDP) that can drag one of five trials
    # down without reflecting the kernel's real performance; the median is
    # robust to that single bad trial. min/median/max are still printed below
    # so the spread stays visible.
    # floor is now per-GPU -- see _FLOOR_TABLE above. RTX 2000 Ada: 2.28
    # (3-run min-of-medians 2.530x, ~10% margin, measured 2026-07-30).
    # H200 SXM: 2.63 (3-run min-of-medians 2.924x, ~10% margin). H100 SXM: 2.74
    # (3-run min-of-medians 3.043x, ~10% margin).
    _PREFIX_FLOOR = _floor_for("prefix_sum")
    _skip_if_uncalibrated(_PREFIX_FLOOR)
    args = _prefix_sum_inputs()
    compiled = torch.compile(vectorized_episodic_prefix_sum)

    ni = _n_iter_gpu(_SEQ_LEN, _NUM_ENVS)
    speedups, tri_ms_list, vec_ms_list = _bench_gpu_spread(
        compute_episodic_prefix_sum, compiled,
        args, args,
        {}, {},
        n_iter=ni, n_trials=5,
    )
    speedup = sorted(speedups)[2]  # median -- see comment above on why this kernel alone gates on median

    print(
        f"\nPrefix-sum  {_NUM_ENVS}x{_SEQ_LEN}  (5-trial spread):"
        f"\n  triton ms : {[f'{x:.3f}' for x in tri_ms_list]}"
        f"\n  vec ms    : {[f'{x:.3f}' for x in vec_ms_list]}"
        f"\n  speedups  : {[f'{x:.2f}x' for x in speedups]}"
        f"\n  min={min(speedups):.2f}x  median={sorted(speedups)[2]:.2f}x  max={max(speedups):.2f}x"
    )
    assert speedup >= _PREFIX_FLOOR, (
        f"Prefix-sum Triton median {speedup:.2f}x vs torch.compile(episodic) -- below {_PREFIX_FLOOR}x floor"
        f" (5-run spread: min={min(speedups):.2f}x max={max(speedups):.2f}x)"
    )


# ---------------------------------------------------------------------------
# Monotonicity gate -- a larger problem must never measure faster than a
# smaller one (2% band). This was missing entirely from the previous gate
# set; added here as a new CI-failing check without touching any existing
# _FLOOR constant. One small and one large config per algorithm;
# check_monotonic_grid (the same helper bench_release.py's production sweep
# uses) does the actual 2%-band comparison.
# ---------------------------------------------------------------------------

# This pair is deliberately far apart -- 128x512=65,536 elements vs.
# 512x4096=2,097,152, a 32x jump -- so that both ends sit in the
# compute/bandwidth-bound regime, well clear of the small-shape corner where
# fixed CUDA dispatch/sync overhead (~25-30us, see the per-GPU floor comment
# above) dominates the measured time. A violation at this gap means the
# kernel's own wall-clock time got smaller as the problem got strictly
# bigger -- not explainable by launch-overhead noise at either end -- so it
# is a genuine regression signal and this test (test_monotonicity_gate,
# below) is correctly a blocking PR gate.
#
# This is NOT the same question bench_release.py's release-sweep
# monotonicity check asks with the same check_monotonic_grid helper. That
# check runs over the full CONFIGS grid, which necessarily includes
# adjacent, overhead-dominated pairs (e.g. seq_len 80->128 at a fixed
# num_envs, in the production-regime table) where a 3-8% negative slope is
# expected GPU clock-ramp/launch-overhead jitter, not a regression -- see the
# RTX 2000 Ada 2026-07-30 sweep in NOTES.md for a concrete instance of
# exactly this. That is why the release sweep's monotonicity check is
# advisory (staged regardless, non-blocking) while this one is blocking:
# same helper, deliberately different severity, because the shape pairs
# being compared are structurally different questions. See NOTES.md's "One
# helper, two monotonicity gates" section before changing either one.
#
# Do not narrow this gap. Moving _MONO_SMALL/_MONO_LARGE closer together
# pulls this pair back into the overhead-dominated corner and starts
# blocking PRs on ordinary jitter, exactly the failure mode the release
# sweep already tolerates by treating its own check as advisory.
_MONO_SMALL = (128, 512)   # (num_envs, seq_len)
_MONO_LARGE = (512, 4096)


def _sized_gae_inputs(num_envs, seq_len):
    torch.manual_seed(0)
    d = "cuda"
    rewards = torch.randn(num_envs, seq_len, device=d)
    values  = torch.randn(num_envs, seq_len, device=d)
    dones   = (torch.rand(num_envs, seq_len, device=d) < 0.05).float()
    return rewards, values, dones


def _sized_vtrace_inputs(num_envs, seq_len):
    torch.manual_seed(0)
    d = "cuda"
    return (
        -torch.rand(num_envs, seq_len, device=d),
        -torch.rand(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, device=d),
        (torch.rand(num_envs, seq_len, device=d) < 0.05).float(),
    )


def _sized_retrace_inputs(num_envs, seq_len, num_actions=4):
    torch.manual_seed(0)
    d = "cuda"
    terminateds = (torch.rand(num_envs, seq_len, device=d) < 0.05).float()
    truncateds  = ((torch.rand(num_envs, seq_len, device=d) < 0.05) & ~terminateds.bool()).float()
    return (
        torch.softmax(torch.randn(num_envs, seq_len, num_actions, device=d), dim=-1),
        torch.rand(num_envs, seq_len, device=d) * 0.8 + 0.1,
        torch.randn(num_envs, seq_len, device=d),
        torch.randn(num_envs, seq_len, num_actions, device=d),
        torch.randint(0, num_actions, (num_envs, seq_len), device=d),
        torch.randn(num_envs, seq_len, device=d),
        terminateds,
        truncateds,
    )


def _sized_returns_inputs(num_envs, seq_len):
    torch.manual_seed(0)
    d = "cuda"
    rewards     = torch.randn(num_envs, seq_len, device=d)
    next_values = torch.randn(num_envs, seq_len, device=d)
    dones       = (torch.rand(num_envs, seq_len, device=d) < 0.05).float()
    return rewards, next_values, dones


def _sized_prefix_sum_inputs(num_envs, seq_len):
    torch.manual_seed(0)
    d = "cuda"
    inputs = torch.randn(num_envs, seq_len, device=d)
    dones  = (torch.rand(num_envs, seq_len, device=d) < 0.05).float()
    return inputs, dones


def _mono_row(fn, args, kwargs, num_envs, seq_len):
    _warmup_gpu(fn, *args, **kwargs)
    ni = _n_iter_gpu(seq_len, num_envs)
    ms = _bench_gpu(fn, *args, n_iter=ni, **kwargs)
    return {"num_envs": num_envs, "seq_len": seq_len, "triton_ms": ms}


@cuda_only
@pytest.mark.perf
def test_monotonicity_gate():
    """A larger problem must never measure faster than a smaller one (2% band),
    checked directly against each Triton kernel (not a baseline comparison)."""
    small_ne, small_sl = _MONO_SMALL
    large_ne, large_sl = _MONO_LARGE

    rows = [_mono_row(compute_gae, _sized_gae_inputs(ne, sl),
                       {"gamma": 0.99, "lambda_": 0.95}, ne, sl)
            for ne, sl in (_MONO_SMALL, _MONO_LARGE)]
    gae_violations = check_monotonic_grid(rows, ms_key="triton_ms")

    rows = [_mono_row(compute_vtrace_fused, _sized_vtrace_inputs(ne, sl),
                       {"gamma": 0.99}, ne, sl)
            for ne, sl in (_MONO_SMALL, _MONO_LARGE)]
    vtrace_violations = check_monotonic_grid(rows, ms_key="triton_ms")

    rows = [_mono_row(compute_retrace, _sized_retrace_inputs(ne, sl),
                       {"gamma": 0.99}, ne, sl)
            for ne, sl in (_MONO_SMALL, _MONO_LARGE)]
    retrace_violations = check_monotonic_grid(rows, ms_key="triton_ms")

    rows = []
    for ne, sl in (_MONO_SMALL, _MONO_LARGE):
        rewards, next_values, dones = _sized_returns_inputs(ne, sl)
        rows.append(_mono_row(compute_lambda_returns, (rewards, next_values, dones),
                               {"gamma": 0.99, "lambda_": 0.95}, ne, sl))
    lambda_violations = check_monotonic_grid(rows, ms_key="triton_ms")

    rows = []
    for ne, sl in (_MONO_SMALL, _MONO_LARGE):
        rewards, _, dones = _sized_returns_inputs(ne, sl)
        rows.append(_mono_row(compute_discounted_returns, (rewards, dones),
                               {"gamma": 0.99}, ne, sl))
    disc_violations = check_monotonic_grid(rows, ms_key="triton_ms")

    rows = []
    for ne, sl in (_MONO_SMALL, _MONO_LARGE):
        rewards, _, dones = _sized_returns_inputs(ne, sl)
        rows.append(_mono_row(compute_eligibility_traces, (rewards, dones),
                               {"gamma": 0.99, "lambda_": 0.9}, ne, sl))
    elig_violations = check_monotonic_grid(rows, ms_key="triton_ms")

    rows = []
    for ne, sl in (_MONO_SMALL, _MONO_LARGE):
        inputs, dones = _sized_prefix_sum_inputs(ne, sl)
        rows.append(_mono_row(compute_episodic_prefix_sum, (inputs, dones), {}, ne, sl))
    prefix_violations = check_monotonic_grid(rows, ms_key="triton_ms")

    all_violations = {
        "gae": gae_violations, "vtrace": vtrace_violations, "retrace": retrace_violations,
        "lambda_returns": lambda_violations, "discounted_returns": disc_violations,
        "eligibility_traces": elig_violations, "prefix_sum": prefix_violations,
    }
    failed = {k: v for k, v in all_violations.items() if v}
    print(f"\nMonotonicity gate ({small_ne}x{small_sl} -> {large_ne}x{large_sl}, 2% band):")
    for algo, violations in all_violations.items():
        status = "FAILED" if violations else "PASSED"
        print(f"  {algo}: {status}" + (f" -- {violations}" if violations else ""))
    assert not failed, f"Monotonicity gate failed for: {failed}"
