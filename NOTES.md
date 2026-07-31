# Implementation Notes

## Why there is no chunked fused V-Trace kernel

`compute_vtrace_fused` eliminates all intermediate tensor allocations and kernel
launches by doing everything -- IS ratio computation, backward scan, target
construction, and advantage computation -- inside a single Triton kernel per
environment row.  It is the fast path for `seq_len <= 131072`.

For longer sequences, `compute_vtrace` falls back to a two-stage path:
PyTorch elementwise ops to compute `deltas` and `decays`, followed by
`compute_vtrace_chunked` for the backward scan.  This path is not fused.

### Why a chunked fused kernel was not built

A chunked fused kernel would extend the single-kernel approach to arbitrary
sequence lengths by splitting the sequence into chunks and running a two-pass
scan (forward pass to collect per-chunk carry values, backward pass to
propagate and correct them).  This is technically feasible but was not
implemented for the following reasons:

1. **Sequence lengths > 131072 are uncommon in practice.**  Most on-policy RL
   training (PPO, IMPALA, V-trace) uses rollouts of 2k–8k steps.  Sequences
   long enough to hit the chunked path arise mainly in model-based RL or when
   packing many short episodes into a single trajectory.

2. **Diminishing returns at long sequences.**  At seq_len > 131072, throughput
   is dominated by global memory bandwidth, not kernel launch overhead.  The
   advantage of fusion -- eliminating intermediate allocations and launch
   latency -- is a much smaller fraction of total runtime than at shorter
   sequences.

3. **Substantial implementation complexity.**  The two-pass chunked scan
   requires coordinating carry propagation across chunk boundaries in a way
   that is tightly coupled to the advantage computation.  The additional
   correctness surface (boundary conditions, bootstrap propagation across
   chunks) is non-trivial and would require a dedicated test suite.

### When to build it

Implement a chunked fused kernel if:

- Users on long-sequence workloads (model-based RL, large context windows)
  report that the chunked path is a measurable bottleneck in profiling.
- A benchmark shows the unfused chunked path is meaningfully slower than what
  a fused version could achieve (i.e., intermediate allocation + launch
  overhead is still significant at the target sequence lengths).

The existing `chunked_gae_kernel` and `vtrace_fused_kernel` are good starting
points: the chunked two-pass structure is already correct and tested, and the
fused kernel's IS-ratio and advantage logic can be reused inside each chunk's
kernel.

## dtype support and autocast

All kernels require float32 inputs and will raise if passed anything else.
This is intentional: the backward scan accumulates additions over potentially
thousands of timesteps, and bf16's limited precision (7 mantissa bits vs 23
for float32) causes meaningful numerical drift at those sequence lengths.

### torch.autocast compatibility

`torch.autocast` silently casts tensors to bf16 inside its context, which will
hit the float32 assertion.  Cast inputs explicitly before entering the scan:

```python
with torch.autocast("cuda"):
    # ... policy forward pass in bf16 ...
    deltas = deltas.float()
    decays = decays.float()
    advantages = compute_gae(deltas, decays)
```

This matches standard mixed-precision RL practice: the policy network runs in
bf16 for speed, while value targets and advantage estimates are kept in float32
for numerical stability.

### bf16 support

bf16 kernels are not planned.  If a use case arises where float32 is a
bottleneck (e.g. very large batches on memory-bandwidth-limited hardware),
the right approach is to benchmark whether the precision loss is acceptable
for that specific workload rather than enabling it globally.

---

## Performance warnings

Set the environment variable `RL_TRITON_PERF_WARNINGS=1` to enable opt-in
warnings about performance bottlenecks detected at runtime.  The default is
silent to avoid noise in production training loops.

Currently warned conditions:

- **Non-contiguous input tensors** -- any `u` or `v` tensor passed to `_run_scan`
  or `_run_scan_forward` that is not contiguous in memory (e.g. produced by
  strided indexing like `tensor[..., 0]`).  The wrapper silently calls
  `.contiguous()`, which allocates a fresh copy.  If this happens inside a hot
  training loop the copy cost accumulates.  Fix: call `.contiguous()` once
  before the loop rather than on every step.

The warnings are emitted via Python's standard `warnings` module, so they can
be filtered or escalated with `warnings.filterwarnings` as usual.

To add a new performance warning, call `_perf_warn(msg)` from
`src/rl_triton/ops/_scan.py`.  The flag check is centralised there.

---

## Handling variable-length episodes in a batch

All kernels in this package operate on a rectangular `[num_envs, seq_len]`
tensor.  There is no per-row length argument.  When episodes in a batch differ
in length, use the **zero-decay convention**:

Set the decay factor to `0.0` at the terminal step of each episode.  Because
every kernel implements a recurrence of the form

```
A[t] = u[t] + decay[t] * A[t+1]
```

a zero decay zeroes out the carry from the right at that position, effectively
resetting the scan.  Timesteps beyond the terminal can be padded with any value
for `u`/`delta`; they will never propagate backwards through the zero.

This convention is already built into the per-step decay tensor that callers
construct.  For example, in GAE:

```python
decay[t] = gamma * lambda_ * (1 - done[t])
```

A `done=1` at the true terminal step produces `decay=0`, which resets the scan
at that boundary.  Pad the remaining columns with zeros for both `deltas` and
`decays`.

### Packing multiple episodes per row

The same convention allows multiple episodes to be packed into a single row.
Place `decay=0` at each episode boundary and the scan resets there, treating
the segment to the right as a separate episode.  This is what frameworks like
numpy do when concatenating episodes into fixed-length trajectory windows.

### Eligibility traces (forward scan)

`compute_eligibility_traces` runs a forward scan.  The same `done` flag zeroes
`v[t] = gamma * lambda * (1 - done[t])`, resetting the trace at episode
boundaries in exactly the same way.

---

## `done` flag convention: end-of-episode vs. start-of-episode

All kernels in this package use the **start-of-episode** convention for `done`
flags:

> `done[t] = 1` means step `t` is the **first step of a new episode**.  The
> carry from the previous step is zeroed *before* accumulating step `t`.

This is expressed directly in the recurrence as `v[t] = f(1 - done[t])`, so
`done[t] = 1` → `v[t] = 0` → the carry is dropped at `t` itself.

### Gymnasium uses the opposite convention

Gymnasium (and most gym-compatible environments) use the **end-of-episode**
convention:

> `done[t] = 1` means step `t` is the **last step of the current episode**.
> Step `t` still belongs to the ending episode; the new episode begins at
> `t+1`.

Passing Gymnasium `dones` directly to any kernel in this package is **silently
wrong**: the episode boundary will be applied one step too early.

### Adapting Gymnasium dones

Shift the `done` signal forward by one position before passing it to the
kernel:

```python
# gym_dones[t] = 1 means t is the last step of an episode (Gymnasium convention)
# kernel_dones[t] = 1 means t is the first step of a new episode (this package)
kernel_dones = torch.roll(gym_dones, shifts=1, dims=1)
kernel_dones[:, 0] = 0  # first step is never a boundary carry-reset
```

After this shift, `done` at the original terminal step `t` now sits at `t+1`,
correctly zeroing the carry at the first step of the next episode.

### Which convention to use

If you build `dones` yourself (e.g. from `truncated | terminated` in a rollout
buffer), use the start-of-episode convention directly -- place `done=1` at the
first timestep of each new episode -- and no shift is needed.

---

## Why there is no chunked forward scan kernel

`compute_eligibility_traces` uses a forward scan (`e[t] = u[t] + v[t] * e[t-1]`)
and is limited to `seq_len <= 131072` (the flat kernel limit).

A chunked forward scan is structurally symmetric to the chunked backward scan
(`chunked_gae_kernel`) and could be implemented by processing chunks
left-to-right and carrying `e[chunk_end]` forward across boundaries instead of
backward.  It was not built for the same reason as the chunked fused V-Trace
kernel: sequences longer than 131072 are uncommon in RL, and the implementation
complexity is not justified until users report it as a bottleneck.

---

## Double HBM read of the values tensor in gae_fused_kernel -- investigated, not a win

The GAE fused kernel (`gae_fused_kernel`) loads the values tensor twice from
HBM -- in *both* the `HAS_TRUNCATIONS=True` and `HAS_TRUNCATIONS=False` paths
(and in the older non-fused `gae_kernel`), not only the truncations path as
originally noted here:

1. `tl.load(values_ptr + base + rev, ...)` -- gives `v_t`
2. `tl.load(values_ptr + base + rev + 1, ...)` -- gives `v_{t+1}`

These are two separate `tl.load` calls at different pointer offsets; they are
**not** loaded once into SRAM and then indexed at two positions. SRAM in
Triton is used for inter-thread communication during the scan, not as a
random-access scratchpad.

**Measured (H200; same CUDA-events full-call + device-only methodology used
throughout this repo's benchmarks -- see `benchmarks.md`'s Methodology
paragraph): adopting the `lambda_returns_fused_kernel` pre-shift strategy
makes this slower, not faster.** Implemented as an A/B kernel
(`gae_fused_kernel_preshift`, taking a caller-supplied `next_values` tensor
instead of the second in-kernel load) and benchmarked against the shipped
kernel:

- **Correctness:** bit-identical output (max diff = 0.0) across 64×512,
  512×4096, 128×8192 -- confirms the two formulations compute the same math.
- **ptxas metadata @ 512×4096:** identical -- 84 regs/thread, 0 spills, in
  *both* variants. Unlike Retrace's redundant 3D reload (which fed a
  register-hungry `[BLOCK_SIZE, ACTION_BLOCK]` reduction), `v_next` here is a
  single scalar-per-lane float -- removing its load doesn't touch register
  pressure at all. GAE was never spilling in the first place.
- **Kernel-only device time** (pre-shift kernel vs shipped kernel, excluding
  the cost of building `next_values`): pre-shift is *slightly slower* at
  seq_len≥4096 (0.90–0.94x). The two loads in the current kernel are offset
  by exactly one element -- well within a single L1/L2 cache line -- so they
  already share most of their traffic; there was no real second HBM
  round-trip to eliminate. Reading from a separate `next_values` tensor
  instead loses that locality.
- **Full path including materialization** (`next_values[:, :-1] =
  values[:, 1:]`, one extra kernel launch + full read/write pass, mirroring
  what a caller or wrapper would have to do since `compute_gae`'s public API
  takes only `values`, not a pre-shifted `next_values` like
  `compute_lambda_returns` does): **0.52–0.73x -- 30–90% slower** across
  64×512 through 512×8192. The extra launch dominates any theoretical
  bandwidth saving, consistent with this project's repeated finding
  (short-horizon sweep, Retrace ceiling fallback) that added kernel launches
  are costly relative to a few extra bytes of already-cached HBM traffic.

**Verdict: not applied.** The premise (this is meaningfully "double HBM
traffic") doesn't hold up under measurement -- it's two cache-friendly loads
of adjacent addresses, not two independent full tensor reads -- and the
proposed fix is a net loss once the caller-side cost is counted honestly. No
further action recommended here; `lambda_returns_fused_kernel`'s pre-shift
design is *not* a copyable win for GAE, because `lambda_returns`'s design
gets its `next_values` "for free" from the caller's own rollout bookkeeping,
whereas GAE would have to synthesize it internally at real extra cost.

---

## Two `torch.compile` correctness bugs found in the benchmark PyTorch baselines (not this project's kernels)

While fixing the vectorized PyTorch baselines used as comparison points in
`benchmarks.md`/`bench_safeguard.py` (see the log-space-underflow note
below), two separate `torch.compile`/Inductor/Dynamo bugs surfaced. Both
produce **silently wrong numeric output -- not NaN/inf, not a crash** -- which
is why they went unnoticed: the baseline's own output was previously never
checked for correctness, only timed. Neither is a bug in this project's
Triton kernels or in `parallel_suffix_scan`/`parallel_prefix_scan`
themselves (both verified correct, eager and compiled, in isolation).

### Bug 1 -- allocating `torch.zeros_like(...)` inside a `torch.compile`'d function corrupts results

`torch.compile(vectorized_gae)` (a thin wrapper that did
`truncateds = torch.zeros_like(terminateds)` *inside* the function body
before calling `vectorized_gae_with_truncations`) gave up to 17% of output
elements wrong (max abs diff ~8, not a rounding-level discrepancy) at some
shapes -- even on a single, freshly-compiled shape with no prior compilation
history. Bisected via a 4-way split (zeros built inside vs. outside the
compiled region, with/without an unrelated `if x is not None` branch):
**only "zeros allocated inside the compiled region, then fed into
`parallel_suffix_scan`'s loop" reproduces it.** The same zeros built in eager
code and passed in as a plain tensor argument to the compiled function is
correct. Suspected cause: Inductor's memory planner treating the
always-zero tensor as a reusable/aliasable buffer candidate, corrupting the
scan's many loop-local temporaries. Affected `vectorized_gae` and
`vectorized_discounted_returns` (both allocate an unconditional all-zero
buffer fed wholesale into the scan); `vectorized_vtrace`,
`vectorized_lambda_returns`, and `vectorized_eligibility_traces` do the same
kind of allocation but were not observed to trigger it -- inconsistent
enough across functions that the fix was applied everywhere as a precaution,
not just where it was caught.

**Fix:** never `torch.compile` a wrapper that allocates tensors internally.
Compile `vectorized_X_with_truncations` directly; build `truncateds`/
`bootstrap_values` (or any other constant/zero tensor) in eager code at the
call site, always as arguments, never inside the traced function.

### Bug 2 -- reusing one `torch.compile`'d object across shapes with different padding behavior corrupts results

Separately: reusing a single `torch.compile(...)`-wrapped object across
*different* input shapes -- the normal pattern in a benchmark sweep that
iterates many `(num_envs, seq_len)` configs -- silently gives wrong output
for some shape transitions, even after fixing Bug 1. Specifically observed
between a shape that needs `parallel_suffix_scan`'s padding path
(`seq_len=80`, padded to `T_pad=128`) and one that doesn't (`seq_len=512`,
already a power of 2) when compiled by the *same* object in sequence. A
freshly-compiled object used for only one shape is always correct; the same
"fresh" object created *inside a loop that has already compiled other
shapes in the same process* can still be wrong -- so this isn't simply
"don't reuse objects," it's cross-compilation contamination in Dynamo's
process-global caching/guard state.

**Fix:** call `torch._dynamo.reset()` immediately before warming up a
compiled baseline at a new shape (reusing the same wrapped object
otherwise). Verified across 11+ shapes spanning both the headline `CONFIGS`
grid and the production-regime grid, including the specific failing pair,
with zero mismatches after the reset. Must run *before* warmup for that
shape, never between warmup and the timed calls (that would force a
mid-measurement recompile and corrupt the timing).

**Takeaway for future benchmark code in this repo:** any `torch.compile`
baseline that (a) wraps a function allocating its own tensors, or (b) gets
reused across more than one input shape in the same process, needs both of
these guards. Neither is specific to the scan algorithms here -- worth
checking for elsewhere if `torch.compile` baselines are added to future
benchmarks.

### Bug 2 addendum -- confirmed independent of `reset()` usage; not confined to padding transitions; mechanism still unproven

A later session re-examined Bug 2 because its original repro (above) still
used `torch._dynamo.reset()` calls elsewhere in the same process, leaving
open the possibility that what was observed was an artifact of resetting
itself (e.g. some reset-accumulation side effect) rather than a bug that
exists independent of resets. This was checked directly.

**What was actually observed (fact, reproduced this session):**
- Compiling two different functions (`vectorized_gae_with_truncations`,
  `vectorized_vtrace_with_truncations`) across the full non-padding `CONFIGS`
  grid, then reusing one of those compiled objects into `PRODUCTION_CONFIGS`
  (crossing into the `seq_len=80` padding shape) -- **with zero
  `torch._dynamo.reset()` calls anywhere in the process** -- reproduces the
  same wrong-output failure at the padding transition that Bug 2 originally
  reported. This rules out "artifact of reset() usage" as the explanation:
  the bug is present with no resets at all.
- A **second, distinct** instance was found in the same zero-reset condition:
  `vectorized_discounted_returns_with_truncations`, compiled by one object
  first at `(num_envs=64, seq_len=512)` then reused at `(128, 1024)` --
  **neither shape needs padding** (both already powers of 2) -- gives ~22.5%
  of elements wrong (not NaN/inf) at the second shape. This is NOT the
  padding-specific mechanism Bug 2 originally described; it is the same
  general *symptom* (cross-shape reuse of one compiled object → silently
  wrong output) triggered by a different, non-padding shape pair, specific
  to this one baseline function. Reproduces even in a subprocess that only
  ever compiles this single object (no other algorithm's compiled objects
  involved) -- i.e. it is not about *total* accumulated compiles in the
  process either; two compiles of one object is already enough. A brand-new
  object's first-ever compile at either shape alone, or the same object used
  at only one shape, is always correct.

**What is interpretation, not proof:** calling this "process-state
corruption in Dynamo/Inductor's caching or guard mechanism" is the working
description, matching the observed shape (reuse-across-shapes triggers it;
a single fresh compile never does). Neither this nor the original Bug 2 was
root-caused to a specific Inductor buffer-reuse or guard bug via
IR/generated-kernel inspection -- that would need reading the Inductor-
generated Triton source per shape and diffing the buffer/guard plan, which
was not done. Treat "why" as probable, not proven, if this is ever reported
upstream to PyTorch; only the "reuse of one compiled object across shapes X
then Y gives wrong output at Y, reproducibly, with zero resets" observation
above is established fact.

**Practical fix applied (`tests/bench_release.py`):** `bench_returns()`'s
`CONFIGS` loop now calls `torch._dynamo.reset()` before *every* shape
(not only at padding transitions, since this instance isn't about padding).
This is safe here because each subprocess (see the `--parent-sweep`
per-algorithm isolation below) only ever accumulates on the order of a dozen
resets, far under the count that separately corrupts CUDA state when resets
accumulate unbounded in one long-lived process.

**Distinct from the illegal-memory-access crash that motivated subprocess
isolation:** the CUDA-level crash (`RuntimeError: illegal memory access`,
raised inside Inductor's autotuning harness) that originally justified
running each algorithm group in its own subprocess was separately root-
caused to the old `parallel_prefix_scan`'s `torch.flip`/`torch.roll`-heavy
fused kernel (see `tests/bench_utils.py`'s `parallel_prefix_scan`
docstring) -- a native forward-scan rewrite with no `torch.flip` compiles
clean with no crash across the full shape sweep. `dmesg` and the driver's
ECC error counters were checked and were clean throughout -- this was a
software (Inductor-generated-kernel) fault, not a hardware fault. That crash is a different
failure mode from this section's bug (a hard CUDA exception vs. a silent
wrong-but-finite numeric result caught by `assert_correctness`) and is
believed fully resolved by the rewrite. Subprocess-per-algorithm isolation
is kept regardless -- it remains a correctness-neutral, low-cost hedge, and
this section's bug demonstrates cross-shape reuse corruption can still
occur *within* a single algorithm's own subprocess, which is exactly why
the per-shape `reset()` fix above is still necessary even with isolation in
place.

### Two distinct `torch.compile` bugs live in this codebase -- do not conflate them

The two sections below describe **separate mechanisms** with **separate fixes**.
Getting them confused is the single most likely way a future change reintroduces one
while "fixing" the other:

| | Bug 2 (above and below) | Cross-object (below) |
|---|---|---|
| Trigger | **One** compiled object reused across **many shapes** | **Two separate** compiled wrappers of the **same function**, after either has compiled **2+ distinct shapes** |
| `torch._dynamo.reset()` between shapes | **Fixes it** | **No effect whatsoever** (bit-identical failure with or without) |
| Fix actually applied | Per-shape `reset()` | Subprocess isolation (separate processes, not separate resets) |
| Confirmed present in | All eight per-algorithm CONFIGS loops that reuse one object | Discounted-returns specifically, among the four algorithms with the two-wrapper structure |

### Bug 2 was incompletely fixed -- seven more CONFIGS loops had the identical exposure, unverified until they were actually isolated

The `reset()` fix above was applied only to `bench_returns()`'s CONFIGS loop, where
Bug 2 was originally found. Every other per-algorithm CONFIGS loop in
`tests/bench_release.py` -- `bench_gae`, `bench_vtrace`, `bench_retrace`'s main loop,
`bench_prefix_sum`, and all four `bench_*_truncation` functions -- has the identical
structure (one `torch.compile(...)` object reused across the same 12-shape grid) and
was equally exposed, but had never received the same reset.

This went unnoticed for a specific reason, not by luck: the cross-object bug
(below) was crashing `bench_discounted_returns_truncation()`'s process before its
loop ever ran deep enough to reach a shape pair that actually triggers Bug 2 there.
Once the cross-object bug was fixed (subprocess isolation), that loop's own Bug 2
instance surfaced directly: wrong output at `(num_envs=512, seq_len=512)` -- the
seventh shape in the grid -- 41.0% of elements mismatched, max absolute difference
~20.4, with zero `reset()` calls anywhere in that loop. Reproduced deterministically
(identical numbers across repeated runs); adding the same per-shape `reset()` used in
`bench_returns()` eliminates it; removing the reset again, in an isolated scratch
reproduction that does not touch the shipped fix, reproduces the identical failure
again, bit-for-bit.

**Shape ordering matters, shape size does not.** The immediately preceding shape,
`(512, 4096)` -- larger on both axes than the failing `(512, 512)` -- compiles and runs
correctly; `(512, 512)` only fails as the *next* shape compiled by the same object
afterward. This rules out "some shapes are just bad" as an explanation: the trigger is
accumulated state from the sequence of prior recompiles on that one object, not any
property of the failing shape itself. Recognize this bug by that signature if it
resurfaces in a different loop -- a shape that fails only when reached via one
particular preceding sequence, and passes fine on its own or via a different route in.

**The eager (uncompiled) implementation is correct at every shape tested, including
the failing one** -- confirmed directly (0 mismatched elements at `(64,512)` and
`(512,512)`, vs. the sequential reference). This is `torch.compile`/Inductor
miscompiling correct code, not a wrong baseline formula.

**Torch-version dependent, but do not read this as "fixed in 2.7.1."** All of the
above was measured on this project's pinned `torch==2.4.1+cu124`. A
project-independent standalone script (no imports from this repo -- the scan algorithm
and the wrapper function inlined directly, inputs built from scratch) reproduces the
identical failure at `(512,512)` under 2.4.1, and was also run, unmodified, against
`torch==2.7.1+cu126` in an isolated environment (same H100, same driver): **it did not
reproduce on 2.7.1 with this exact shape sequence (3/3 repeated trials, 0 mismatched
elements at every shape).** That is a narrower claim than "fixed" -- this bug depends
on accumulated Dynamo/Inductor state built up across a *specific sequence* of prior
shapes (see the shape-ordering note above: `(512,512)` fails only as the seventh shape
in this particular sequence, after these particular six predecessors). A newer torch
version could plausibly shift *which* sequence triggers the corruption rather than
eliminate the underlying defect -- three clean runs of one fixed sequence cannot
distinguish "fixed" from "this specific trigger no longer applies." Do not remove the
per-shape `reset()` fix on the strength of this result alone; see the version-state
summary below for what would actually be needed before doing that. (The cross-object
bug below was not independently re-tested on 2.7.1 at all -- this check covers only
the Bug 2 instance above.)

**Fix:** the same per-shape `torch._dynamo.reset()` is now applied to all eight
CONFIGS loops that reuse one compiled object across the grid (the one in
`bench_returns()` already had it). None of the other seven has been shown to actually
misfire on its own the way `bench_discounted_returns_truncation()` did -- but per the
masking story above, "hasn't misfired yet" was already true of that one function
before it was ever run in genuine isolation, so the fix is applied uniformly rather
than only where a failure has actually been observed. Treat any *future* CONFIGS loop
added to this file the same way: reusing one compiled object across shapes needs this
reset unless proven otherwise, not the other way around. (`bench_retrace_truncation()`'s
CONFIGS loop, added later, received the same per-shape reset from the start -- see its
own comment.)

### The reset() fix is verified for the CONFIGS loops; its effect on the post-sweep headline path is inferred, not measured

`_bench_truncation_headline()` (called once, at a single fixed shape, from
`_finalize()` after a full sweep's CONFIGS-loop functions have already run in the
same process) had shown a 24-27% run-to-run swing in GAE/V-Trace/lambda-returns
before the CONFIGS-loop `reset()` fix above landed, with discounted-returns and
Retrace comparatively stable (~2.6%/3.5%). The theory: `_bench_truncation_headline()`
builds its own fresh `torch.compile(...)` wrappers at one new shape, late in a
process that has already accumulated Dynamo/Inductor state from many prior
compiles at other shapes in the CONFIGS grid -- the same cross-shape corruption
mechanism as Bug 2 above, just surfacing as timing noise here rather than wrong
values, since `_bench_truncation_headline()` is never itself checked bit-for-bit
against a fixed expected number.

Attempted to confirm the fix directly: four back-to-back calls to
`_bench_truncation_headline()` alone, each in its own fresh, otherwise-idle
process (H100, 2026-07-30). Result was inconclusive by construction, not
reassuring: an isolated call like this never runs any CONFIGS-loop code first, so
it never had the contamination path available to it *either before or after* the
fix -- tight numbers here (V-Trace 5.2%, lambda-returns 3.9%, GAE 11.0%,
discounted-returns 6.4%, Retrace 5.4% spread across the 4 runs) mostly show the
compiled objects/kernels are individually stable, not that the fix resolved the
original post-sweep contamination. GAE's 11.0% is worth flagging on its own terms
(down substantially from 24%, but still ~2x the other four's spread here, driven
by one low outlier run) but the isolated-process methodology cannot speak to
whether that residual is the same phenomenon as the original swing.

The one data point actually produced under the original failure conditions (full
sweep, `_bench_truncation_headline()` running post-CONFIGS-loops, fix active) is
the H100 `--parent-sweep` candidate staged 2026-07-30: GAE headline = 1.9x,
within the pre-fix 1.70-2.11x range. Consistent with the fix working, but a
single post-fix sample cannot distinguish "fixed" from "still swings and this
run landed in-range" -- treat the post-sweep headline path as *plausibly* fixed
by the same mechanism as the CONFIGS loops, not *confirmed* by direct repeated
measurement under matching conditions. Confirming it properly would mean
rerunning the full sweep 3-4 times and comparing the embedded headline number
each time (each run ~12 min on H100) -- not yet done.

### A third, distinct `torch.compile` bug: cross-object miscompilation -- not Bug 2, and `reset()` does not help

A **second, separate** `torch.compile(...)` wrapper of the *same* underlying function
gives wrong output the first time it is invoked, if *any earlier* wrapper of that
identical function has already been compiled at two or more distinct shapes anywhere
earlier in the same process -- regardless of which object did the compiling,
regardless of `torch._dynamo.reset()` usage between those shapes, and regardless of
whether the second wrapper's own shape is one it has seen before. This is a different
mechanism from Bug 2 (one object, many shapes) -- see the contrast table above.

**What was observed (fact, reproduced deterministically, 3/3 identical):**
- Object A: a `torch.compile(...)` wrapper of `vectorized_discounted_returns_with_
  truncations`, compiled and invoked at two or more distinct `(num_envs, seq_len)`
  shapes. Object B: a **separate**, freshly-created `torch.compile(...)` wrapper of
  the **identical** function, compiled and invoked at `(64, 512)` -- the first shape it
  has ever seen. Object B's output is wrong: 40.7% of elements mismatched, max
  absolute difference ~15.6, identical across every repeated trial.
- **Threshold is exactly two distinct shapes.** Object A compiled at only one
  distinct shape (even recompiled at that same shape via an intervening `reset()`)
  never corrupts object B. Two or more distinct shapes always does -- tested at 2, 3,
  6, 12, and via the unrelated production-regime grid alone (11 shapes, no overlap
  with the CONFIGS grid), all with identical results.
- **`torch._dynamo.reset()` has no effect.** Removing every `reset()` call from
  object A's shape loop entirely produces a **bit-identical** failure -- same element
  count, same max absolute difference -- as keeping them. The mitigation that fixes
  Bug 2 does nothing for this bug.
- **Requires the same underlying function.** Object A wrapping a *different*
  function (`vectorized_gae_with_truncations` or
  `vectorized_lambda_returns_with_truncations`) never corrupts a discounted-returns
  object B. Cross-function compile history is harmless.
- **Observed only for discounted-returns among the four algorithms with this
  structure.** GAE, V-Trace, and λ-returns each have the identical two-wrapper
  structure (a `bench_X()` and a `bench_X_truncation()`, both `torch.compile`-ing the
  same `vectorized_X_with_truncations` function independently) and were tested under
  the identical A-then-B pattern. None reproduced the corruption.

**What is interpretation, not proof:** why discounted-returns specifically, and not
the other three sharing the same structure, is **not understood** -- not root-caused
via Inductor-generated-code inspection, no guard-cache or codegen-cache comparison
done. Do not read "GAE/V-Trace/λ-returns didn't reproduce it" as "they are safe by
design" -- only as "not observed to fail under the conditions tested."

**Fix:** subprocess isolation, not `reset()` and not sharing the compiled object
between the two call sites. A process boundary protects all four algorithms with
this structure regardless of *why* three of them don't currently misfire under it --
relying on an unexplained safety property is not acceptable for numbers a release
publishes. `tests/bench_release.py`'s `--variant {plain,truncation}` flag and
`_PARENT_SWEEP_GROUPS` now run each algorithm's plain and truncation-path tables in
separate subprocesses for exactly these four algorithms; retrace,
eligibility_traces, and prefix_sum have only one `torch.compile(...)` wrapper of
their function each and need no such split. This does **not** protect against the
same mechanism arising between any two wrappers of a function this repo hasn't
tested this way, and does **not** explain or fix whatever Inductor/Dynamo behavior
actually causes it.

**This bug masked Bug 2's presence in `bench_discounted_returns_truncation()`.** The
first full-sweep run to reach this function crashed on this bug (object A =
`bench_returns()`'s compiled object, object B =
`bench_discounted_returns_truncation()`'s own, corrupted at the very first CONFIGS
shape). That crash meant the loop never ran far enough to reach the seventh shape
where Bug 2 independently lives in that same function -- so Bug 2's instance there
was invisible until this bug was fixed first. The other seven CONFIGS loops that
also lacked Bug 2's `reset()` protection had never been exercised in genuine
isolation before either, for the same class of reason (each either doesn't share a
process with a same-function second wrapper, or hadn't been split into its own
subprocess yet) -- "no failure observed" in any of them prior to this investigation
is not evidence they were safe, only that they hadn't really been tested.

### Current version-verification state -- read this before upgrading torch or removing either mitigation

Recorded explicitly so this doesn't have to be re-derived later:

- **Bug 2** (per-shape `reset()`, all eight CONFIGS loops): did **not** reproduce on
  torch 2.7.1 with the one shape sequence tested (3/3 clean). Not tested against any
  other sequence, any other function, or any other torch version. This is not
  evidence the underlying Inductor defect is fixed -- only that this one sequence no
  longer triggers it on this one version.
- **Cross-object bug** (subprocess isolation): unverified on anything past torch
  2.4.1. No re-test attempted.
- **Subprocess isolation is retained for a second, independent reason** beyond the
  cross-object bug: it also hedges the unrelated historical illegal-memory-access
  crash described earlier in this section (root-caused separately, to the old
  `parallel_prefix_scan`'s `torch.flip`/`torch.roll` kernel, believed fixed by that
  kernel's rewrite -- see that discussion above). Removing subprocess isolation would
  reopen exposure to that crash regardless of the cross-object bug's status.
- **Both mitigations are retained deliberately, not out of caution-by-default.** If
  this project ever upgrades its pinned torch version, re-run the *direct*
  verification for both bugs (the standalone script for Bug 2; the object-A-then-
  object-B isolation script for the cross-object bug) before removing either one. A
  green full sweep is not sufficient evidence to remove a mitigation -- that is
  exactly the failure mode this whole investigation started from: both bugs produce
  finite, plausible-looking output, and both were previously masked by something
  else (a crash, then a different bug) rather than by actually being absent.

### Known gap: the monotonicity gate cannot catch a baseline that becomes silently faster-but-wrong

`check_monotonic_grid` (used by every `bench_*` function's monotonicity check) only
ever watches `triton_ms` -- the Triton kernel's own timing, checked for monotonic
scaling within a single run. Nothing in this codebase compares a `torch.compile(...)`
baseline's measured timing *across separate sweep runs*, and nothing checks whether
a baseline's timing shift between two runs correlates with anything suspicious. A
baseline that silently starts computing something wrong -- while happening to also
run faster or slower -- produces no signal any automated gate here would flag; it
would only surface (if at all) via a human manually diffing two candidates' rendered
tables side by side after the fact, exactly as happened for the observation below.
This is a known gap in the gating as it exists today, not something being fixed as
part of this work -- recorded so it isn't rediscovered as a surprise later.

### Unresolved historical observation: V-Trace-with-truncations' `(512,512)` timing shift across the pre/post-mitigation sweeps

**This concerns a superseded candidate, not shipped numbers -- read the framing
before the data.** Comparing the pre-mitigation (2026-07-28) and post-mitigation
(2026-07-29) H100 sweep candidates, the largest single-cell change anywhere in
either sweep was V-Trace-with-truncations at `(num_envs=512, seq_len=512)`: the
full-call ratio moved from 4.9x to 2.9x, entirely via the *baseline's* measured time
(`compile(vec-trunc)`: 0.257ms → 0.158ms; the Triton kernel's own time was unchanged
within noise). The correctness gate passed for this exact config in the
post-mitigation run, with both mitigations (per-shape `reset()` in
`bench_vtrace_truncation()`'s loop, subprocess isolation separating it from
`bench_vtrace()`) confirmed structurally active for that run.

What makes this worth recording rather than dismissing as ordinary jitter: the
*pre-mitigation* run had **both** risk structures for the two bugs above present in
one process simultaneously (`bench_vtrace()` and `bench_vtrace_truncation()` shared
a single subprocess before the variant split, and that loop had no `reset()`) -- the
same shape, `(512,512)`, that is Bug 2's confirmed trigger for discounted-returns.

**This was never reconstructed or resolved.** The exact combined scenario that would
settle it -- object A = `bench_vtrace()`'s real compile history, object B =
`bench_vtrace_truncation()`'s full 12-shape loop with no `reset()`, checked against
the reference at `(512,512)` -- was never run for V-Trace; only a single-shape
cross-object check and a same-object-across-shapes check were run for it separately,
neither reproducing anything, and neither matching this exact combined condition.
The two candidate explanations -- the pre-mitigation baseline was actually
miscompiled at this shape (same class of bug as discounted-returns, unconfirmed
instance), versus ordinary `torch.compile` autotuning variance picking a different
kernel configuration between two separate process runs (V-Trace's baseline is a more
complex computation than discounted-returns', plausibly more autotuning-sensitive on
its own) -- remain unseparated. Do not read this as an open defect in any currently
staged or published number: the current (post-mitigation) candidate's value at this
cell is correctness-gate-verified. It is an unresolved question about a candidate
that no longer exists as the current one, kept here only so the observation isn't
lost or re-litigated from scratch later.

---

## The log-space baseline underflow -- why `compile(vec)` is a doubling scan, not a cumsum

Five of the six "vectorized" PyTorch comparison baselines (GAE, V-trace,
λ-returns, eligibility-traces, Retrace) originally computed their
decay-weighted scan via `exp(cumsum(log(decay.clamp(min=1e-38))))`. The
`clamp(min=1e-38)` exists only so `log(0)` at an episode-terminated step
(`decay=0` there) doesn't produce `-inf` -- but it is exactly what breaks the
formula: each termination contributes `log(1e-38) ≈ -87.5` to the running
suffix sum, and float32 underflows (flushes to zero, then `exp` propagates
inf/nan downstream) anything below `≈ -103.3`. At this project's ~5%/step
termination rate, 2-3 terminations anywhere in a window -- expected within
the first couple dozen steps of *any* row, independent of overall
`seq_len` -- is enough. This is not an extreme-scale-only failure: it hit
90-99% of output elements at every size this project's release table
actually uses, from 64×512 up to the production regime at 4096×128, not
only the long-`seq_len` regime `benchmarks/chunked_scan.md` originally
probed.

**Fix:** `compile(vec)` no longer uses the log-space formula anywhere.
GAE/V-trace/λ-returns/discounted-returns became thin wrappers around their
already-correct `*_with_truncations` siblings (`truncateds=0`,
`bootstrap=0`), which were already built on `parallel_suffix_scan` -- a
linear-space `log2(T)`-doubling associative scan with no `log`/`exp`
anywhere. Eligibility-traces and Retrace were rewritten directly on the
same primitive (`parallel_prefix_scan` / `parallel_suffix_scan`
respectively). The old `compile(assoc)` column in `benchmarks.md` is gone
because there is no longer a separate specialized no-truncation baseline to
compare it against -- the with-truncations implementation *is* the baseline
now, called with zero truncations.

**Caveat: this is the strongest *correct* baseline built, not necessarily
the strongest possible one.** The doubling-scan baseline pays 6-12 kernel
launches per call (growing with `seq_len`, one per doubling step), against
1-2 launches for the Triton kernel it's compared to. Isolated primitive
timing at a fixed shape found the (underflow-buggy, only safe without
terminations) log-space cumsum takes ~37µs across 2 launches, versus the
doubling scan's ~148µs across 6 launches -- a 4x gap attributable to launch
count alone, not to the underlying arithmetic. A numerically-stable
non-log-space cumsum formulation (e.g. a segmented/blocked scan that
periodically renormalizes instead of ever taking one running log-sum) could
plausibly avoid the underflow without paying the doubling scan's launch
count, and would be faster. This means published ratios like "N× vs.
torch.compile" describe the doubling-scan baseline specifically, not an
upper bound on what a correct PyTorch baseline could achieve -- a materially
different claim than the older (numerically wrong) few-launch-cumsum
numbers this project published before the underflow was found. Add-only, not
yet built: nobody has implemented the faster correct baseline to check how
much of the gap it would close.

**Strategy for when that baseline gets built.** The 4x gap above is
launch-count-bound, not arithmetic-bound: torch.compile/Inductor does not
fuse this elementwise-then-scan pattern into as few kernels as a
hand-written Triton kernel can, so a faster correct PyTorch baseline
plausibly closes part of the gap but not all of it -- the launch-count
floor is architectural, not an artifact of this particular baseline being
unoptimized. If/when a faster correct baseline is built (segmented or
periodically-renormalizing cumsum, per the paragraph above), re-running the
comparison and republishing lower ratios is expected routine maintenance,
not a regression or a threat to the library's validity -- the same way this
project's move from the broken log-space baseline to the current
doubling-scan one produced a one-time, honest correction rather than a
loss. Any published "N× vs. torch.compile" ratio should be read as bounded
by the strongest *correct* baseline known at publication time, with the
mechanism (single-kernel fusion vs. torch.compile's multi-launch
decomposition of the same algorithm) as the durable claim, not the specific
number. Same discipline applies to torch.compile/Inductor itself improving
this pattern's fusion in a future PyTorch release -- pin the PyTorch
version per benchmark (already done) and re-verify on major version bumps
rather than assuming old ratios still hold.

---

## Per-GPU floor calibration is a correctness requirement, not an optimisation

`tests/bench_safeguard.py`'s CI-facing performance gate times the Triton
kernel against `torch.compile(vec)` at one small shape (128×1024) and
asserts a minimum speedup. At that shape, both sides' *device* kernel time
is only a few microseconds -- dwarfed by ~25-30µs of fixed CUDA
dispatch/sync overhead that both sides pay identically regardless of which
kernel is actually faster. A slower GPU's kernel work dominates that fixed
cost, so its measured wall-clock ratio reflects real kernel quality; a
faster GPU's kernel work shrinks by an order of magnitude while the ~25-30µs
floor stays fixed, compressing the wall-clock ratio toward 1x independent of
kernel quality. Concretely: eligibility-traces' kernel is a genuine,
reproducible 2.7x faster in raw device time on H100/H200-class hardware,
yet the reported *wall-clock* speedup compresses from ~11x on an older,
slower card to under 2x on the faster one -- launch-overhead amortization,
not a kernel regression.

**Consequence:** a floor calibrated on one GPU does not transfer to a
different one, faster or slower. `_FLOOR_TABLE` in `bench_safeguard.py` is
keyed by device-name substring (see that file's own module comment for the
mechanism and the currently-calibrated cards); an unrecognized GPU must
`pytest.skip()` loudly rather than silently reuse another card's floor --
reusing a floor across GPU families is exactly the mistake that let this
gate pass vacuously in the past (calibrated on the wrong card entirely, see
`bench_safeguard.py`'s own history comment for that story).

---

## Open question: the H100-vs-RTX margin direction is shape-dependent, not fixed -- no consumer-vs-datacenter framing is supportable

Comparing the two GPUs at two different shapes gives two different answers
about which card shows the larger Triton-vs-`torch.compile` margin. Both are
correct measurements; neither generalizes to "this card is better."

**At (num_envs=128, seq_len=1024)** -- the `bench_safeguard.py` floor-
calibration shape, 3 independent runs each, min-of-5-trials per run (see
`_H100_MEASURED_2026_07_28` / `_RTX_2000_ADA_MEASURED_2026_07_30` in that
file):

| algo | H100 (min-of-3) | RTX (min-of-3) | RTX vs H100 |
|---|---|---|---|
| gae | 3.500 | 3.052 | -13% |
| vtrace | 3.167 | 2.758 | -13% |
| retrace | 2.164 | 1.794 | -17% |
| lambda_returns | 2.938 | 2.671 | -9% |

RTX is *smaller* than H100 across every algorithm at this shape.

**At (num_envs=4096, seq_len=128)** -- the production-regime shape the README
summary table uses (see `benchmarks.md`'s v0.1.1 production tables, `vs vec
(full-call)`):

| algo | H100 | RTX | RTX vs H100 |
|---|---|---|---|
| GAE | 3.50 | 5.15 | +47% |
| V-Trace | 3.40 | 5.66 | +66% |
| Retrace | 1.83 | 1.60 | -13% |
| lambda-returns | 3.03 | 4.57 | +51% |

RTX is *larger* than H100 for GAE/V-Trace/lambda-returns at this shape --
the opposite direction from the floor-calibration shape. Retrace is smaller
on RTX at both shapes; it does not reverse.

**Decomposition -- what actually moves.** Comparing each card against
*itself* across the two shapes isolates the driver:

| algo | H100 @128×1024 | H100 @4096×128 | RTX @128×1024 | RTX @4096×128 |
|---|---|---|---|---|
| GAE | 3.500 | 3.50 | 3.052 | 5.15 |
| V-Trace | 3.167 | 3.40 | 2.758 | 5.66 |
| lambda-returns | 2.938 | 3.03 | 2.671 | 4.57 |
| retrace | 2.164 | 1.83 | 1.794 | 1.60 |

H100's margin is roughly flat across the two shapes (within ~0.3x either
way). RTX's margin nearly *doubles* at (4096,128) versus (128,1024) for
GAE/V-Trace/lambda-returns specifically. **That is what flips the cross-card
comparison -- RTX getting relatively better at the production shape, not
H100 degrading.** Retrace moves the same direction on both cards (down at
the production shape vs. the floor-calibration shape), just from different
starting points, and stays RTX-lower than H100 at both.

**What this means, stated plainly:** the direction of "which card shows the
larger margin" is a function of problem shape, not a fixed property of
either card. Any framing along the lines of "consumer GPUs show [larger /
smaller] margins than datacenter GPUs" is directly contradicted by this
data -- the sign of the comparison flips depending on which shape you look
at. Neither this project's benchmarks nor the forthcoming paper should make
that claim in either direction until the mechanism is understood.

**The mechanism is not understood.** No hypothesis here has been checked
against data yet. Plausible candidate directions (untested): the vec
baseline's log2(seq_len)-doubling-scan launch count differs between
seq_len=1024 (~10 launches) and seq_len=128 (~7 launches), and whatever
governs Triton's own launch/wrapper overhead may scale differently with
num_envs than with seq_len -- but this is speculation, not an explanation
derived from measurement.

**The obvious next probe needs no GPU.** Both staged candidates already
carry a `dev` (device-only) column alongside `triton` (full-call) at every
config, in both the floor-calibration-adjacent full CONFIGS grid and the
production-regime table (see either GPU's section in `benchmarks.md`). Since
this data already exists for both cards at both shapes, decomposing the
reversal into "device-time behavior" vs. "launch/wrapper-overhead behavior"
is a pure analysis pass over already-collected numbers -- no new sweep, no
GPU time, before reaching for a hardware-level explanation.

---

## One helper, two monotonicity gates: different shape-pair severity, by design

`check_monotonic_grid` (in `bench_utils.py`) is the single implementation
behind two separate monotonicity checks, and they are deliberately not the
same strictness. Do not "fix the inconsistency" between them -- it isn't one.

**`tests/bench_safeguard.py`'s `test_monotonicity_gate`** (the PR-blocking
safeguard suite) compares exactly one pair per algorithm: `_MONO_SMALL =
(128, 512)` vs. `_MONO_LARGE = (512, 4096)`, a 32x element jump
(65,536 -> 2,097,152). Both ends sit well clear of the small-shape corner
where fixed CUDA dispatch/sync overhead dominates (see "Per-GPU floor
calibration" above) -- both are compute/bandwidth-bound. A violation at this
gap cannot be explained by launch-overhead noise at either end, so it means
the kernel's own wall-clock time got smaller as the problem got strictly
bigger -- a genuine regression. This check is correctly a blocking PR gate.

**`tests/bench_release.py`'s per-algorithm release-sweep monotonicity
checks** run the same helper over the *full* `CONFIGS` grid (and the
production-regime grid), which necessarily includes adjacent,
overhead-dominated pairs -- e.g. `seq_len` 80->128 at a fixed `num_envs` in
the production table. At that granularity, a 3-8% negative slope is expected
GPU clock-ramp/launch-overhead jitter, not a regression (see the concrete
RTX 2000 Ada instance immediately below). This is why `bench_release.py`
treats its own monotonicity result as advisory: `_finalize()` stages the
candidate regardless of violations and only affects the process's exit
code -- it does not block staging the way the safeguard suite blocks a PR.

**The two gates ask structurally different questions with the same code.**
Widely-separated, compute-bound shapes -> a violation is a real regression,
block on it. Densely-sampled, overhead-included shapes -> some negative
slopes are expected noise, treat violations as advisory. Do not narrow
`_MONO_SMALL`/`_MONO_LARGE` toward each other (that pulls the safeguard gate
into the overhead-dominated corner and starts blocking PRs on jitter), and do
not make the release sweep's check blocking or the safeguard's advisory --
either change silently erases the distinction this section exists to record.

### Concrete instance: RTX 2000 Ada release-candidate sweep (2026-07-30) -- 5 monotonicity violations, all in the overhead-dominated corner, none at the safeguard's blocking pair

The RTX 2000 Ada `--parent-sweep` run staged into
`docs/benchmark-history/unreleased.md` fired the release sweep's
(advisory) monotonicity gate 5 times:

```
vtrace:              num_envs=512, seq_len 128->512:   0.0608ms -> 0.0571ms  (-6.2%)
vtrace:              seq_len=128, num_envs 512->4096:  0.0608ms -> 0.0567ms  (-6.8%)
eligibility-traces:  num_envs=4096, seq_len 80->128:    0.0521ms -> 0.0479ms  (-8.1%)  [production regime]
eligibility-traces:  num_envs=8192, seq_len 80->128:    0.0530ms -> 0.0492ms  (-7.1%)  [production regime]
prefix_sum:          num_envs=8192, seq_len 80->128:    0.0483ms -> 0.0467ms  (-3.4%)  [production regime]
```

Every one of these is a small-elements/short-`seq_len` pair (largest absolute
time involved is 0.061ms) squarely in the launch-overhead-dominated corner,
and none of them is anywhere near the safeguard suite's `(128,512)` /
`(512,4096)` pair. The correctness/baseline-validity gate (`assert_correctness`
against both the Triton kernel and the compiled vec baseline) passed for
every config in this sweep, and the safeguard suite's own blocking
`test_monotonicity_gate` was not run as part of this release sweep at all --
this is exactly the scenario the section above describes: fine-grained,
overhead-corner jitter that the release sweep correctly tolerates rather than
blocking on. The candidate was staged with these violations present in the
console gate summary only, not reflected in the staged tables themselves.

### Concrete instance: H100 release-candidate sweep (2026-07-30) -- 14 monotonicity violations, all in the overhead-dominated corner, zero in Retrace

The H100 `--parent-sweep` run staged into `docs/benchmark-history/unreleased.md`
(the corrected v0.1.1 candidate, CONFIGS-loop `reset()` fix and
`bench_retrace_truncation()` both active) fired the release sweep's (advisory)
monotonicity gate 14 times, spanning gae, vtrace, discounted_returns,
eligibility_traces, and prefix_sum (both the `triton_ms` and production-regime
checks) -- none in lambda-returns' `triton_ms` check, none anywhere in Retrace
(plain, production, or the new truncation table):

```
gae:                 seq_len=1024, num_envs 128->256:        0.0377ms -> 0.0342ms  (-9.1%)
vtrace:              seq_len=1024, num_envs 128->256:        0.0390ms -> 0.0380ms  (-2.5%)
vtrace:              seq_len=128, num_envs 4096->8192:       0.0452ms -> 0.0404ms  (-10.7%)  [production regime]
lambda-returns:      seq_len=128, num_envs 8192->16384:      0.0414ms -> 0.0377ms  (-9.0%)   [production regime]
discounted_returns:  seq_len=1024, num_envs 128->256:        0.0326ms -> 0.0303ms  (-7.1%)
discounted_returns:  seq_len=128, num_envs 512->4096:        0.0315ms -> 0.0308ms  (-2.2%)
discounted-returns:  num_envs=8192, seq_len 80->128:         0.0387ms -> 0.0324ms  (-16.1%)  [production regime]
discounted-returns:  num_envs=8192, seq_len 80->16384(*):    0.0387ms -> 0.0350ms  (-9.5%)   [production regime]
eligibility_traces:  num_envs=512, seq_len 2048->4096:       0.0311ms -> 0.0298ms  (-4.0%)
eligibility-traces:  num_envs=8192, seq_len 80->128:         0.0303ms -> 0.0295ms  (-2.5%)   [production regime]
eligibility-traces:  seq_len=128, num_envs 4096->8192:       0.0303ms -> 0.0295ms  (-2.4%)   [production regime]
prefix_sum:          num_envs=512, seq_len 128->512:         0.0282ms -> 0.0268ms  (-5.0%)
prefix_sum:          num_envs=32768, seq_len 80->128:        0.0470ms -> 0.0427ms  (-9.1%)   [production regime]
prefix_sum:          num_envs=38400, seq_len 80->128:        0.0497ms -> 0.0460ms  (-7.5%)   [production regime]
```
(*) the seq_len axis is fixed at 80 here; the pair varies num_envs 8192->16384.

Every one is a small-elements/short-`seq_len` pair (absolute times 0.027-0.050ms
throughout, percentages 2.2-16.1%) in the same launch-overhead-dominated corner
as the RTX instance above, and none is near the safeguard suite's `(128,512)` /
`(512,4096)` pair. The correctness gate passed for every config reached across
all 12 `--parent-sweep` subprocess groups, including both of
`bench_retrace_truncation()`'s gates (Triton vs `reference_retrace`, and
`compiled_vec_trunc` vs `reference_retrace`) at all 12 CONFIGS shapes -- and
Retrace's monotonicity checks (plain, production, and truncation) all passed
cleanly, no violations. The candidate was staged with these 14 violations
present in the console gate summary only, not reflected in the staged tables
themselves.

---

## Retrace's register-spill ceiling and its reroute (see `retrace.py` for the authoritative account)

`compute_retrace` dispatches on `_TRITON_SEQ_LEN_CEILING = 2048`:
below it, the fused Triton kernel; above it, a reroute through the generic
`_run_scan` associative-scan path. `src/rl_triton/ops/retrace.py`'s own
comment block above that constant is the authoritative source for the
mechanism (register pressure from re-reading the 3D action-probability
tensor a second time for `c[t+1]`, pinning `ptxas` at 128 regs/thread and
25% occupancy from `seq_len=4096` onward) and for the reroute's own
performance (measured 0.38-0.47x vs. `torch.compile` -- **a smaller loss,
not a win**; below `seq_len=8192` the un-rerouted fused kernel is actually
still faster than the reroute despite already losing to `torch.compile`
itself). Read the code comment there rather than duplicating the numbers
here; this note exists only so the ceiling's rationale isn't lost with the
session reports that originally investigated it.

---

## GAE's share of a real PPO step is small; end-to-end speedup is a wash at realistic network sizes

Two independent end-to-end PPO measurements (different `(num_envs, seq_len,
hidden_size)` configurations, one interleaving eager/compiled A/B runs, one
a single realistic Isaac-Gym-Ant-sized config with a full forward/GAE/
loss/backward/Adam step) converge on the same conclusion: GAE is a small
enough fraction of total step time that its kernel-level speedup does not
show up end-to-end once a real policy network shares the step.

| hidden size | GAE share of step (approx.) | end-to-end speedup |
|---|---:|---:|
| small (256,256) | 0.3% – 3.6% | up to ~1.18x |
| realistic (1024,1024) | 0.07% – 0.6% | **~1.00x -- no measurable difference** |

Full per-stage breakdown (4 configs, all four pipeline stages) is in
`docs/benchmark-history/ppo-e2e-measurement-2026-07-25.md`; this section is
the condensed takeaway, not a replacement for it. The larger hidden size is
the realistic case in both measurements -- backward
pass dominates the step (60-66% of total time) at that size, and GAE's
isolated ~1.9x device-level kernel speedup becomes invisible in full
training throughput. **This is a bound the paper's evaluation section must
respect: the isolated-microbenchmark speedup claim holds, an end-to-end
training-speed claim from this kernel alone does not**, at realistic network
sizes. No win/lose framing beyond that is warranted -- GAE was never the
bottleneck this optimization needed to target end-to-end; it targets the
per-call kernel cost directly, which is a legitimate but narrower claim.

---

## Two more baseline bugs found while auditing benchmark correctness (algorithm bugs, not `torch.compile` artifacts)

Distinct from the `torch.compile`-specific bugs above and from the
log-space underflow: two of the vectorized PyTorch baselines had their own
independent formula bugs, both caught only because baseline output started
being correctness-checked against the sequential reference for the first
time (previously only the Triton side was checked).

**`vectorized_discounted_returns`:** used `log(gamma) * done` inside its
(pre-underflow-fix) log-space formula instead of an actual reset -- a
`done=1` step contributed `0` to the log-sum rather than terminating the
discount chain, so the accumulator never reset. Result: finite but wrong
output (max abs error ~43 vs. the sequential reference on a 64-step test) --
a different failure mode from the inf/nan underflow affecting the other
five baselines, since this one never produced a value extreme enough to
underflow.

**`vectorized_episodic_prefix_sum`:** the old formula
(`running = cumsum(inputs); boundary = running*dones; offset =
cumsum(boundary)-boundary; result = running-offset`) resets one step later
than this codebase's convention (`done[t]=1` means the reset applies *at*
`t` itself, not at `t+1`). Wrong at every shape ever benchmarked with it --
44-95.8% of elements, including the smallest configs -- because the
baseline's own correctness had never been checked, only timed. **Fix:**
replaced with `parallel_prefix_scan(inputs, 1.0 - dones)`, which encodes the
reset convention directly rather than reconstructing it after the fact.
Any previously-published prefix-sum speedup number predates this fix and
should not be trusted; the current `benchmarks.md` table reflects the
corrected baseline.

---

## rl-triton vs. PufferLib: a real design tradeoff at short-horizon, high-env-count workloads -- not a bug, not fixed by the warps-table fix

At very short horizons and very high environment counts (`num_envs >= 4096`,
`seq_len <= 16`; the crossover point shifts to `seq_len ~= 128` by
`num_envs = 32768`), PufferLib's hand-written sequential-per-row CUDA kernel
measures faster than rl-triton's one-program-per-environment Triton kernel,
on both device time and wall-clock. Mechanism: PufferLib's per-row O(T) cost
is nearly `num_envs`-insensitive (thread count scales directly with
`num_envs`, packing the GPU efficiently even at tiny per-row problem sizes),
while rl-triton's one-program-per-env grid design pays more grid waves per
SM as `num_envs` grows (up to ~62 waves at `num_envs=32768` vs. ~8 at 4096)
-- narrowed by the small-`BLOCK_SIZE` warps-table fix (see each op's `_WARPS`
comment) but not eliminated by it, since that fix addresses per-program warp
over-provisioning, not grid-wave count.

**Consequence for anyone choosing between these kernels:** rl-triton's
design is right for the long-horizon, moderate-env-count on-policy rollout
regime this project targets (see the production-regime table in
`benchmarks.md`), and measurably wrong for the shortest-horizon,
highest-env-count corner of massively-parallel simulation (Isaac Gym/Isaac
Lab style, tens of thousands of envs at single-digit-to-low-double-digit
horizons). This is a genuine per-workload tradeoff, not a defect to fix --
but it means "rl-triton is faster than PufferLib" is not a universal claim;
see `benchmarks/pufferlib.md` for where each kernel wins.

---

## Eligibility-traces' apparent device-time ceiling is a small-grid occupancy ramp, not an architectural limit

An earlier hypothesis attributed eligibility-traces' device-time speedup
plateauing around ~2.4-2.5x to L2-cache residency (the working set falling
out of L2 above some size). That hypothesis was tested directly and
refuted: no configuration ever exceeds HBM peak bandwidth, and there is no
discontinuity at the size where the working set crosses the L2 capacity
boundary. The actual mechanism is a small-grid occupancy ramp -- one program
per environment means small `num_envs` doesn't generate enough concurrent
thread blocks to saturate all SMs and hide memory latency. Extending the
swept `num_envs` range well past where earlier measurements stopped showed
achieved bandwidth still climbing (roughly 67% -> 78% -> 84% of peak) with
no plateau in sight -- the "ceiling" earlier measurements read as
architectural was an artifact of not having swept `num_envs` far enough.
Addressable in principle (persistent-kernel or multi-env-per-program
launch), not attempted -- out of scope until a real workload needs
eligibility-traces at very small `num_envs`.

---

## Truncation-path speedup grows with scale -- durable, unlike the plain path's high-end compression

For the four algorithms with a truncation-aware path (GAE, V-trace,
λ-returns, discounted-returns), the Triton-vs-baseline speedup on that path
*grows* with `num_envs`/`seq_len` once the baseline is the corrected
doubling-scan implementation (e.g. GAE: 2.76x -> 9.88x from 512 to 8192
envs at `seq_len=2048`) -- the opposite of the plain path's behavior on
fast GPUs (see the per-GPU floor calibration note above, where the ratio
*compresses* toward 1x as hardware gets faster). Reason: the baseline's
multi-launch (6-12 launches) doubling-scan construct moves far more total
HBM traffic than the Triton kernel's single fused launch (an analytical
estimate puts it at roughly 45x more bytes moved, treated as a lower
bound), so the gap widens rather than narrows as problem size grows and
launch-count overhead stops being the dominant cost on either side. Unlike
the plain path, wall-clock and device time track within 1-2% of each other
throughout this path, because the baseline's own multi-launch overhead is
large enough that the fixed ~26µs dispatch cost is a small fraction of
total time on both sides -- the truncation path's numbers are less sensitive
to the same GPU-speed effect that makes the plain-path floor
hardware-specific.
