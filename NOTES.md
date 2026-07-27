# Implementation Notes

## Why there is no chunked fused V-Trace kernel

`compute_vtrace_fused` eliminates all intermediate tensor allocations and kernel
launches by doing everything — IS ratio computation, backward scan, target
construction, and advantage computation — inside a single Triton kernel per
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
   advantage of fusion — eliminating intermediate allocations and launch
   latency — is a much smaller fraction of total runtime than at shorter
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

- **Non-contiguous input tensors** — any `u` or `v` tensor passed to `_run_scan`
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
buffer), use the start-of-episode convention directly — place `done=1` at the
first timestep of each new episode — and no shift is needed.

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

## Double HBM read of the values tensor in gae_fused_kernel — investigated, not a win

The GAE fused kernel (`gae_fused_kernel`) loads the values tensor twice from
HBM — in *both* the `HAS_TRUNCATIONS=True` and `HAS_TRUNCATIONS=False` paths
(and in the older non-fused `gae_kernel`), not only the truncations path as
originally noted here:

1. `tl.load(values_ptr + base + rev, ...)` — gives `v_t`
2. `tl.load(values_ptr + base + rev + 1, ...)` — gives `v_{t+1}`

These are two separate `tl.load` calls at different pointer offsets; they are
**not** loaded once into SRAM and then indexed at two positions. SRAM in
Triton is used for inter-thread communication during the scan, not as a
random-access scratchpad.

**Measured (H100, `tests/h100_short_horizon_l2_retrace_ppo_report.md`-style
methodology): adopting the `lambda_returns_fused_kernel` pre-shift strategy
makes this slower, not faster.** Implemented as an A/B kernel
(`gae_fused_kernel_preshift`, taking a caller-supplied `next_values` tensor
instead of the second in-kernel load) and benchmarked against the shipped
kernel:

- **Correctness:** bit-identical output (max diff = 0.0) across 64×512,
  512×4096, 128×8192 — confirms the two formulations compute the same math.
- **ptxas metadata @ 512×4096:** identical — 84 regs/thread, 0 spills, in
  *both* variants. Unlike Retrace's redundant 3D reload (which fed a
  register-hungry `[BLOCK_SIZE, ACTION_BLOCK]` reduction), `v_next` here is a
  single scalar-per-lane float — removing its load doesn't touch register
  pressure at all. GAE was never spilling in the first place.
- **Kernel-only device time** (pre-shift kernel vs shipped kernel, excluding
  the cost of building `next_values`): pre-shift is *slightly slower* at
  seq_len≥4096 (0.90–0.94x). The two loads in the current kernel are offset
  by exactly one element — well within a single L1/L2 cache line — so they
  already share most of their traffic; there was no real second HBM
  round-trip to eliminate. Reading from a separate `next_values` tensor
  instead loses that locality.
- **Full path including materialization** (`next_values[:, :-1] =
  values[:, 1:]`, one extra kernel launch + full read/write pass, mirroring
  what a caller or wrapper would have to do since `compute_gae`'s public API
  takes only `values`, not a pre-shifted `next_values` like
  `compute_lambda_returns` does): **0.52–0.73x — 30–90% slower** across
  64×512 through 512×8192. The extra launch dominates any theoretical
  bandwidth saving, consistent with this project's repeated finding
  (short-horizon sweep, Retrace ceiling fallback) that added kernel launches
  are costly relative to a few extra bytes of already-cached HBM traffic.

**Verdict: not applied.** The premise (this is meaningfully "double HBM
traffic") doesn't hold up under measurement — it's two cache-friendly loads
of adjacent addresses, not two independent full tensor reads — and the
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
produce **silently wrong numeric output — not NaN/inf, not a crash** — which
is why they went unnoticed: the baseline's own output was previously never
checked for correctness, only timed. Neither is a bug in this project's
Triton kernels or in `parallel_suffix_scan`/`parallel_prefix_scan`
themselves (both verified correct, eager and compiled, in isolation).

### Bug 1 — allocating `torch.zeros_like(...)` inside a `torch.compile`'d function corrupts results

`torch.compile(vectorized_gae)` (a thin wrapper that did
`truncateds = torch.zeros_like(terminateds)` *inside* the function body
before calling `vectorized_gae_with_truncations`) gave up to 17% of output
elements wrong (max abs diff ~8, not a rounding-level discrepancy) at some
shapes — even on a single, freshly-compiled shape with no prior compilation
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
kind of allocation but were not observed to trigger it — inconsistent
enough across functions that the fix was applied everywhere as a precaution,
not just where it was caught.

**Fix:** never `torch.compile` a wrapper that allocates tensors internally.
Compile `vectorized_X_with_truncations` directly; build `truncateds`/
`bootstrap_values` (or any other constant/zero tensor) in eager code at the
call site, always as arguments, never inside the traced function.

### Bug 2 — reusing one `torch.compile`'d object across shapes with different padding behavior corrupts results

Separately: reusing a single `torch.compile(...)`-wrapped object across
*different* input shapes — the normal pattern in a benchmark sweep that
iterates many `(num_envs, seq_len)` configs — silently gives wrong output
for some shape transitions, even after fixing Bug 1. Specifically observed
between a shape that needs `parallel_suffix_scan`'s padding path
(`seq_len=80`, padded to `T_pad=128`) and one that doesn't (`seq_len=512`,
already a power of 2) when compiled by the *same* object in sequence. A
freshly-compiled object used for only one shape is always correct; the same
"fresh" object created *inside a loop that has already compiled other
shapes in the same process* can still be wrong — so this isn't simply
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
these guards. Neither is specific to the scan algorithms here — worth
checking for elsewhere if `torch.compile` baselines are added to future
benchmarks.

### Bug 2 addendum — confirmed independent of `reset()` usage; not confined to padding transitions; mechanism still unproven

A later session re-examined Bug 2 because its original repro (above) still
used `torch._dynamo.reset()` calls elsewhere in the same process, leaving
open the possibility that what was observed was an artifact of resetting
itself (e.g. some reset-accumulation side effect) rather than a bug that
exists independent of resets. This was checked directly.

**What was actually observed (fact, reproduced this session):**
- Compiling two different functions (`vectorized_gae_with_truncations`,
  `vectorized_vtrace_with_truncations`) across the full non-padding `CONFIGS`
  grid, then reusing one of those compiled objects into `PRODUCTION_CONFIGS`
  (crossing into the `seq_len=80` padding shape) — **with zero
  `torch._dynamo.reset()` calls anywhere in the process** — reproduces the
  same wrong-output failure at the padding transition that Bug 2 originally
  reported. This rules out "artifact of reset() usage" as the explanation:
  the bug is present with no resets at all.
- A **second, distinct** instance was found in the same zero-reset condition:
  `vectorized_discounted_returns_with_truncations`, compiled by one object
  first at `(num_envs=64, seq_len=512)` then reused at `(128, 1024)` —
  **neither shape needs padding** (both already powers of 2) — gives ~22.5%
  of elements wrong (not NaN/inf) at the second shape. This is NOT the
  padding-specific mechanism Bug 2 originally described; it is the same
  general *symptom* (cross-shape reuse of one compiled object → silently
  wrong output) triggered by a different, non-padding shape pair, specific
  to this one baseline function. Reproduces even in a subprocess that only
  ever compiles this single object (no other algorithm's compiled objects
  involved) — i.e. it is not about *total* accumulated compiles in the
  process either; two compiles of one object is already enough. A brand-new
  object's first-ever compile at either shape alone, or the same object used
  at only one shape, is always correct.

**What is interpretation, not proof:** calling this "process-state
corruption in Dynamo/Inductor's caching or guard mechanism" is the working
description, matching the observed shape (reuse-across-shapes triggers it;
a single fresh compile never does). Neither this nor the original Bug 2 was
root-caused to a specific Inductor buffer-reuse or guard bug via
IR/generated-kernel inspection — that would need reading the Inductor-
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
docstring) — a native forward-scan rewrite with no `torch.flip` compiles
clean with no crash across the full shape sweep. That crash is a different
failure mode from this section's bug (a hard CUDA exception vs. a silent
wrong-but-finite numeric result caught by `assert_correctness`) and is
believed fully resolved by the rewrite. Subprocess-per-algorithm isolation
is kept regardless — it remains a correctness-neutral, low-cost hedge, and
this section's bug demonstrates cross-shape reuse corruption can still
occur *within* a single algorithm's own subprocess, which is exactly why
the per-shape `reset()` fix above is still necessary even with isolation in
place.
