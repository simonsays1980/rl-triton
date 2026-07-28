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

**Measured (H200; same CUDA-events full-call + device-only methodology used
throughout this repo's benchmarks — see `benchmarks.md`'s Methodology
paragraph): adopting the `lambda_returns_fused_kernel` pre-shift strategy
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
clean with no crash across the full shape sweep. `dmesg` and the driver's
ECC error counters were checked and were clean throughout — this was a
software (Inductor-generated-kernel) fault, not a hardware fault. That crash is a different
failure mode from this section's bug (a hard CUDA exception vs. a silent
wrong-but-finite numeric result caught by `assert_correctness`) and is
believed fully resolved by the rewrite. Subprocess-per-algorithm isolation
is kept regardless — it remains a correctness-neutral, low-cost hedge, and
this section's bug demonstrates cross-shape reuse corruption can still
occur *within* a single algorithm's own subprocess, which is exactly why
the per-shape `reset()` fix above is still necessary even with isolation in
place.

---

## The log-space baseline underflow — why `compile(vec)` is a doubling scan, not a cumsum

Five of the six "vectorized" PyTorch comparison baselines (GAE, V-trace,
λ-returns, eligibility-traces, Retrace) originally computed their
decay-weighted scan via `exp(cumsum(log(decay.clamp(min=1e-38))))`. The
`clamp(min=1e-38)` exists only so `log(0)` at an episode-terminated step
(`decay=0` there) doesn't produce `-inf` — but it is exactly what breaks the
formula: each termination contributes `log(1e-38) ≈ -87.5` to the running
suffix sum, and float32 underflows (flushes to zero, then `exp` propagates
inf/nan downstream) anything below `≈ -103.3`. At this project's ~5%/step
termination rate, 2-3 terminations anywhere in a window — expected within
the first couple dozen steps of *any* row, independent of overall
`seq_len` — is enough. This is not an extreme-scale-only failure: it hit
90-99% of output elements at every size this project's release table
actually uses, from 64×512 up to the production regime at 4096×128, not
only the long-`seq_len` regime `benchmarks/chunked_scan.md` originally
probed.

**Fix:** `compile(vec)` no longer uses the log-space formula anywhere.
GAE/V-trace/λ-returns/discounted-returns became thin wrappers around their
already-correct `*_with_truncations` siblings (`truncateds=0`,
`bootstrap=0`), which were already built on `parallel_suffix_scan` — a
linear-space `log2(T)`-doubling associative scan with no `log`/`exp`
anywhere. Eligibility-traces and Retrace were rewritten directly on the
same primitive (`parallel_prefix_scan` / `parallel_suffix_scan`
respectively). The old `compile(assoc)` column in `benchmarks.md` is gone
because there is no longer a separate specialized no-truncation baseline to
compare it against — the with-truncations implementation *is* the baseline
now, called with zero truncations.

**Caveat: this is the strongest *correct* baseline built, not necessarily
the strongest possible one.** The doubling-scan baseline pays 6-12 kernel
launches per call (growing with `seq_len`, one per doubling step), against
1-2 launches for the Triton kernel it's compared to. Isolated primitive
timing at a fixed shape found the (underflow-buggy, only safe without
terminations) log-space cumsum takes ~37µs across 2 launches, versus the
doubling scan's ~148µs across 6 launches — a 4x gap attributable to launch
count alone, not to the underlying arithmetic. A numerically-stable
non-log-space cumsum formulation (e.g. a segmented/blocked scan that
periodically renormalizes instead of ever taking one running log-sum) could
plausibly avoid the underflow without paying the doubling scan's launch
count, and would be faster. This means published ratios like "N× vs.
torch.compile" describe the doubling-scan baseline specifically, not an
upper bound on what a correct PyTorch baseline could achieve — a materially
different claim than the older (numerically wrong) few-launch-cumsum
numbers this project published before the underflow was found. Add-only, not
yet built: nobody has implemented the faster correct baseline to check how
much of the gap it would close.

---

## Per-GPU floor calibration is a correctness requirement, not an optimisation

`tests/bench_safeguard.py`'s CI-facing performance gate times the Triton
kernel against `torch.compile(vec)` at one small shape (128×1024) and
asserts a minimum speedup. At that shape, both sides' *device* kernel time
is only a few microseconds — dwarfed by ~25-30µs of fixed CUDA
dispatch/sync overhead that both sides pay identically regardless of which
kernel is actually faster. A slower GPU's kernel work dominates that fixed
cost, so its measured wall-clock ratio reflects real kernel quality; a
faster GPU's kernel work shrinks by an order of magnitude while the ~25-30µs
floor stays fixed, compressing the wall-clock ratio toward 1x independent of
kernel quality. Concretely: eligibility-traces' kernel is a genuine,
reproducible 2.7x faster in raw device time on H100/H200-class hardware,
yet the reported *wall-clock* speedup compresses from ~11x on an older,
slower card to under 2x on the faster one — launch-overhead amortization,
not a kernel regression.

**Consequence:** a floor calibrated on one GPU does not transfer to a
different one, faster or slower. `_FLOOR_TABLE` in `bench_safeguard.py` is
keyed by device-name substring (see that file's own module comment for the
mechanism and the currently-calibrated cards); an unrecognized GPU must
`pytest.skip()` loudly rather than silently reuse another card's floor —
reusing a floor across GPU families is exactly the mistake that let this
gate pass vacuously in the past (calibrated on the wrong card entirely, see
`bench_safeguard.py`'s own history comment for that story).

---

## Retrace's register-spill ceiling and its reroute (see `retrace.py` for the authoritative account)

`compute_retrace` dispatches on `_TRITON_SEQ_LEN_CEILING = 2048`:
below it, the fused Triton kernel; above it, a reroute through the generic
`_run_scan` associative-scan path. `src/rl_triton/ops/retrace.py`'s own
comment block above that constant is the authoritative source for the
mechanism (register pressure from re-reading the 3D action-probability
tensor a second time for `c[t+1]`, pinning `ptxas` at 128 regs/thread and
25% occupancy from `seq_len=4096` onward) and for the reroute's own
performance (measured 0.38-0.47x vs. `torch.compile` — **a smaller loss,
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
| realistic (1024,1024) | 0.07% – 0.6% | **~1.00x — no measurable difference** |

Full per-stage breakdown (4 configs, all four pipeline stages) is in
`docs/benchmark-history/ppo-e2e-measurement-2026-07-25.md`; this section is
the condensed takeaway, not a replacement for it. The larger hidden size is
the realistic case in both measurements — backward
pass dominates the step (60-66% of total time) at that size, and GAE's
isolated ~1.9x device-level kernel speedup becomes invisible in full
training throughput. **This is a bound the paper's evaluation section must
respect: the isolated-microbenchmark speedup claim holds, an end-to-end
training-speed claim from this kernel alone does not**, at realistic network
sizes. No win/lose framing beyond that is warranted — GAE was never the
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
(pre-underflow-fix) log-space formula instead of an actual reset — a
`done=1` step contributed `0` to the log-sum rather than terminating the
discount chain, so the accumulator never reset. Result: finite but wrong
output (max abs error ~43 vs. the sequential reference on a 64-step test) —
a different failure mode from the inf/nan underflow affecting the other
five baselines, since this one never produced a value extreme enough to
underflow.

**`vectorized_episodic_prefix_sum`:** the old formula
(`running = cumsum(inputs); boundary = running*dones; offset =
cumsum(boundary)-boundary; result = running-offset`) resets one step later
than this codebase's convention (`done[t]=1` means the reset applies *at*
`t` itself, not at `t+1`). Wrong at every shape ever benchmarked with it —
44-95.8% of elements, including the smallest configs — because the
baseline's own correctness had never been checked, only timed. **Fix:**
replaced with `parallel_prefix_scan(inputs, 1.0 - dones)`, which encodes the
reset convention directly rather than reconstructing it after the fact.
Any previously-published prefix-sum speedup number predates this fix and
should not be trusted; the current `benchmarks.md` table reflects the
corrected baseline.

---

## rl-triton vs. PufferLib: a real design tradeoff at short-horizon, high-env-count workloads — not a bug, not fixed by the warps-table fix

At very short horizons and very high environment counts (`num_envs >= 4096`,
`seq_len <= 16`; the crossover point shifts to `seq_len ~= 128` by
`num_envs = 32768`), PufferLib's hand-written sequential-per-row CUDA kernel
measures faster than rl-triton's one-program-per-environment Triton kernel,
on both device time and wall-clock. Mechanism: PufferLib's per-row O(T) cost
is nearly `num_envs`-insensitive (thread count scales directly with
`num_envs`, packing the GPU efficiently even at tiny per-row problem sizes),
while rl-triton's one-program-per-env grid design pays more grid waves per
SM as `num_envs` grows (up to ~62 waves at `num_envs=32768` vs. ~8 at 4096)
— narrowed by the small-`BLOCK_SIZE` warps-table fix (see each op's `_WARPS`
comment) but not eliminated by it, since that fix addresses per-program warp
over-provisioning, not grid-wave count.

**Consequence for anyone choosing between these kernels:** rl-triton's
design is right for the long-horizon, moderate-env-count on-policy rollout
regime this project targets (see the production-regime table in
`benchmarks.md`), and measurably wrong for the shortest-horizon,
highest-env-count corner of massively-parallel simulation (Isaac Gym/Isaac
Lab style, tens of thousands of envs at single-digit-to-low-double-digit
horizons). This is a genuine per-workload tradeoff, not a defect to fix —
but it means "rl-triton is faster than PufferLib" is not a universal claim;
see `benchmarks/pufferlib.md` for where each kernel wins.

---

## Eligibility-traces' apparent device-time ceiling is a small-grid occupancy ramp, not an architectural limit

An earlier hypothesis attributed eligibility-traces' device-time speedup
plateauing around ~2.4-2.5x to L2-cache residency (the working set falling
out of L2 above some size). That hypothesis was tested directly and
refuted: no configuration ever exceeds HBM peak bandwidth, and there is no
discontinuity at the size where the working set crosses the L2 capacity
boundary. The actual mechanism is a small-grid occupancy ramp — one program
per environment means small `num_envs` doesn't generate enough concurrent
thread blocks to saturate all SMs and hide memory latency. Extending the
swept `num_envs` range well past where earlier measurements stopped showed
achieved bandwidth still climbing (roughly 67% -> 78% -> 84% of peak) with
no plateau in sight — the "ceiling" earlier measurements read as
architectural was an artifact of not having swept `num_envs` far enough.
Addressable in principle (persistent-kernel or multi-env-per-program
launch), not attempted — out of scope until a real workload needs
eligibility-traces at very small `num_envs`.

---

## Truncation-path speedup grows with scale — durable, unlike the plain path's high-end compression

For the four algorithms with a truncation-aware path (GAE, V-trace,
λ-returns, discounted-returns), the Triton-vs-baseline speedup on that path
*grows* with `num_envs`/`seq_len` once the baseline is the corrected
doubling-scan implementation (e.g. GAE: 2.76x -> 9.88x from 512 to 8192
envs at `seq_len=2048`) — the opposite of the plain path's behavior on
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
total time on both sides — the truncation path's numbers are less sensitive
to the same GPU-speed effect that makes the plain-path floor
hardware-specific.
