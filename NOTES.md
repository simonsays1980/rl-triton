# Implementation Notes

## Why there is no chunked fused V-Trace kernel

`compute_vtrace_fused` eliminates all intermediate tensor allocations and kernel
launches by doing everything — IS ratio computation, backward scan, target
construction, and advantage computation — inside a single Triton kernel per
environment row.  It is the fast path for `seq_len <= 131072`.

For longer sequences, `compute_vtrace_triton` falls back to a two-stage path:
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
RLlib do when concatenating episodes into fixed-length trajectory windows.

### Eligibility traces (forward scan)

`compute_eligibility_traces` runs a forward scan.  The same `done` flag zeroes
`v[t] = gamma * lambda * (1 - done[t])`, resetting the trace at episode
boundaries in exactly the same way.

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
