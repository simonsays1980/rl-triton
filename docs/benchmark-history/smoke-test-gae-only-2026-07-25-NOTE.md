# GAE-only smoke test — 2026-07-25 — NVIDIA H100 80GB HBM3

Run via `python tests/bench_release.py --no-update --algos gae --gpu "NVIDIA H100 80GB HBM3"`,
per the unattended-safety instruction to smoke-test GAE alone (main grid + production regime +
correctness + monotonicity) before running the other six algorithms. Full console output in
`smoke-test-gae-only-2026-07-25.log` in this directory.

## Correctness gate: PASSED

`assert_correctness` (atol=rtol=1e-4 vs. the sequential reference) held at every config reached —
main grid, production regime (seq_len [80,128] x num_envs [4096..38400]), and the boundary marker
(num_envs=16384, seq_len=16). It raises immediately on failure, so no exception means every
correctness check up to the point of gate failure held.

## Monotonicity gate: FAILED (6 violations, main grid only; production regime passed)

```
seq_len=1024,num_envs=128   -> seq_len=1024,num_envs=256:   0.0638ms -> 0.0624ms (-2.2%)
seq_len=128,num_envs=512    -> seq_len=128,num_envs=4096:    0.0638ms -> 0.0612ms (-4.2%)
seq_len=128,num_envs=4096   -> seq_len=128,num_envs=16384:   0.0612ms -> 0.0411ms (-32.9%)
num_envs=512,seq_len=2048   -> num_envs=512,seq_len=4096:    0.1577ms -> 0.1510ms (-4.2%)   [truncation path]
seq_len=2048,num_envs=512   -> seq_len=2048,num_envs=4096:   0.1577ms -> 0.1486ms (-5.7%)   [truncation path]
```//see full log for the complete list

All violations are in the sub-0.2ms absolute-time regime (tens of microseconds apart), which is
the same clock-state-jitter range this repository already documents as a known measurement-floor
effect on this hardware (`bench_utils.py`'s min-of-5-medians rationale; near-identical monotonicity
dips already diagnosed as noise, not a structural regression, in
`benchmarks/benchmark_gae_vs_pufferlib.py`'s `report_monotonicity()` (moved from tests/ since) —
H100 idles at 210MHz and boosts
to 3105MHz, and a single clock-state transition can swing a measurement well past a 2% band at
these tiny absolute times). This assessment is offered as context, not as a unilateral override —
per the autonomy boundary, a smoke-test gate failure is a STOP-and-report item, so the remaining
six algorithms were NOT run pending human review of this result.

## Status

- Harness code changes (device-time column, correctness gate, monotonicity gate, production-regime
  table, amortized variant) are implemented and committed on `bench/h100-release`.
- The other six algorithms (V-Trace, Retrace, λ-returns, discounted returns, eligibility traces,
  episodic prefix sum) have NOT been run yet, pending a decision on how to proceed given the above.
