<!-- README_BENCH_DRAFT: NOT auto-applied. Prepared 2026-07-30 on NVIDIA RTX 2000 Ada Generation. -->

All rows at num_envs=4096, seq_len=128 (PufferLib/Gigaflow-default rollout size) for a genuine apples-to-apples comparison across algorithms.

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 5.1× |
| V-Trace | 5.7× |
| Retrace | 1.6× |
| lambda-returns | 4.6× |

With truncations (terminations + time-limit truncations + bootstrap values), same config — Retrace's kernel supports truncations (terminated/truncated are both mandatory, distinct arguments) but has no vectorized-baseline benchmark for that path yet, so it's omitted here rather than compared against an invalid baseline:

| algorithm | speedup vs torch.compile, with truncations (full-call) |
|:---|:---:|
| GAE | 2.5× |
| V-Trace | 4.8× |
| lambda-returns | 3.8× |

See [benchmarks.md](benchmarks.md) for the full sweep, methodology, and truncation-path results.