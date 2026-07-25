<!-- README_BENCH_DRAFT: NOT auto-applied. Prepared 2026-07-25 on NVIDIA H100 80GB HBM3. -->

All rows at num_envs=4096, seq_len=128 (PufferLib/Gigaflow-default rollout size) for a genuine apples-to-apples comparison across algorithms.

| algorithm | speedup vs torch.compile (full-call) |
|:---|:---:|
| GAE | 2.2× |
| V-Trace | 2.3× |
| Retrace | 1.8× |
| lambda-returns | 1.9× |

See [benchmarks.md](benchmarks.md) for the full sweep, methodology, and truncation-path results.