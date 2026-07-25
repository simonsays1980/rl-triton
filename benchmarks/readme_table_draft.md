<!-- README_BENCH_DRAFT: NOT auto-applied. Prepared 2026-07-25 on NVIDIA H100 80GB HBM3. -->

| algorithm | num_envs | seq_len | speedup vs torch.compile (full-call) |
|:---|:---:|:---:|:---:|
| GAE | 4096 | 128 | 2.2× |
| V-Trace | 16384 | 128 | 4.5× |
| Retrace | 4096 | 80 | 1.9× |
| lambda-returns | 38400 | 128 | 6.6× |

See [benchmarks.md](benchmarks.md) for the full sweep, methodology, and truncation-path results.