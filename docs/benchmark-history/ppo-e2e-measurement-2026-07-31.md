# PPO end-to-end measurement -- 2026-07-31

ONE-OFF measurement for the paper's evaluation section, not a recurring benchmark table.
GPU: NVIDIA H100 80GB HBM3 · torch 2.4.1+cu124
num_envs=4096, seq_len=128, 4 epochs x 4 minibatches, 30 interleaved-A/B iterations (10 warmup).

### hidden=(256, 256), net mode=eager

| stage | Triton GAE | baseline (torch.compile vectorized) GAE |
|---|---|---|
| forward | 12.5889 ms (15.3%) | 12.5018 ms (14.9%) |
| gae | 0.2698 ms (0.3%) | 0.7431 ms (0.9%) |
| loss | 11.5932 ms (14.1%) | 11.7037 ms (14.0%) |
| backward | 46.0470 ms (56.1%) | 46.7309 ms (55.8%) |
| optimizer | 11.6407 ms (14.2%) | 12.0861 ms (14.4%) |
| **total** | **82.1396 ms** | **83.7656 ms** |

GAE share of step: 0.33% (triton arm) / 0.89% (baseline arm). End-to-end speedup: 1.020x.

### hidden=(256, 256), net mode=torch.compile

| stage | Triton GAE | baseline (torch.compile vectorized) GAE |
|---|---|---|
| forward | 19.8104 ms (20.3%) | 19.7985 ms (20.0%) |
| gae | 0.2946 ms (0.3%) | 0.6322 ms (0.6%) |
| loss | 13.7309 ms (14.1%) | 13.9419 ms (14.1%) |
| backward | 49.8528 ms (51.1%) | 50.4707 ms (51.0%) |
| optimizer | 13.9169 ms (14.3%) | 14.0993 ms (14.2%) |
| **total** | **97.6057 ms** | **98.9426 ms** |

GAE share of step: 0.30% (triton arm) / 0.64% (baseline arm). End-to-end speedup: 1.014x.

### hidden=(1024, 1024), net mode=eager

| stage | Triton GAE | baseline (torch.compile vectorized) GAE |
|---|---|---|
| forward | 42.2184 ms (26.8%) | 42.1354 ms (26.9%) |
| gae | 0.1595 ms (0.1%) | 0.4397 ms (0.3%) |
| loss | 6.5433 ms (4.2%) | 6.5205 ms (4.2%) |
| backward | 101.3289 ms (64.4%) | 100.7590 ms (64.3%) |
| optimizer | 7.1068 ms (4.5%) | 6.8560 ms (4.4%) |
| **total** | **157.3569 ms** | **156.7105 ms** |

GAE share of step: 0.10% (triton arm) / 0.28% (baseline arm). End-to-end speedup: 0.996x.

### hidden=(1024, 1024), net mode=torch.compile

| stage | Triton GAE | baseline (torch.compile vectorized) GAE |
|---|---|---|
| forward | 45.0436 ms (28.3%) | 44.9526 ms (28.3%) |
| gae | 0.1559 ms (0.1%) | 0.3741 ms (0.2%) |
| loss | 6.8701 ms (4.3%) | 6.8641 ms (4.3%) |
| backward | 99.6012 ms (62.6%) | 99.1272 ms (62.4%) |
| optimizer | 7.5470 ms (4.7%) | 7.5461 ms (4.8%) |
| **total** | **159.2178 ms** | **158.8641 ms** |

GAE share of step: 0.10% (triton arm) / 0.24% (baseline arm). End-to-end speedup: 0.998x.
