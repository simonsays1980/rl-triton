# PPO end-to-end measurement -- 2026-08-05

ONE-OFF measurement for the paper's evaluation section, not a recurring benchmark table.
GPU: NVIDIA H100 80GB HBM3 · torch 2.4.1+cu124
num_envs=4096, seq_len=128, 4 epochs x 4 minibatches, 30 interleaved-A/B iterations (10 warmup).

### hidden=(256, 256), net mode=eager

| stage | Triton GAE | baseline (torch.compile vectorized) GAE |
|---|---|---|
| forward | 11.8529 ms (15.9%) | 11.7325 ms (16.0%) |
| gae | 0.1995 ms (0.3%) | 0.3871 ms (0.5%) |
| loss | 9.2840 ms (12.4%) | 8.5889 ms (11.7%) |
| backward | 44.1703 ms (59.1%) | 44.3124 ms (60.4%) |
| optimizer | 9.1830 ms (12.3%) | 8.3907 ms (11.4%) |
| **total** | **74.6898 ms** | **73.4116 ms** |

GAE share of step: 0.27% (triton arm) / 0.53% (baseline arm). End-to-end speedup: 0.983x.

### hidden=(256, 256), net mode=torch.compile

| stage | Triton GAE | baseline (torch.compile vectorized) GAE |
|---|---|---|
| forward | 18.2606 ms (19.6%) | 17.8609 ms (19.4%) |
| gae | 0.2551 ms (0.3%) | 0.4438 ms (0.5%) |
| loss | 13.4993 ms (14.5%) | 12.9917 ms (14.1%) |
| backward | 48.0164 ms (51.7%) | 48.1171 ms (52.2%) |
| optimizer | 12.9299 ms (13.9%) | 12.7578 ms (13.8%) |
| **total** | **92.9613 ms** | **92.1713 ms** |

GAE share of step: 0.27% (triton arm) / 0.48% (baseline arm). End-to-end speedup: 0.992x.

### hidden=(1024, 1024), net mode=eager

| stage | Triton GAE | baseline (torch.compile vectorized) GAE |
|---|---|---|
| forward | 42.1032 ms (27.6%) | 42.0069 ms (27.7%) |
| gae | 0.0911 ms (0.1%) | 0.2015 ms (0.1%) |
| loss | 4.9734 ms (3.3%) | 4.8512 ms (3.2%) |
| backward | 100.5186 ms (66.0%) | 100.0814 ms (65.9%) |
| optimizer | 4.7194 ms (3.1%) | 4.6160 ms (3.0%) |
| **total** | **152.4058 ms** | **151.7571 ms** |

GAE share of step: 0.06% (triton arm) / 0.13% (baseline arm). End-to-end speedup: 0.996x.

### hidden=(1024, 1024), net mode=torch.compile

| stage | Triton GAE | baseline (torch.compile vectorized) GAE |
|---|---|---|
| forward | 43.7808 ms (28.8%) | 43.6642 ms (28.8%) |
| gae | 0.0876 ms (0.1%) | 0.1629 ms (0.1%) |
| loss | 4.9792 ms (3.3%) | 4.9680 ms (3.3%) |
| backward | 98.5013 ms (64.7%) | 98.2436 ms (64.7%) |
| optimizer | 4.8564 ms (3.2%) | 4.7963 ms (3.2%) |
| **total** | **152.2053 ms** | **151.8350 ms** |

GAE share of step: 0.06% (triton arm) / 0.11% (baseline arm). End-to-end speedup: 0.998x.
