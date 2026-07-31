# PPO end-to-end measurement -- 2026-07-31

ONE-OFF measurement for the paper's evaluation section, not a recurring benchmark table.
GPU: NVIDIA H100 80GB HBM3 · torch 2.4.1+cu124
Arms: Triton GAE, baseline (torch.compile vectorized) GAE, baseline (torch.compile sequential loop) GAE.
num_envs=4096, seq_len=128, 4 epochs x 4 minibatches, 30 interleaved iterations (10 warmup), round-robin over all arm orderings.

### hidden=(256, 256), net mode=eager

| stage | Triton GAE | baseline (torch.compile vectorized) GAE | baseline (torch.compile sequential loop) GAE |
|---|---|---|---|
| forward | 12.4881 ms (15.0%) | 12.4435 ms (14.9%) | 12.4374 ms (14.8%) |
| gae | 0.2301 ms (0.3%) | 0.6136 ms (0.7%) | 1.4521 ms (1.7%) |
| loss | 11.7655 ms (14.1%) | 11.9412 ms (14.3%) | 11.8545 ms (14.1%) |
| backward | 47.1454 ms (56.7%) | 46.8590 ms (56.1%) | 46.7039 ms (55.6%) |
| optimizer | 11.5716 ms (13.9%) | 11.7358 ms (14.0%) | 11.6078 ms (13.8%) |
| **total** | **83.2007 ms** | **83.5930 ms** | **84.0557 ms** |

GAE share of step: 0.28% (triton), 0.73% (vectorized), 1.73% (loop). End-to-end speedup vs. triton arm: 1.005x (vectorized / triton), 1.010x (loop / triton).

### hidden=(256, 256), net mode=torch.compile

| stage | Triton GAE | baseline (torch.compile vectorized) GAE | baseline (torch.compile sequential loop) GAE |
|---|---|---|---|
| forward | 19.6680 ms (20.6%) | 19.5821 ms (20.3%) | 19.6113 ms (20.2%) |
| gae | 0.2785 ms (0.3%) | 0.6323 ms (0.7%) | 1.6201 ms (1.7%) |
| loss | 14.4447 ms (15.1%) | 14.4341 ms (15.0%) | 14.3608 ms (14.8%) |
| backward | 47.1792 ms (49.3%) | 47.7503 ms (49.5%) | 47.5125 ms (48.9%) |
| optimizer | 14.1072 ms (14.7%) | 14.0478 ms (14.6%) | 14.0192 ms (14.4%) |
| **total** | **95.6776 ms** | **96.4466 ms** | **97.1238 ms** |

GAE share of step: 0.29% (triton), 0.66% (vectorized), 1.67% (loop). End-to-end speedup vs. triton arm: 1.008x (vectorized / triton), 1.015x (loop / triton).

### hidden=(1024, 1024), net mode=eager

| stage | Triton GAE | baseline (torch.compile vectorized) GAE | baseline (torch.compile sequential loop) GAE |
|---|---|---|---|
| forward | 41.3315 ms (27.6%) | 41.3507 ms (27.5%) | 41.3534 ms (27.4%) |
| gae | 0.0935 ms (0.1%) | 0.2358 ms (0.2%) | 0.5326 ms (0.4%) |
| loss | 4.7650 ms (3.2%) | 4.8157 ms (3.2%) | 4.8013 ms (3.2%) |
| backward | 99.1410 ms (66.1%) | 99.2792 ms (66.0%) | 99.4392 ms (66.0%) |
| optimizer | 4.5880 ms (3.1%) | 4.6575 ms (3.1%) | 4.6118 ms (3.1%) |
| **total** | **149.9190 ms** | **150.3390 ms** | **150.7383 ms** |

GAE share of step: 0.06% (triton), 0.16% (vectorized), 0.35% (loop). End-to-end speedup vs. triton arm: 1.003x (vectorized / triton), 1.005x (loop / triton).

### hidden=(1024, 1024), net mode=torch.compile

| stage | Triton GAE | baseline (torch.compile vectorized) GAE | baseline (torch.compile sequential loop) GAE |
|---|---|---|---|
| forward | 43.0024 ms (28.6%) | 43.0015 ms (28.6%) | 43.0049 ms (28.5%) |
| gae | 0.0908 ms (0.1%) | 0.2162 ms (0.1%) | 0.5454 ms (0.4%) |
| loss | 5.0061 ms (3.3%) | 4.9947 ms (3.3%) | 5.0056 ms (3.3%) |
| backward | 97.4316 ms (64.8%) | 97.1658 ms (64.7%) | 97.2651 ms (64.6%) |
| optimizer | 4.7828 ms (3.2%) | 4.7912 ms (3.2%) | 4.8098 ms (3.2%) |
| **total** | **150.3138 ms** | **150.1693 ms** | **150.6308 ms** |

GAE share of step: 0.06% (triton), 0.14% (vectorized), 0.36% (loop). End-to-end speedup vs. triton arm: 0.999x (vectorized / triton), 1.002x (loop / triton).
