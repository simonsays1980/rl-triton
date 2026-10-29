"""Task 4 diagnostic: separate the effect of synchronization from the effect of the
summary statistic (mean vs. min-of-5-medians) on the loop's (4096,128) time.
Per-call wall time with perf_counter, 20 warmup, 5 trials x 50 calls."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import torch  # noqa: E402
import bench_release as br  # noqa: E402

args = br._make_gae(4096, 128)
f = lambda: br._ref_gae(*args, gamma=0.99, lambda_=0.95)
out = {}
for sync in (False, True):
    for _ in range(20):
        f()
    torch.cuda.synchronize()
    trials = []
    for _ in range(5):
        ts = []
        for _ in range(50):
            if sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            f()
            if sync:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        trials.append(ts)
    torch.cuda.synchronize()
    allv = [x for t in trials for x in t]
    out["sync" if sync else "nosync"] = {
        "mean_ms": statistics.mean(allv),
        "min_of_trial_medians_ms": min(statistics.median(t) for t in trials),
        "max_ms": max(allv),
    }
print(json.dumps(out, indent=2))
(Path(__file__).resolve().parent / "task4_diag.json").write_text(json.dumps(out, indent=2))
