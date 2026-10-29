"""Driver: one subprocess per (shape, kernel), as tests/bench_release.py does
(torch.compile/Inductor process-state isolation). Resumable: (shape, kernel)
pairs already present in raw_results.jsonl are skipped.

    python benchmarks/ablation_serial/run_ablation.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_results.jsonl"

GRID_A = [(64, 512), (128, 1024), (256, 1024), (512, 128), (512, 512), (512, 2048),
          (512, 4096), (4096, 128), (4096, 512), (4096, 2048), (16384, 128), (16384, 512)]
GRID_B = [(e, t) for e in (4096, 8192, 16384, 32768) for t in (8, 16, 32, 64, 128)]
SHAPES = list(dict.fromkeys(GRID_A + GRID_B))
KERNELS = ["tree", "tree_lib", "scan", "serial_env", "serial_time"]


def done():
    seen = set()
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] == "result":
                k = row["kernel"].removesuffix("_naive")
                k = "serial_time" if k in ("transpose_inputs", "serial_time+transpose") else k
                seen.add((row["num_envs"], row["seq_len"], k))
    return seen


def main():
    seen = done()
    failures = []
    for num_envs, seq_len in SHAPES:
        for kernel in KERNELS:
            if (num_envs, seq_len, kernel) in seen:
                continue
            print(f"[{num_envs}x{seq_len}] {kernel}", flush=True)
            p = subprocess.run(
                [sys.executable, str(HERE / "bench_worker.py"), "--kernel", kernel,
                 "--num-envs", str(num_envs), "--seq-len", str(seq_len), "--out", str(RAW)],
                capture_output=True, text=True)
            if p.returncode != 0:
                failures.append((num_envs, seq_len, kernel))
                print(f"  SUBPROCESS FAILED rc={p.returncode}\n{p.stderr[-2000:]}", flush=True)
    print(f"done; subprocess failures: {failures or 'none'}", flush=True)


if __name__ == "__main__":
    main()
