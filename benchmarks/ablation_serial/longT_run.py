"""Task 2 driver: one subprocess per (shape, kernel); resumable.

    python benchmarks/ablation_serial/longT_run.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "longT_raw.jsonl"
SHAPES = [(e, t) for e in (16, 64, 256) for t in (8192, 16384, 32768, 65536)]
JOBS = ([(s, k) for k in ("tree", "scan", "serial_env", "serial_time") for s in SHAPES]
        + [(s, "loop") for s in SHAPES if s[1] <= 16384])


def done():
    seen = set()
    if RAW.exists():
        for line in RAW.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "result":
                k = r["kernel"].removesuffix("_naive")
                k = "serial_time" if k in ("transpose_inputs", "serial_time+transpose") else k
                seen.add((r["num_envs"], r["seq_len"], k))
    return seen


def main():
    seen, t0, fails = done(), time.time(), []
    for (e, t), k in JOBS:
        if (e, t, k) in seen:
            continue
        s = time.time()
        p = subprocess.run([sys.executable, str(HERE / "longT_worker.py"), "--kernel", k,
                            "--num-envs", str(e), "--seq-len", str(t), "--out", str(RAW)],
                           capture_output=True, text=True)
        print(f"[{e}x{t}] {k}: {time.time() - s:.0f}s (total {(time.time() - t0) / 60:.1f} min)"
              + ("" if p.returncode == 0 else f"  FAILED rc={p.returncode}\n{p.stderr[-1500:]}"), flush=True)
        if p.returncode:
            fails.append((e, t, k))
    print(f"done; failures: {fails or 'none'}", flush=True)


if __name__ == "__main__":
    main()
