"""Task 3 driver: one subprocess per shape; resumable.

    python benchmarks/ablation_serial/packed_run.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "packed_raw.jsonl"
SHAPES = [(e, t) for e in (4096, 8192, 16384, 32768) for t in (8, 16, 32, 64, 128)] + [(512, 512), (512, 4096)]


def main():
    seen = set()
    if RAW.exists():
        seen = {(r["num_envs"], r["seq_len"]) for r in map(json.loads, RAW.read_text().splitlines())
                if r["kind"] == "result"}
    t0, fails = time.time(), []
    for e, t in SHAPES:
        if (e, t) in seen:
            continue
        s = time.time()
        p = subprocess.run([sys.executable, str(HERE / "packed_worker.py"), "--num-envs", str(e),
                            "--seq-len", str(t), "--out", str(RAW)], capture_output=True, text=True)
        fl = [l for l in p.stdout.splitlines() if "failure" in l or "skipped" in l]
        print(f"[{e}x{t}] packed: {time.time() - s:.0f}s (total {(time.time() - t0) / 60:.1f} min)"
              + (f"\n  " + "\n  ".join(fl) if fl else "")
              + ("" if p.returncode == 0 else f"  FAILED rc={p.returncode}\n{p.stderr[-1500:]}"), flush=True)
        if p.returncode:
            fails.append((e, t))
    print(f"done; subprocess failures: {fails or 'none'}", flush=True)


if __name__ == "__main__":
    main()
