"""Task A driver: one subprocess per shape; resumable.

    python benchmarks/ablation_serial/warp_run.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / (sys.argv[1] if len(sys.argv) > 1 else "warp_raw.jsonl")  # repeat run: warp_raw_repeat.jsonl
SHAPES = [(e, t) for t in (16384, 32768, 65536) for e in (16, 64, 256)]

seen = set()
if RAW.exists():
    seen = {(r["num_envs"], r["seq_len"]) for r in map(json.loads, RAW.read_text().splitlines())}
t0 = time.time()
for e, t in SHAPES:
    if (e, t) in seen:
        continue
    s = time.time()
    p = subprocess.run([sys.executable, str(HERE / "warp_worker.py"), "--num-envs", str(e),
                        "--seq-len", str(t), "--out", str(RAW)], capture_output=True, text=True)
    print(f"[{e}x{t}] {time.time() - s:.0f}s (total {(time.time() - t0) / 60:.1f} min)"
          + ("" if p.returncode == 0 else f"  FAILED rc={p.returncode}\n{p.stderr[-1500:]}"), flush=True)
    for line in p.stdout.splitlines():
        r = json.loads(line)
        print(f"    nw={r['num_warps']:<2} valid={r['validated']} "
              + (f"full {r['fullcall_ms']:.4f} dev {r['device_ms']:.4f}" if r.get("fullcall_ms") else r.get("skip_reason", "")),
              flush=True)
print("done", flush=True)
