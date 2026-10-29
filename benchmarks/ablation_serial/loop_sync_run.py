"""Task B driver: one subprocess per job, rows appended to loop_sync_raw.jsonl."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALGOS = ("gae", "vtrace", "retrace", "lambda_returns", "discounted_returns",
         "eligibility_traces", "prefix_sum")
JOBS = [(a, 4096, 128) for a in ALGOS] + [("gae", 64, 512), ("gae", 512, 4096)]

out = HERE / "loop_sync_raw.jsonl"
out.write_text("")
for algo, E, T in JOBS:
    t0 = time.time()
    p = subprocess.run([sys.executable, str(HERE / "loop_sync_worker.py"), algo, str(E), str(T)],
                       capture_output=True, text=True)
    row = next((json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")), None)
    if row is None:
        row = {"algo": algo, "num_envs": E, "seq_len": T, "validated": False,
               "error": (p.stderr or p.stdout)[-500:]}
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{algo} {E}x{T}] {time.time()-t0:.0f}s {row.get('loop_sync_ms')} "
          f"(bench_cpu {row.get('loop_bench_cpu_same_session_ms')}) {row.get('error','')}", flush=True)
