"""Task B report: loop_sync_raw.jsonl + benchmarks.md (H100 v0.1.3 per-algorithm tables)
-> loop_sync.csv (+ markdown table on stdout).

Tree time = "triton full-call (ms)" of the matching row in the per-algorithm table of the
"## v0.1.3 – 2026-08-08 – NVIDIA H100 80GB HBM3" section; recorded speedup = that row's
"vs loop" column (loop timed with _bench_cpu: wall clock, no per-call sync).
synced speedup = loop_sync_ms / tree full-call (recorded, 3-decimal precision).
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE.parents[1] / "benchmarks.md"
HEADERS = {"gae": "#### GAE (`compute_gae`)", "vtrace": "#### V-Trace (`compute_vtrace`)",
           "retrace": "#### Retrace(λ) (`compute_retrace`)",
           "lambda_returns": "#### λ-returns (`compute_lambda_returns`)",
           "discounted_returns": "#### Discounted returns (`compute_discounted_returns`)",
           "eligibility_traces": "#### Eligibility traces (`compute_eligibility_traces`)",
           "prefix_sum": "#### Episodic prefix sum (`compute_episodic_prefix_sum`)"}

lines = MD.read_text().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("## v0.1.3") and "H100" in l)
end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))


def recorded(algo, E, T):
    h = next(i for i in range(start, end) if lines[i].strip() == HEADERS[algo])
    i = h + 1
    while not lines[i].startswith("|"):
        i += 1
    cols = [c.strip() for c in lines[i].strip("|").split("|")]
    while lines[i].startswith("|"):
        cells = [c.strip() for c in lines[i].strip("|").split("|")]
        if cells[:2] == [str(E), str(T)]:
            row = dict(zip(cols, cells))
            return (float(row["triton full-call (ms)"]), float(row["loop gpu (ms)"]),
                    float(row["vs loop"].rstrip("x")), i + 1)
        i += 1
    raise KeyError((algo, E, T))


out = []
for r in (json.loads(l) for l in (HERE / "loop_sync_raw.jsonl").read_text().splitlines() if l.strip()):
    tree, loop_rec, sp_rec, ln = recorded(r["algo"], r["num_envs"], r["seq_len"])
    sp_sync = r["loop_sync_ms"] / tree
    out.append({"algo": r["algo"], "num_envs": r["num_envs"], "seq_len": r["seq_len"],
                "validated": r["validated"], "max_abs_err_loop_vs_kernel": r["max_abs_err_loop_vs_kernel"],
                "loop_sync_ms": r["loop_sync_ms"],
                "loop_bench_cpu_same_session_ms": r["loop_bench_cpu_same_session_ms"],
                "loop_recorded_ms": loop_rec, "tree_fullcall_recorded_ms": tree,
                "tree_source": f"benchmarks.md:{ln} (H100 v0.1.3, {HEADERS[r['algo']][5:]} table)",
                "speedup_sync": sp_sync, "speedup_recorded": sp_rec,
                "sync_over_recorded": sp_sync / sp_rec})

with open(HERE / "loop_sync.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)

print("| algorithm | E × T | loop sync (ms) | loop recorded (ms) | tree full-call, recorded (ms) | "
      "speedup sync | speedup recorded | sync / recorded | source line |")
print("|---|---|---|---|---|---|---|---|---|")
for r in out:
    print(f"| {r['algo']} | {r['num_envs']} × {r['seq_len']} | {r['loop_sync_ms']:.3f} | "
          f"{r['loop_recorded_ms']:.3f} | {r['tree_fullcall_recorded_ms']:.3f} | {r['speedup_sync']:.1f}× | "
          f"{r['speedup_recorded']:.1f}× | {r['sync_over_recorded']:.2f} | {r['tree_source'].split(' ')[0]} |")
