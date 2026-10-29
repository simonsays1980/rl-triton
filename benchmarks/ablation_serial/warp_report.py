"""Task A report: warp_raw.jsonl + longT_results.csv -> warp_retune.csv (+ markdown table on stdout).

Per shape: shipped tree time (this run's row at the shipped num_warps, plus the Task 2 tree row
for cross-checking), best re-tuned config (lowest full-call over num_warps in {4,8,16,32};
only validated configs), and the ratio best-tree / X for X in {compiled scan, S_env, S_time}
taken from longT_results.csv (ratio < 1 = tree faster).
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = [json.loads(l) for l in (HERE / "warp_raw.jsonl").read_text().splitlines() if l.strip()]
lt = {(int(r["num_envs"]), int(r["seq_len"]), r["kernel"]): r
      for r in csv.DictReader(open(HERE / "longT_results.csv"))}

shapes = sorted({(r["num_envs"], r["seq_len"]) for r in raw}, key=lambda s: (s[1], s[0]))
out = []
for E, T in shapes:
    rows = [r for r in raw if (r["num_envs"], r["seq_len"]) == (E, T)]
    ok = [r for r in rows if r.get("validated") is True]
    ship = next(r for r in rows if r["is_shipped_config"])
    best = min(ok, key=lambda r: r["fullcall_ms"])
    rec = {"num_envs": E, "seq_len": T,
           "shipped_num_warps": ship["shipped_num_warps"],
           "shipped_fullcall_ms": ship.get("fullcall_ms"), "shipped_device_ms": ship.get("device_ms"),
           "task2_tree_fullcall_ms": float(lt[(E, T, "tree")]["fullcall_ms"]),
           "task2_tree_device_ms": float(lt[(E, T, "tree")]["device_ms"]),
           "best_num_warps": best["num_warps"],
           "best_fullcall_ms": best["fullcall_ms"], "best_device_ms": best["device_ms"],
           "retune_gain_fullcall": ship["fullcall_ms"] / best["fullcall_ms"],
           "n_validated": len(ok), "n_configs": len(rows),
           "per_warps_fullcall_ms": ";".join(f"{r['num_warps']}:{r['fullcall_ms']:.4f}" if r.get("validated") else
                                              f"{r['num_warps']}:{r.get('skip_reason', 'invalid')}" for r in rows),
           "per_warps_device_ms": ";".join(f"{r['num_warps']}:{r['device_ms']:.4f}" for r in ok)}
    for k, name in (("scan", "scan"), ("serial_env", "senv"), ("serial_time", "stime")):
        b = lt[(E, T, k)]
        rec[f"{name}_fullcall_ms"] = float(b["fullcall_ms"])
        rec[f"{name}_device_ms"] = float(b["device_ms"])
        rec[f"best_tree_over_{name}_fullcall"] = best["fullcall_ms"] / float(b["fullcall_ms"])
        rec[f"best_tree_over_{name}_device"] = best["device_ms"] / float(b["device_ms"])
    out.append(rec)

with open(HERE / "warp_retune.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)

print("| E × T | shipped warps | shipped full / dev (ms) | best warps | best full / dev (ms) | gain | "
      "tree/scan full (dev) | tree/S_env full (dev) | tree/S_time full (dev) |")
print("|---|---|---|---|---|---|---|---|---|")
for r in out:
    print(f"| {r['num_envs']} × {r['seq_len']} | {r['shipped_num_warps']} | "
          f"{r['shipped_fullcall_ms']:.4f} / {r['shipped_device_ms']:.4f} | {r['best_num_warps']} | "
          f"{r['best_fullcall_ms']:.4f} / {r['best_device_ms']:.4f} | {r['retune_gain_fullcall']:.2f}× | "
          f"{r['best_tree_over_scan_fullcall']:.2f} ({r['best_tree_over_scan_device']:.2f}) | "
          f"{r['best_tree_over_senv_fullcall']:.3f} ({r['best_tree_over_senv_device']:.3f}) | "
          f"{r['best_tree_over_stime_fullcall']:.3f} ({r['best_tree_over_stime_device']:.3f}) |")
