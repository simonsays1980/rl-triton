"""Step 4 gap fill: PPO end-to-end, hidden=(128,128), num_envs=16384,
seq_len=128, Triton arm vs. the compiled-scan arm (previously measured only
against the loop arm).

Reuses benchmarks/ppo_e2e_measurement.py's own Config / correctness_gate /
measure / _row unchanged -- only the config (not in that script's build_grid)
and the arm pair are chosen here. Protocol: eager nets, event timing,
n_warmup=40, n_iters=30, interleaved rotating arm order, 9 reruns each in a
fresh subprocess, |non-GAE vs ref - 1| >= 0.01 flagged as contaminated.

    python benchmarks/ablation_serial/e2e_gapfill.py            # 9 reruns -> e2e_gapfill.csv
    python benchmarks/ablation_serial/e2e_gapfill.py --one OUT  # (internal) one rerun
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_RERUNS = 9
ARMS = ("triton", "scan")
FIELDS = ["rerun", "arm", "num_envs", "seq_len", "hidden", "compiled_net", "n_warmup", "n_iters",
          "gate", "gate_max_abs", "total_ms", "total_iqr_ms", "gae_ms", "gae_dev_ms", "gae_share_pct",
          "e2e_vs_triton", "e2e_vs_triton_iqr", "resid_ms", "nongae_vs_ref", "contaminated"]


def one(out_path):
    sys.path.insert(0, str(HERE.parent))
    import torch
    import ppo_e2e_measurement as m

    cfg = m.Config(16384, 128, (128, 128), False, n_iters=30, n_warmup=40)
    gate = m.correctness_gate(cfg, ARMS)
    samples = m.measure(cfg, timing="event", arms=ARMS)
    r = m._row(cfg, samples, ARMS)
    rows = []
    for arm in ARMS:
        a = r[arm]
        # correctness_gate checks every arm against `triton` (raises on FAIL);
        # triton itself is gated vs. the independent reference in results.csv.
        ok, max_abs, _ = gate.get(arm, (True, 0.0, 0.0))
        rows.append({
            "arm": arm, "num_envs": cfg.num_envs, "seq_len": cfg.seq_len, "hidden": "128x128",
            "compiled_net": cfg.compiled, "n_warmup": cfg.n_warmup, "n_iters": cfg.n_iters,
            "gate": "REF" if arm == "triton" else ("PASS" if ok else "FAIL"), "gate_max_abs": max_abs,
            "total_ms": a["total"], "total_iqr_ms": a["total_iqr"], "gae_ms": a["gae"],
            "gae_dev_ms": a["dev"], "gae_share_pct": a["share"],
            "e2e_vs_triton": a["speedup"], "e2e_vs_triton_iqr": a["speedup_iqr"],
            "resid_ms": a["resid"], "nongae_vs_ref": a["nongae_vs_ref"],
            "contaminated": abs(a["nongae_vs_ref"] - 1.0) >= m._CONTAMINATION_THRESHOLD,
        })
    Path(out_path).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--one")
    a = p.parse_args()
    if a.one:
        one(a.one)
        return
    all_rows = []
    for i in range(N_RERUNS):
        tmp = HERE / f".e2e_rerun_{i}.json"
        print(f"rerun {i + 1}/{N_RERUNS}", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--one", str(tmp)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-3000:], flush=True)
            continue
        for row in json.loads(tmp.read_text())["rows"]:
            all_rows.append({"rerun": i, **row})
            print(f"  {row['arm']:<7} total {row['total_ms']:.3f}ms  GAE {row['gae_ms']:.4f}ms "
                  f"share {row['gae_share_pct']:.3f}%  vs-triton {row['e2e_vs_triton']:.4f}  "
                  f"non-GAE {row['nongae_vs_ref']:.4f}{'  CONTAMINATED' if row['contaminated'] else ''}",
                  flush=True)
        tmp.unlink()
    with open(HERE / "e2e_gapfill.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)


if __name__ == "__main__":
    main()
