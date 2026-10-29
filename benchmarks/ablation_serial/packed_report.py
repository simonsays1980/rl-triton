"""Task 3 report: packed_raw.jsonl + results.csv -> packed_results.csv,
packed_sweep.csv, fig_packed.pdf/.png, statistics on stdout.

    python benchmarks/ablation_serial/packed_report.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_report import MARKERS, PALETTE  # noqa: E402

HERE = Path(__file__).resolve().parent
JOIN = ("tree", "serial_env", "serial_time", "serial_time+transpose", "scan")


def main():
    packed, sweep = {}, []
    for line in (HERE / "packed_raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "result":
            packed[(r["num_envs"], r["seq_len"])] = r
        else:
            sweep.append(r)
    base = {}
    for r in csv.DictReader(open(HERE / "results.csv")):
        base[(int(r["num_envs"]), int(r["seq_len"]), r["kernel"])] = r
    shapes = sorted(packed)

    rows = []
    for e, t in shapes:
        p = packed[(e, t)]
        rows.append({"num_envs": e, "seq_len": t, "kernel": "packed", "layout": p.get("layout", ""),
                     "best_config": p.get("config", ""), "device_ms": p.get("device_ms", ""),
                     "fullcall_ms": p.get("fullcall_ms", ""), "validated": p["validated"],
                     "device_best_config": p.get("device_best_config", ""),
                     "device_best_ms": p.get("device_best_ms", ""),
                     "source": "packed_raw.jsonl (this task)"})
        for k in JOIN:
            b = base.get((e, t, k))
            if b:
                rows.append({"num_envs": e, "seq_len": t, "kernel": k, "layout": b["layout"],
                             "best_config": b["best_config"], "device_ms": b["device_ms"],
                             "fullcall_ms": b["fullcall_ms"], "validated": b["validated"],
                             "device_best_config": "", "device_best_ms": "",
                             "source": "results.csv (earlier ablation run)"})
    for c in sweep:
        if c.get("skip_reason"):
            rows.append({"num_envs": c["num_envs"], "seq_len": c["seq_len"], "kernel": "packed (skipped config)",
                         "layout": "env-major [E,T]", "best_config": c["config"], "device_ms": "",
                         "fullcall_ms": "", "validated": f"skipped: {c['skip_reason']}",
                         "device_best_config": "", "device_best_ms": "",
                         "source": "packed_raw.jsonl (this task)"})
    with open(HERE / "packed_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(HERE / "packed_sweep.csv", "w", newline="") as f:
        keys = ["num_envs", "seq_len", "config", "block_e", "num_warps", "validated", "max_abs_err",
                "screen_device_ms", "fullcall_ms", "error", "skip_reason"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sweep)

    def pdev(e, t):  # packed device time: best-device config, re-measured (5 trials)
        p = packed[(e, t)]
        return p.get("device_best_ms") if p["validated"] else None

    def bdev(e, t, k, m="device_ms"):
        b = base.get((e, t, k))
        return float(b[m]) if b and b["validated"] == "True" else None

    print("## failures")
    bad = [s for s in sweep if not s["validated"]]
    skipped = [s for s in bad if s.get("skip_reason")]
    failed = [s for s in bad if not s.get("skip_reason")]
    print(f"  {len(skipped)}/{len(sweep)} packed configs skipped (never run): "
          + (", ".join(f"{s['num_envs']}x{s['seq_len']} {s['config']} ({s['skip_reason']})" for s in skipped) or "none"))
    print(f"  {len(failed)}/{len(sweep)} packed configs failed compile/correctness: "
          + (", ".join(f"{s['num_envs']}x{s['seq_len']} {s['config']} ({s.get('error') or 'mismatch'})" for s in failed) or "none"))
    print("\n## per shape (device us: tree / packed[best-dev cfg] / S_time / S_env; full-call us: tree / packed / S_time)")
    for e, t in shapes:
        p = packed[(e, t)]
        if not p["validated"]:
            print(f"  {e}x{t}: packed NOT VALIDATED")
            continue
        tr, st, se = bdev(e, t, "tree"), bdev(e, t, "serial_time"), bdev(e, t, "serial_env")
        trf, stf = bdev(e, t, "tree", "fullcall_ms"), bdev(e, t, "serial_time", "fullcall_ms")
        print(f"  {e}x{t}: dev {tr*1e3:.1f} / {pdev(e, t)*1e3:.1f} [{p['device_best_config']}] / {st*1e3:.1f} / {se*1e3:.1f}"
              f"  -> tree/packed {tr/pdev(e, t):.2f}x, S_time/packed {st/pdev(e, t):.2f}x"
              f" | full {trf*1e3:.1f} / {p['fullcall_ms']*1e3:.1f} [{p['config']}] / {stf*1e3:.1f}"
              f"  -> S_time/packed {stf/p['fullcall_ms']:.2f}x")

    envs = sorted({e for e, _ in shapes if e != 512})
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), dpi=150, sharey=True)
    for ax, k, title in ((axes[0], "tree", "shipped tree / packed tree"),
                         (axes[1], "serial_time", "S_time / packed tree")):
        for i, e in enumerate(envs + [512]):
            pts = [(t, bdev(e, t, k) / pdev(e, t)) for ee, t in shapes
                   if ee == e and pdev(e, t) and bdev(e, t, k)]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=PALETTE[i], marker=MARKERS[i], markersize=5,
                        linewidth=2, label=f"num_envs={e}" + (" (long-T control)" if e == 512 else ""))
        ax.axhline(1.0, color="#52514e", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Sequence length T")
        ax.set_title(title + " (device time)", fontsize=9)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Device-time ratio\n>1: packed faster")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(HERE / "fig_packed.pdf")
    fig.savefig(HERE / "fig_packed.png")
    print("\nwrote packed_results.csv, packed_sweep.csv, fig_packed.pdf/.png")


if __name__ == "__main__":
    main()
