"""Task 2 report: longT_raw.jsonl -> longT_results.csv, longT_sweep_serial.csv,
fig_longT.pdf/.png, and statistics on stdout (saved as longT_report_stats.txt).

    python benchmarks/ablation_serial/longT_report.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_report import MARKERS, PALETTE, crossover  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNELS = ("tree", "scan", "loop", "serial_env", "serial_env_naive", "serial_time",
           "serial_time_naive", "serial_time+transpose", "transpose_inputs")


def main():
    res, sweep = {}, []
    for line in (HERE / "longT_raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        (res.__setitem__((r["num_envs"], r["seq_len"], r["kernel"]), r) if r["kind"] == "result"
         else sweep.append(r))
    # Clean isolated rerun of the 12 compiled-scan rows (the original scan rows
    # overlapped a few seconds of an unrelated small GPU probe). Rule fixed in
    # advance: replace a recorded row only if the rerun differs by more than
    # the +-10% noise band in full-call or device time.
    rerun = HERE / "longT_scan_rerun.jsonl"
    if rerun.exists():
        for r in map(json.loads, rerun.read_text().splitlines()):
            key = (r["num_envs"], r["seq_len"], "scan")
            o = res[key]
            fr, dr = r["fullcall_ms"] / o["fullcall_ms"], r["device_ms"] / o["device_ms"]
            if r["validated"] and not (0.9 <= fr <= 1.1 and 0.9 <= dr <= 1.1):
                r["note"] = (f"replaced by isolated rerun (full-call {fr:.3f}x, device {dr:.3f}x "
                             f"of original {o['fullcall_ms']:.4f}/{o['device_ms']:.4f} ms)")
                res[key] = r
            else:
                o["note"] = f"isolated rerun within band (full-call {fr:.3f}x, device {dr:.3f}x); original kept"
    shapes = sorted({(e, t) for e, t, _ in res})
    envs = sorted({e for e, _ in shapes})

    def ok(e, t, k):
        r = res.get((e, t, k))
        return r if r and r.get("validated") else None

    rows = []
    for e, t in shapes:
        tree, loop = ok(e, t, "tree"), ok(e, t, "loop")
        for k in KERNELS:
            r = res.get((e, t, k))
            if r is None:
                continue
            v = r.get("validated", False)
            row = {"grid": "longT", "num_envs": e, "seq_len": t, "kernel": k, "layout": r["layout"],
                   "best_config": r.get("config", ""),
                   "device_ms": r.get("device_ms", "") if v else "",
                   "fullcall_ms": r.get("fullcall_ms", "") if v else "",
                   "validated": "n/a (no GAE output)" if k == "transpose_inputs" else v,
                   "max_abs_err": r.get("max_abs_err", ""),
                   "tree_speedup_fullcall": "", "tree_speedup_device": "",
                   "note": r.get("note", ""),
                   "loop_fullcall_ms": loop["fullcall_ms"] if loop else "not run (T > 16384)"}
            if v and tree and k not in ("tree", "transpose_inputs"):
                row["tree_speedup_fullcall"] = r["fullcall_ms"] / tree["fullcall_ms"]
                row["tree_speedup_device"] = r["device_ms"] / tree["device_ms"]
            rows.append(row)
    with open(HERE / "longT_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = ["num_envs", "seq_len", "kernel", "layout", "config", "block_e", "num_warps", "unroll",
            "num_stages", "validated", "max_abs_err", "screen_device_ms", "fullcall_ms", "error"]
    with open(HERE / "longT_sweep_serial.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sweep)

    def ratio(e, t, k, m):
        a, b = ok(e, t, k), ok(e, t, "tree")
        return a[m] / b[m] if a and b else None

    print("## not validated")
    for (e, t, k), r in sorted(res.items()):
        if not r.get("validated", True):
            print(f"  {e}x{t} {k}: {r.get('note', '')}")
    print(f"  serial sweep configs failing: {sum(not s['validated'] for s in sweep)}/{len(sweep)}")
    print("\n## absolute times (device / full-call, ms)")
    for e, t in shapes:
        cells = []
        for k in ("tree", "serial_env", "serial_time", "scan", "loop"):
            r = ok(e, t, k)
            cells.append(f"{k} {r['device_ms']:.4f}/{r['fullcall_ms']:.4f}" if r else f"{k} -")
        print(f"  {e}x{t}: " + " | ".join(cells))
    for m in ("fullcall_ms", "device_ms"):
        print(f"\n## tree speedup ({m}) = other / tree")
        for k in ("serial_env", "serial_time", "serial_time+transpose", "scan", "loop"):
            for e in envs:
                pts = [(t, ratio(e, t, k, m)) for ee, t in shapes if ee == e]
                pts = [p for p in pts if p[1] is not None]
                cx = crossover(pts)
                print(f"  {k:<22} E={e:<4} " + ", ".join(f"T{t}:{x:.2f}" for t, x in pts)
                      + f"  | crossover: {'; '.join(cx) if cx else 'none in range'}")

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=150, sharey=True)
    for ax, k, title in ((axes[0], "serial_env", "vs. S_env (env-major)"),
                         (axes[1], "serial_time", "vs. S_time (time-major)"),
                         (axes[2], "scan", "vs. compiled scan")):
        for i, e in enumerate(envs):
            pts = [(t, ratio(e, t, k, "fullcall_ms")) for ee, t in shapes if ee == e]
            pts = [p for p in pts if p[1] is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=PALETTE[i], marker=MARKERS[i], markersize=5, linewidth=2,
                        label=f"num_envs={e}")
        ax.axhline(1.0, color="#52514e", linestyle="--", linewidth=1)
        ax.axvline(16384, color="#52514e", linestyle=":", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Sequence length T")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Tree speedup (full-call)\n>1: tree faster")
    axes[2].text(16384 * 1.1, axes[2].get_ylim()[0] * 1.15,
                 "tree warps\nuntuned →", fontsize=7, color="#52514e")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HERE / "fig_longT.pdf")
    fig.savefig(HERE / "fig_longT.png")
    print("\nwrote longT_results.csv, longT_sweep_serial.csv, fig_longT.pdf/.png")


if __name__ == "__main__":
    main()
