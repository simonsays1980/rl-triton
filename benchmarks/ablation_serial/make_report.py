"""raw_results.jsonl -> results.csv, sweep_serial.csv, fig_tree_vs_serial.pdf,
and the crossover / ratio statistics printed for SUMMARY.md.

    python benchmarks/ablation_serial/make_report.py
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_ablation import GRID_A, GRID_B  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_results.jsonl"

# Validated default categorical palette (dataviz skill), fixed slot order, same
# source as benchmarks/benchmark_gae_vs_pufferlib.py's COLORS. Color = num_envs.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SERIAL = ("serial_env", "serial_time", "serial_env_naive", "serial_time_naive", "serial_time+transpose")


def grid_of(shape):
    a, b = shape in GRID_A, shape in GRID_B
    return "A+B" if a and b else ("A" if a else "B")


def load():
    results, sweep = {}, []
    for line in RAW.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "result":
            results[(r["num_envs"], r["seq_len"], r["kernel"])] = r
        else:
            sweep.append(r)
    return results, sweep


def crossover(points):
    """points: [(seq_len, ratio)] sorted. Returns list of crossing descriptions,
    log2-interpolated between bracketing seq_lens."""
    out = []
    for (t1, r1), (t2, r2) in zip(points, points[1:]):
        if (r1 - 1) * (r2 - 1) < 0:
            l1, l2 = math.log2(t1), math.log2(t2)
            lr1, lr2 = math.log(r1), math.log(r2)
            x = l1 + (0 - lr1) * (l2 - l1) / (lr2 - lr1)
            direction = "serial->tree" if r2 > 1 else "tree->serial"
            out.append(f"between T={t1} ({r1:.2f}x) and T={t2} ({r2:.2f}x), "
                       f"interp. T~{2 ** x:.0f} [{direction} wins above]")
    return out


def main():
    results, sweep = load()
    shapes = sorted({(e, t) for e, t, _ in results})

    rows = []
    for e, t in shapes:
        tree = results.get((e, t, "tree"))
        for k in ("tree", "tree_lib", "scan", *SERIAL, "transpose_inputs"):
            r = results.get((e, t, k))
            if r is None:
                continue
            ok = r.get("validated", False)
            row = {"grid": grid_of((e, t)), "num_envs": e, "seq_len": t, "kernel": k,
                   "layout": r["layout"], "best_config": r.get("config", ""),
                   "device_ms": r.get("device_ms", "") if ok else "",
                   "fullcall_ms": r.get("fullcall_ms", "") if ok else "",
                   "validated": ok, "max_abs_err": r.get("max_abs_err", ""),
                   "tree_speedup_fullcall": "", "tree_speedup_device": "",
                   "note": r.get("note", "")}
            if k == "transpose_inputs":
                row["validated"] = "n/a (no GAE output)"
            if ok and tree and tree.get("validated") and k not in ("tree", "transpose_inputs"):
                row["tree_speedup_fullcall"] = r["fullcall_ms"] / tree["fullcall_ms"]
                row["tree_speedup_device"] = r["device_ms"] / tree["device_ms"]
            rows.append(row)

    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    keys = ["num_envs", "seq_len", "kernel", "layout", "config", "block_e", "num_warps", "unroll",
            "num_stages", "validated", "max_abs_err", "screen_device_ms", "fullcall_ms", "error"]
    with open(HERE / "sweep_serial.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sweep)

    # ---- statistics -------------------------------------------------------
    def ratio(e, t, k, metric):
        tr, s = results.get((e, t, "tree")), results.get((e, t, k))
        if not (tr and s and tr.get("validated") and s.get("validated")):
            return None
        return s[metric] / tr[metric]

    envs = sorted({e for e, _ in shapes})
    print("## invalid / excluded")
    for (e, t, k), r in sorted(results.items()):
        if not r.get("validated", True):
            print(f"  {e}x{t} {k}: NOT VALIDATED -- {r.get('note', '')}")
    n_bad = sum(1 for s in sweep if not s["validated"])
    print(f"  serial sweep configs failing compile/correctness: {n_bad}/{len(sweep)}")
    print("  " + "; ".join(sorted({f"{s['kernel']} {s['config']} ({s.get('error', 'mismatch')})"
                                    for s in sweep if not s["validated"]}))[:1500])

    for metric in ("fullcall_ms", "device_ms"):
        print(f"\n## tree speedup over serial ({metric}): ratio = serial / tree")
        for k in ("serial_env", "serial_time", "serial_env_naive", "serial_time_naive",
                  "serial_time+transpose", "scan"):
            vals = [(ratio(e, t, k, metric), e, t) for e, t in shapes]
            vals = [v for v in vals if v[0] is not None]
            mx, mn = max(vals), min(vals)
            wins = [f"{e}x{t} ({r:.2f}x)" for r, e, t in sorted(vals, key=lambda v: (v[1], v[2])) if r < 1]
            print(f"  {k:<22} max {mx[0]:.2f}x @ {mx[1]}x{mx[2]}  min {mn[0]:.2f}x @ {mn[1]}x{mn[2]}  "
                  f"serial/baseline wins at {len(wins)}/{len(vals)}: {', '.join(wins) or 'none'}")
        for k in ("serial_env", "serial_time", "serial_time+transpose"):
            print(f"  crossovers {k}:")
            for e in envs:
                pts = sorted((t, ratio(e, t, k, metric)) for ee, t in shapes if ee == e)
                pts = [p for p in pts if p[1] is not None]
                cx = crossover(pts)
                desc = "; ".join(cx) if cx else (
                    "no crossover in range ("
                    + ("tree wins everywhere" if all(r > 1 for _, r in pts) else
                       "serial wins everywhere" if all(r < 1 for _, r in pts) else "mixed") + ")")
                print(f"    num_envs={e:<6} T={[t for t, _ in pts]}: {desc}")
                print(f"      ratios: {', '.join(f'{t}:{r:.2f}' for t, r in pts)}")

    # ---- figures ----------------------------------------------------------
    # Main figure is full-call (spec); the device-time twin isolates kernel
    # cost from the ~25-30us Triton launch floor every kernel shares.
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    for metric, stem, ylab in (
            ("fullcall_ms", "fig_tree_vs_serial", "Tree speedup over serial (full-call)"),
            ("device_ms", "fig_tree_vs_serial_device", "Tree speedup over serial (device time)")):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), dpi=150, sharey=True)
        for ax, k, title in ((axes[0], "serial_env", "vs. S_env (env-major [E,T], strided)"),
                             (axes[1], "serial_time", "vs. S_time (time-major [T,E], coalesced)")):
            for i, e in enumerate(envs):
                pts = sorted((t, ratio(e, t, k, metric)) for ee, t in shapes if ee == e)
                pts = [p for p in pts if p[1] is not None]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=PALETTE[i], marker=MARKERS[i], markersize=5,
                        linewidth=2 if len(pts) > 1 else 0, label=f"num_envs={e}")
            if metric == "fullcall_ms":
                ax.axhspan(0.9, 1.1, color="#52514e", alpha=0.08, linewidth=0)
            ax.axhline(1.0, color="#52514e", linestyle="--", linewidth=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("Sequence length T")
            ax.set_title(title, fontsize=9)
            ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        axes[0].set_ylabel(ylab + "\n>1: tree faster, <1: serial faster")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        fig.savefig(HERE / f"{stem}.pdf")
        fig.savefig(HERE / f"{stem}.png")
        plt.close(fig)
    print("\nwrote results.csv, sweep_serial.csv, fig_tree_vs_serial{,_device}.pdf/.png")


if __name__ == "__main__":
    main()
