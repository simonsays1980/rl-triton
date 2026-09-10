"""Step 3 (conditional): concatenated-episode ablation.

Only run because Step 2 showed a real Triton-vs-unilab_numpy difference
(3.8-10.1x). Tests: at UniLab's real N, does reshaping to fewer/longer rows
(concatenate 4 consecutive T=24 chunks into one T=96 row, same total
transitions, N/4 rows) change the TRITON DEVICE-ONLY time? The kernel resets
its scan carry at episode boundaries via the dones-driven beta=0 mechanism
(see rl_triton/ops/vtrace.py docstring), so forcing dones=1 at each internal
chunk boundary (steps 23, 47, 71 of the T=96 row) reproduces the same
per-chunk recurrence as four independent T=24 calls -- no kernel change
needed, this is purely a batching/grid-shape change.

Prediction going in: at N in the hundreds, this is unlikely to help
(Triton's per-call time in Step 2 was ~100us essentially flat across T=24..128,
i.e. launch-overhead-bound, not grid-occupancy-bound) -- reporting whatever is
actually measured, including if it confirms this prediction.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import capture_metadata, device_profile, today, unique_path, warmup_gpu  # noqa: E402

from rl_triton.ops.vtrace import compute_vtrace  # noqa: E402

UNILAB_REPO_DIR = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = Path(__file__).parent.parent
N_WARMUP, N_ITER = 20, 100
CHUNK_T = 24
CONCAT_FACTOR = 4

# (baseline N) for the real production shapes.
NS = [512, 1024]


def make_concat_inputs(N, gamma, seed=0, device="cuda"):
    """Baseline: (num_envs=N, seq_len=CHUNK_T). Concatenated: (num_envs=N/CONCAT_FACTOR,
    seq_len=CHUNK_T*CONCAT_FACTOR), with dones forced to 1 at each internal chunk
    boundary so the scan resets exactly like CONCAT_FACTOR independent calls."""
    g = torch.Generator(device=device).manual_seed(seed)
    T_base = CHUNK_T
    log_pi_target = -torch.rand(N, T_base, generator=g, device=device)
    log_pi_behavior = -torch.rand(N, T_base, generator=g, device=device)
    values = torch.randn(N, T_base, generator=g, device=device)
    rewards = torch.randn(N, T_base, generator=g, device=device)
    dones = (torch.rand(N, T_base, generator=g, device=device) < 0.05).float()
    last_value = torch.randn(N, generator=g, device=device)

    assert N % CONCAT_FACTOR == 0
    N_concat = N // CONCAT_FACTOR
    T_concat = T_base * CONCAT_FACTOR

    def reshape_concat(x):
        # [N, T_base] -> group CONCAT_FACTOR consecutive envs' rows end-to-end
        # along time: [N_concat, T_concat].
        return x.reshape(N_concat, CONCAT_FACTOR, T_base).reshape(N_concat, T_concat)

    log_pi_target_c = reshape_concat(log_pi_target)
    log_pi_behavior_c = reshape_concat(log_pi_behavior)
    values_c = reshape_concat(values)
    rewards_c = reshape_concat(rewards)
    dones_c = reshape_concat(dones).clone()
    # Force a boundary "done" at the end of each internal chunk except the
    # very last (that one already gets bootstrap_values via last_value).
    for k in range(CONCAT_FACTOR - 1):
        dones_c[:, (k + 1) * CHUNK_T - 1] = 1.0
    # last_value for the concatenated row must supply, at each internal
    # boundary, the *next* chunk's first value as the true continuation --
    # but since dones=1 there gates the bootstrap to zero in the delta/advantage
    # formula (non_terminal=0), any boundary value works EXCEPT the final
    # column, which needs the real last_value of the 4th sub-env in the group.
    last_value_c = last_value.reshape(N_concat, CONCAT_FACTOR)[:, -1].contiguous()

    return (
        (log_pi_target, log_pi_behavior, values, rewards, dones, last_value),
        (log_pi_target_c, log_pi_behavior_c, values_c, rewards_c, dones_c, last_value_c),
    )


def main():
    metadata = capture_metadata(UNILAB_REPO_DIR)
    gamma, rho_bar, c_bar = 0.99, 1.0, 1.0
    results = []

    for N in NS:
        (base_args, concat_args) = make_concat_inputs(N, gamma)
        lpt, lpb, v, r, d, lv = base_args
        lpt_c, lpb_c, v_c, r_c, d_c, lv_c = concat_args

        def base_fn():
            return compute_vtrace(lpt, lpb, v, r, d, truncateds=None,
                                   gamma=gamma, rho_bar=rho_bar, c_bar=c_bar, last_value=lv)

        def concat_fn():
            return compute_vtrace(lpt_c, lpb_c, v_c, r_c, d_c, truncateds=None,
                                   gamma=gamma, rho_bar=rho_bar, c_bar=c_bar, last_value=lv_c)

        warmup_gpu(base_fn, n_warmup=N_WARMUP)
        base_device_ms = device_profile(base_fn, n_iter=N_ITER, n_warmup=N_WARMUP)
        warmup_gpu(concat_fn, n_warmup=N_WARMUP)
        concat_device_ms = device_profile(concat_fn, n_iter=N_ITER, n_warmup=N_WARMUP)

        delta_pct = 100.0 * (concat_device_ms - base_device_ms) / base_device_ms
        row = {
            "N": N,
            "baseline_shape": f"[{N}, {CHUNK_T}]",
            "concatenated_shape": f"[{N // CONCAT_FACTOR}, {CHUNK_T * CONCAT_FACTOR}]",
            "baseline_device_us": base_device_ms * 1000.0,
            "concatenated_device_us": concat_device_ms * 1000.0,
            "delta_pct": delta_pct,
            "citable": True,  # device-only timing comparison, not a correctness claim
        }
        results.append(row)
        print(f"N={N:5d}  baseline {row['baseline_shape']:>12s} device={base_device_ms*1000:8.2f}us  "
              f"concatenated {row['concatenated_shape']:>12s} device={concat_device_ms*1000:8.2f}us  "
              f"delta={delta_pct:+.1f}%")

    out = {
        "metadata": metadata,
        "note": "Appended into unilab_vtrace_standalone JSON's concatenation_ablation key by run_all.sh; "
                "also written standalone here.",
        "chunk_t": CHUNK_T, "concat_factor": CONCAT_FACTOR,
        "results": results,
    }
    out_path = unique_path(OUT_DIR / f"unilab_vtrace_concat_ablation-{today()}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
