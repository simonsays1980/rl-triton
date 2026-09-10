"""Step 1: correctness of the Triton V-trace kernel vs. UniLab's real vtrace_advantages().

Formula mapping (verified line-by-line against uni_rl/algos/appo/learner.py's
vtrace_advantages, gamma/rho_bar/c_bar/delta/carry-boundary semantics -- see
run notes):

  UniLab [T, N]                          Triton compute_vtrace [num_envs, seq_len]
  ------------------------------------   -------------------------------------------
  target_log_probs      -> transpose ->  log_pi_target
  behavior_log_probs     -> transpose -> log_pi_behavior
  values                 -> transpose -> values
  rewards                -> transpose -> rewards
  dones                  -> transpose -> terminateds   (truncateds=None: kernel
                                                          docstring says terminateds
                                                          is then used for both roles,
                                                          which matches UniLab having
                                                          no separate truncated flag)
  bootstrap_values [N]   -> as-is    ->  last_value [N] (populates the boundary
                                                          column exactly like UniLab's
                                                          cat([values[1:], bootstrap_values]))
  gamma, clip_rho, clip_c -> as-is   ->  gamma, rho_bar, c_bar

Run with the isolated venv's python (unilab_rl needs torch>=2.7; this repo's
own kernels pin torch>=2.4.1, so this study runs outside the system env --
see UNILAB_VENV in run_all.sh).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import assert_finite_and_close, capture_metadata, today, unique_path  # noqa: E402

from uni_rl.algos.appo.learner import vtrace_advantages as unilab_vtrace_advantages  # noqa: E402
from rl_triton.ops.vtrace import compute_vtrace  # noqa: E402

UNILAB_REPO_DIR = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = Path(__file__).parent.parent  # benchmarks/
ATOL = RTOL = 1e-4

SHAPES = [
    (24, 8), (24, 64),      # small correctness-check shapes
    (24, 512), (24, 1024),  # Step 2/4 production shapes -- gate these too so
    (64, 512), (128, 512),  # citable in Step 2/3/4 rows means "this exact shape passed"
]


def make_inputs(T, N, gamma, clip_rho, clip_c, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    behavior_log_probs = -torch.rand(T, N, generator=g, device=device)
    target_log_probs = -torch.rand(T, N, generator=g, device=device)
    rewards = torch.randn(T, N, generator=g, device=device)
    values = torch.randn(T, N, generator=g, device=device)
    bootstrap_values = torch.randn(N, generator=g, device=device)
    dones = (torch.rand(T, N, generator=g, device=device) < 0.05).float()
    return behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones


def run_unilab(behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones,
               gamma, clip_rho, clip_c):
    return unilab_vtrace_advantages(
        behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones,
        gamma=gamma, clip_rho=clip_rho, clip_c=clip_c,
    )


def run_triton(behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones,
               gamma, clip_rho, clip_c):
    t = lambda x: x.transpose(0, 1).contiguous().float()
    vs, adv = compute_vtrace(
        log_pi_target=t(target_log_probs),
        log_pi_behavior=t(behavior_log_probs),
        values=t(values),
        rewards=t(rewards),
        terminateds=t(dones),
        truncateds=None,
        gamma=gamma, rho_bar=clip_rho, c_bar=clip_c,
        last_value=bootstrap_values.float(),
    )
    return vs.transpose(0, 1), adv.transpose(0, 1)


def main():
    metadata = capture_metadata(UNILAB_REPO_DIR)
    gamma, clip_rho, clip_c = 0.99, 1.0, 1.0
    records = []
    all_pass = True

    for (T, N) in SHAPES:
        behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones = \
            make_inputs(T, N, gamma, clip_rho, clip_c)

        vs_unilab, adv_unilab = run_unilab(
            behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones,
            gamma, clip_rho, clip_c,
        )
        vs_triton, adv_triton = run_triton(
            behavior_log_probs, target_log_probs, rewards, values, bootstrap_values, dones,
            gamma, clip_rho, clip_c,
        )

        vs_pass, vs_max, vs_mean, vs_rmse = assert_finite_and_close(
            vs_triton, vs_unilab, f"vs T={T} N={N}", atol=ATOL, rtol=RTOL
        )
        adv_pass, adv_max, adv_mean, adv_rmse = assert_finite_and_close(
            adv_triton, adv_unilab, f"advantages T={T} N={N}", atol=ATOL, rtol=RTOL
        )
        shape_pass = vs_pass and adv_pass
        all_pass = all_pass and shape_pass

        record = {
            "T": T, "N": N,
            "pass": shape_pass,
            "atol": ATOL, "rtol": RTOL,
            "vs_max_abs_err": vs_max, "vs_mean_abs_err": vs_mean, "vs_rmse": vs_rmse, "vs_pass": vs_pass,
            "adv_max_abs_err": adv_max, "adv_mean_abs_err": adv_mean, "adv_rmse": adv_rmse, "adv_pass": adv_pass,
        }
        records.append(record)
        status = "PASS" if shape_pass else "FAIL"
        print(f"T={T:4d} N={N:4d}  {status}  "
              f"vs: max_abs={vs_max:.3e} mean_abs={vs_mean:.3e} rmse={vs_rmse:.3e} ({'PASS' if vs_pass else 'FAIL'})  "
              f"adv: max_abs={adv_max:.3e} mean_abs={adv_mean:.3e} rmse={adv_rmse:.3e} ({'PASS' if adv_pass else 'FAIL'})")

    out = {
        "metadata": metadata,
        "gamma": gamma, "clip_rho": clip_rho, "clip_c": clip_c,
        "all_pass": all_pass,
        "records": records,
    }
    out_path = unique_path(OUT_DIR / f"unilab_vtrace_correctness-{today()}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"ALL SHAPES PASS: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
