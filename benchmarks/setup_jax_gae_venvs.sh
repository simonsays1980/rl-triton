#!/usr/bin/env bash
# Builds the two virtualenvs benchmark_gae_vs_jax_scan_{triton,jax,report}.py
# needs, in the right locations, at the right (verified-compatible) package
# versions.
#
# WHY TWO VENVS: torch==2.4.1 (this repo's pinned version) hard-requires
# nvidia-cudnn-cu12==9.1.0.70; jax[cuda12]'s GPU plugin requires cuDNN
# >=9.8.0. The two cannot coexist in one venv -- confirmed on a real pod,
# not theoretical: sharing one venv makes XLA's GPU compiler hard-crash with
# a null cuDNN handle (RET_CHECK dnn_support != nullptr) on ANY GPU program,
# even though this benchmark's actual computation never calls a cuDNN op.
# See NOTES.md's "torch==2.4.1 and jax[cuda12] cannot share one venv" section
# and benchmark_gae_vs_jax_scan_triton.py's module docstring for the full
# story.
#
# Usage (from the repo root, or anywhere -- paths below are absolute):
#   bash benchmarks/setup_jax_gae_venvs.sh
#
# Creates:
#   ~/venvs/rl-triton-torch  -- torch==2.4.1, triton==3.0.0, matplotlib, etc.
#                                (this repo's [dev] extra)
#   ~/venvs/rl-triton-jax    -- jax[cuda12], numpy only. NEVER installs this
#                                repo's package or [dev] extra here -- that
#                                would pull in the conflicting torch pin.
#
# Override the venv parent directory with VENV_DIR, e.g.:
#   VENV_DIR=/workspace/venvs bash benchmarks/setup_jax_gae_venvs.sh
#
# After this script completes, run the three-phase pipeline:
#   source ~/venvs/rl-triton-torch/bin/activate
#   python benchmarks/benchmark_gae_vs_jax_scan_triton.py
#   deactivate
#
#   source ~/venvs/rl-triton-jax/bin/activate
#   python benchmarks/benchmark_gae_vs_jax_scan_jax.py
#   deactivate
#
#   source ~/venvs/rl-triton-torch/bin/activate
#   python benchmarks/benchmark_gae_vs_jax_scan_report.py
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs}"
TORCH_VENV="${VENV_DIR}/rl-triton-torch"
JAX_VENV="${VENV_DIR}/rl-triton-jax"

# Repo root -- this script lives in benchmarks/, so its parent's parent.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================================================"
echo "Repo root:  ${REPO_ROOT}"
echo "Venv dir:   ${VENV_DIR}"
echo "================================================================================"

# ---------------------------------------------------------------------------
# 1. torch/triton venv (this repo's package + [dev] extra)
# ---------------------------------------------------------------------------
echo
echo "--- Building torch/triton venv at ${TORCH_VENV} ---"
python3 -m venv "${TORCH_VENV}"
"${TORCH_VENV}/bin/pip" install --upgrade pip --quiet
"${TORCH_VENV}/bin/pip" install -e "${REPO_ROOT}[dev]"

echo
echo "torch venv versions:"
"${TORCH_VENV}/bin/python" -c "
import torch, triton
print('  torch:  ', torch.__version__, '(cuda', torch.version.cuda, ')')
print('  triton: ', triton.__version__)
print('  cuda ok:', torch.cuda.is_available())
"

# ---------------------------------------------------------------------------
# 2. JAX-only venv -- deliberately NOT this repo's package or [dev]/[jax]
#    extra (installing this repo's [jax] extra here would still be fine on
#    its own, but the point of a SEPARATE venv is that nothing pulls in the
#    pinned torch==2.4.1 here; installing jax[cuda12] standalone is what
#    guarantees that).
# ---------------------------------------------------------------------------
echo
echo "--- Building JAX-only venv at ${JAX_VENV} ---"
python3 -m venv "${JAX_VENV}"
"${JAX_VENV}/bin/pip" install --upgrade pip --quiet
"${JAX_VENV}/bin/pip" install "jax[cuda12]" numpy

echo
echo "JAX venv versions:"
"${JAX_VENV}/bin/python" -c "
import jax
print('  jax:     ', jax.__version__)
print('  devices: ', jax.devices())
print('  backend: ', jax.default_backend())
gpu = [d for d in jax.devices() if d.platform == 'gpu']
if not gpu:
    print('  WARNING: no GPU device found -- check the [cuda12] extra installed a')
    print('           CUDA-enabled jaxlib matching this machine\'s CUDA version.')
"

echo
echo "================================================================================"
echo "Both venvs ready. Next:"
echo "  source ${TORCH_VENV}/bin/activate"
echo "  python ${REPO_ROOT}/benchmarks/benchmark_gae_vs_jax_scan_triton.py"
echo "================================================================================"
