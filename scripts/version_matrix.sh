#!/usr/bin/env bash
# Torch/Triton version-compatibility matrix -- run on the RunPod GPU pod.
#
# Tests three endpoints (floor, one middle version, latest), not the full
# torch x triton grid: PyTorch hard-pins an exact Triton per release, so the
# test axis is the torch version -- Triton follows automatically via each
# torch wheel's own dependency metadata. Never pin triton explicitly here;
# only assert the resolved version matches what each torch wheel is known to
# pull, to catch the case where PyTorch's own pin silently changes.
#
# Every venv is created with Python 3.12, specifically -- not "whatever's on
# PATH." torch==2.4.1's triton dependency carries the marker
# `python_version < "3.13"`, and ships no cp313+ wheels itself either, so on
# Python 3.13+ the floor venv would silently resolve WITHOUT triton. uv
# provides a standalone CPython 3.12 (python-build-standalone) independent of
# whatever Python this base image ships -- see .runpod/start.sh, which
# pre-fetches it via `uv python install 3.12` at pod boot. This script never
# installs anything at the interpreter/system level itself: if a 3.12
# interpreter can't be obtained, it fails loudly rather than substituting
# another version.
#
# Usage:
#   scripts/version_matrix.sh              # run all three endpoints
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="${VENV_ROOT:-/tmp/rl-triton-version-matrix}"

# torch version -> expected triton version, read from each torch wheel's own
# METADATA (Requires-Dist: triton==...) -- not from memory, and not 1:1 with
# the torch minor (e.g. torch 2.12.0 pins triton 3.7.0, but 2.12.1 pins
# 3.7.1; the triton patch is coupled to the torch patch, not just the minor).
declare -A EXPECTED_TRITON=(
    [2.4.1]=3.0.0
    [2.9.1]=3.5.1
    [2.12.0]=3.7.0
)
ENDPOINTS=(2.4.1 2.9.1 2.12.0)

if ! command -v uv >/dev/null; then
    echo "FATAL: uv not found on PATH. This script requires uv to create a" >&2
    echo "Python 3.12 venv independent of the base image's Python -- see" >&2
    echo ".runpod/start.sh, which installs it at pod boot. Refusing to fall" >&2
    echo "back to system python." >&2
    exit 1
fi

declare -A RESULTS

for TORCH_VERSION in "${ENDPOINTS[@]}"; do
    EXPECTED_TRITON_VERSION="${EXPECTED_TRITON[$TORCH_VERSION]}"
    VENV_DIR="${VENV_ROOT}/torch-${TORCH_VERSION}"

    echo ""
    echo "=============================================================================="
    echo "torch==${TORCH_VERSION} (expecting triton==${EXPECTED_TRITON_VERSION})"
    echo "=============================================================================="

    rm -rf "${VENV_DIR}"
    uv venv --python 3.12 "${VENV_DIR}"

    # Fail loudly rather than silently continuing on a wrong interpreter --
    # no fallback to whatever uv actually produced.
    ACTUAL_PYTHON_VERSION="$("${VENV_DIR}/bin/python" -V)"
    if [[ "${ACTUAL_PYTHON_VERSION}" != "Python 3.12."* ]]; then
        echo "FATAL: venv for torch==${TORCH_VERSION} resolved to" \
             "'${ACTUAL_PYTHON_VERSION}', not Python 3.12.x. Refusing to" \
             "continue -- torch==2.4.1's triton dependency requires" \
             "python_version < 3.13, and a wrong interpreter here would" \
             "silently produce a venv with no triton at all." >&2
        exit 1
    fi

    VENV_PY="${VENV_DIR}/bin/python"

    uv pip install --quiet --python "${VENV_PY}" "torch==${TORCH_VERSION}"

    ACTUAL_TRITON_VERSION="$("${VENV_PY}" -c "import triton; print(triton.__version__)")"
    if [[ "${ACTUAL_TRITON_VERSION}" != "${EXPECTED_TRITON_VERSION}" ]]; then
        echo "FATAL: torch==${TORCH_VERSION} pulled triton==${ACTUAL_TRITON_VERSION}," \
             "expected triton==${EXPECTED_TRITON_VERSION}. PyTorch's own pin" \
             "has changed since this matrix was last verified against live" \
             "wheel metadata -- update EXPECTED_TRITON rather than ignoring" \
             "this." >&2
        exit 1
    fi

    uv pip install --quiet --python "${VENV_PY}" --no-deps -e "${REPO_DIR}"
    uv pip install --quiet --python "${VENV_PY}" pytest pytest-benchmark numpy

    echo "torch:  ${TORCH_VERSION}"
    echo "triton: ${ACTUAL_TRITON_VERSION} (confirmed)"

    if "${VENV_PY}" -m pytest "${REPO_DIR}/tests/" -v -m "not slow and not perf" --tb=short; then
        RESULTS[$TORCH_VERSION]="PASS"
    else
        RESULTS[$TORCH_VERSION]="FAIL"
    fi
done

echo ""
echo "=============================================================================="
echo "Version matrix summary"
echo "=============================================================================="
FAILED=0
for TORCH_VERSION in "${ENDPOINTS[@]}"; do
    printf "  torch==%-10s triton==%-8s %s\n" \
        "${TORCH_VERSION}" "${EXPECTED_TRITON[$TORCH_VERSION]}" "${RESULTS[$TORCH_VERSION]}"
    [[ "${RESULTS[$TORCH_VERSION]}" == "FAIL" ]] && FAILED=1
done

exit "${FAILED}"
