#!/usr/bin/env bash
# RunPod pod start script — self-hosted GitHub Actions runner.
#
# Set this as the pod's "Container Start Command" in the RunPod UI.
# Every boot reconstructs state from scratch (no network volume).
#
# Required env vars (set in RunPod pod environment):
#   GH_PAT      — classic PAT with repo scope, or fine-grained with
#                  Administration: write (needed to mint registration tokens)
#   GITHUB_REPO — owner/repo, e.g. simonsays1980/rl-triton
#
# Optional:
#   RUNNER_NAME  — defaults to runpod-$(hostname)

set -euo pipefail

REPO_URL="https://x-access-token:${GH_PAT}@github.com/${GITHUB_REPO}.git"
WORK_DIR="/root/actions-runner"
REPO_DIR="/root/rl-triton"

# Some RunPod base images set this, which floods stdout with repeated
# low-level PyTorch C++ log lines during benchmark/test runs.
unset TORCH_CPP_LOG_LEVEL

# Silence torch.compile/dynamo symbolic-shapes warnings (e.g. "q1 is not in
# var_ranges, defaulting to unknown range"), which are noisy but harmless
# during benchmark/test runs.
export TORCH_LOGS="-dynamo"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
apt-get update -qq
apt-get install -y --no-install-recommends tmux git curl jq ca-certificates

# GitHub CLI — needed by the safeguard workflow to post PR comments.
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list
apt-get update -qq
apt-get install -y --no-install-recommends gh

# ---------------------------------------------------------------------------
# 2. Install uv + pre-fetch Python 3.12
#    uv provides a standalone CPython 3.12 (python-build-standalone) that
#    version_matrix.sh's venvs use, independent of whatever Python this base
#    image ships (currently py3.11 per the image tag -- see this directory's
#    README.md). This
#    keeps that dependency out of the test script itself: version_matrix.sh
#    only asserts a 3.12 venv was created and fails loudly otherwise, it never
#    installs anything. Non-privileged, user-local install -- no apt/deadsnakes
#    PPA, so a flaky PPA can never masquerade as a test failure.
# ---------------------------------------------------------------------------
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

# ---------------------------------------------------------------------------
# 3. Clone repo and install package + dev deps
#    --no-deps skips torch/triton so the pre-installed CUDA build is preserved.
#    The individual dev extras (pytest, pytest-benchmark, numpy) don't pull
#    torch, so a plain install of those is safe.
# ---------------------------------------------------------------------------
git clone --depth=1 "${REPO_URL}" "${REPO_DIR}"
pip install --quiet --no-deps -e "${REPO_DIR}"
pip install --quiet pytest pytest-benchmark numpy

# ---------------------------------------------------------------------------
# 4. Download latest actions/runner release
# ---------------------------------------------------------------------------
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

RUNNER_VERSION=$(curl -fsSL \
    -H "Authorization: Bearer ${GH_PAT}" \
    "https://api.github.com/repos/actions/runner/releases/latest" \
    | jq -r '.tag_name' | sed 's/^v//')

curl -fsSL \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    | tar -xz

# ---------------------------------------------------------------------------
# 5. Mint a fresh registration token (short-lived — must be done at boot)
# ---------------------------------------------------------------------------
REG_TOKEN=$(curl -fsSL \
    -X POST \
    -H "Authorization: Bearer ${GH_PAT}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${GITHUB_REPO}/actions/runners/registration-token" \
    | jq -r '.token')

# ---------------------------------------------------------------------------
# 6. Configure the runner
#    Labels: gpu,triton — matched by runs-on: [self-hosted, linux, gpu]
#    --replace: re-registers if a stale runner entry exists from a prior boot.
#    RUNNER_ALLOW_RUNASROOT: RunPod pods run as root; the runner refuses otherwise.
# ---------------------------------------------------------------------------
RUNNER_NAME="${RUNNER_NAME:-runpod-$(hostname)}"

RUNNER_ALLOW_RUNASROOT=1 ./config.sh \
    --url "https://github.com/${GITHUB_REPO}" \
    --token "${REG_TOKEN}" \
    --name  "${RUNNER_NAME}" \
    --labels "gpu,triton" \
    --unattended \
    --replace

# ---------------------------------------------------------------------------
# 7. Launch runner in a detached tmux session, then sleep to keep pod alive.
#    Attach over SSH with: tmux attach -t runner
# ---------------------------------------------------------------------------
tmux new-session -d -s runner \
    "cd ${WORK_DIR} && RUNNER_ALLOW_RUNASROOT=1 ./run.sh"

echo "Runner '${RUNNER_NAME}' started in tmux session 'runner'."
echo "Attach with: tmux attach -t runner"

sleep infinity
