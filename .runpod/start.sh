#!/usr/bin/env bash
# RunPod pod start script — self-hosted GitHub Actions runner.
#
# Set this as the pod's "Container Start Command" in the RunPod UI.
# Every boot reconstructs state from scratch (no network volume).
#
# Required env vars (set in RunPod pod environment):
#   GITHUB_PAT   — classic PAT with repo scope, or fine-grained with
#                  Administration: write (needed to mint registration tokens)
#   GITHUB_REPO  — owner/repo, e.g. simonsays1980/rl-triton
#
# Optional:
#   RUNNER_NAME  — defaults to runpod-$(hostname)

set -euo pipefail

REPO_URL="https://x-access-token:${GITHUB_PAT}@github.com/${GITHUB_REPO}.git"
WORK_DIR="/root/actions-runner"
REPO_DIR="/root/rl-triton"

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
# 2. Install package + dev deps
#    The repo is already at REPO_DIR — cloned by the Container Start Command
#    one-liner before this script was invoked.
#    --no-deps skips torch/triton so the pre-installed CUDA build is preserved.
#    The individual dev extras (pytest, pytest-benchmark, numpy) don't pull
#    torch, so a plain install of those is safe.
# ---------------------------------------------------------------------------
pip install --quiet --no-deps -e "${REPO_DIR}"
pip install --quiet pytest pytest-benchmark numpy

# ---------------------------------------------------------------------------
# 3. Download latest actions/runner release
# ---------------------------------------------------------------------------
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

RUNNER_VERSION=$(curl -fsSL \
    -H "Authorization: Bearer ${GITHUB_PAT}" \
    "https://api.github.com/repos/actions/runner/releases/latest" \
    | jq -r '.tag_name' | sed 's/^v//')

curl -fsSL \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    | tar -xz

# ---------------------------------------------------------------------------
# 4. Mint a fresh registration token (short-lived — must be done at boot)
# ---------------------------------------------------------------------------
REG_TOKEN=$(curl -fsSL \
    -X POST \
    -H "Authorization: Bearer ${GITHUB_PAT}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${GITHUB_REPO}/actions/runners/registration-token" \
    | jq -r '.token')

# ---------------------------------------------------------------------------
# 5. Configure the runner
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
# 6. Launch runner in a detached tmux session, then sleep to keep pod alive.
#    Attach over SSH with: tmux attach -t runner
# ---------------------------------------------------------------------------
tmux new-session -d -s runner \
    "cd ${WORK_DIR} && RUNNER_ALLOW_RUNASROOT=1 ./run.sh"

echo "Runner '${RUNNER_NAME}' started in tmux session 'runner'."
echo "Attach with: tmux attach -t runner"

sleep infinity
