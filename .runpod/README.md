# RunPod runner setup (maintainer notes)

Internal notes for setting up the self-hosted GPU runner that powers CI.

## Prerequisites

A GitHub PAT with either:
- **Classic**: `repo` scope
- **Fine-grained**: `Administration: Read & Write` on this repo

Needed for both cloning and minting runner registration tokens at boot.

## Launching the pod

1. RunPod UI → **Pods** → **+ New Pod**
2. GPU: RTX 4090 (sufficient for all benchmarks)
3. Image: a PyTorch image matching the target CUDA version, e.g.
   `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
4. Environment variables:

   | Variable | Value |
   |---|---|
   | `GH_PAT` | your PAT |
   | `GITHUB_REPO` | `simonsays1980/rl-triton` |
   | `RUNNER_NAME` | `runpod-gpu-1` (optional) |

5. Container Start Command -- paste this one-liner (bootstraps then runs start.sh):

   ```
   bash -c "git clone https://x-access-token:${GH_PAT}@github.com/${GITHUB_REPO}.git /root/rl-triton && bash /root/rl-triton/.runpod/start.sh"
   ```

6. Deploy. After ~60 s the runner appears as **Idle** under
   **GitHub → Settings → Actions → Runners**.

## Attaching to the runner session

```bash
tmux attach -t runner   # Detach: Ctrl-B D
```

## Notes

- The runner re-registers itself on every boot via `--replace` -- no manual
  token rotation needed as long as the PAT is valid.
- Stop/start the pod from the RunPod UI; the runner re-registers automatically.
