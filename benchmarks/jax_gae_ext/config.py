"""Shared constants for the three-phase JAX-vs-Triton GAE benchmark. Kept
free of torch/jax/triton imports so it loads identically in either venv --
see benchmarks/benchmark_gae_vs_jax_scan_triton.py's module docstring for
why this benchmark is split into phases across two virtualenvs in the first
place (torch's exact `nvidia-cudnn-cu12==9.1.0.70` pin is incompatible with
the cuDNN version jax[cuda12]'s GPU plugin requires -- confirmed on a real
pod, not theoretical: XLA's GPU compiler hard-fails with a null cuDNN handle
if the two share one environment)."""

GAMMA = 0.99
LAMBDA = 0.95
TERM_PROB = 0.05
SEED = 0

SEQ_LENS = [128, 512, 1024, 2048, 4096]
NUM_ENVS_LIST = [128, 512, 2048, 8192]

# Massively-parallel-sim regime: short horizon, high env count (Isaac Gym-style).
SEQ_LENS_SHORT = [8, 16, 32, 64, 128]
NUM_ENVS_LIST_LARGE = [4096, 8192, 16384, 32768]

N_ITER = 100
N_TRIALS = 11
N_AMORTIZED_CALLS = 100
N_PROFILE_ITER = 20

H100_PEAK_GBPS = 3350.0  # HBM3 datasheet peak, H100 SXM5 80GB

# Equivalence-gate shapes: (num_envs, seq_len, seed).
EQUIVALENCE_CASES = [
    (64, 256, 1), (32, 4096, 2), (37, 129, 3),
    # Massively-parallel-sim regime: short T, high num_envs.
    (4096, 8, 4), (8192, 16, 5), (16384, 64, 6), (2048, 128, 7),
]

# Categorical palette (fixed hue order), light-mode -- from the dataviz skill's
# validated default palette. Color = num_envs (identity); linestyle = impl.
COLORS = {128: "#2a78d6", 512: "#eb6834", 2048: "#1baf7a", 8192: "#eda100"}
COLORS_SHORT = {8: "#2a78d6", 16: "#eb6834", 32: "#1baf7a", 64: "#eda100", 128: "#e87ba4"}


def all_sweep_cells():
    """(num_envs, seq_len, regime_label) for every cell across both regimes,
    in the fixed order both the triton and jax phases iterate them -- keeping
    this in one place guarantees the two phases can never drift out of sync
    on which cells exist or what order they're produced in."""
    cells = []
    for num_envs in NUM_ENVS_LIST:
        for seq_len in SEQ_LENS:
            cells.append((num_envs, seq_len, "production"))
    for num_envs in NUM_ENVS_LIST_LARGE:
        for seq_len in SEQ_LENS_SHORT:
            cells.append((num_envs, seq_len, "short_horizon"))
    return cells


def cell_key(num_envs, seq_len):
    return f"n{num_envs}_t{seq_len}"
