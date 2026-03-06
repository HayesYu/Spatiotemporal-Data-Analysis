"""
Author: Yu Huize (hayes_yu@163.com)
Date: 2026-02-11
Course: NUS ME5311 Project 1
"""

"""
data_loader.py — Data Loading & Preprocessing
==============================================
- Load vector_64.npy  →  shape (nt, ny, nx, 2)
- Compute time-averaged mean field  mean_field  →  (ny, nx, 2)
- Compute fluctuation field          fluctuation →  (nt, ny, nx, 2)
- Build data matrix                  data_matrix →  (N, T)  N=8192, T=15000
- Compute derived quantities: vorticity ω, divergence ∇·u (periodic boundary finite differences)
"""

from pathlib import Path
import numpy as np

# ── Dataset parameters ──────────────────────────────────────
NX, NY = 64, 64              # Spatial grid resolution
NT     = 15000               # Number of time snapshots
DT     = 0.2                 # Time sampling interval (simulation time units)
N_DOF  = NX * NY * 2         # Degrees of freedom per snapshot  8192
T_TOTAL = NT * DT            # Total duration 3000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw(fname: str = "vector_64.npy") -> np.ndarray:
    """Load raw data → (nt, ny, nx, 2)"""
    data = np.load(DATA_DIR / fname)
    assert data.shape == (NT, NY, NX, 2), f"Unexpected shape {data.shape}"
    print(f"[data_loader] Loaded {fname}  shape={data.shape}  "
          f"dtype={data.dtype}  size={data.nbytes / 1e9:.2f} GB")
    return data


def compute_mean_field(data: np.ndarray) -> np.ndarray:
    """Time-averaged mean field ū(x,y) → (ny, nx, 2)"""
    return data.mean(axis=0)


def compute_fluctuation(data: np.ndarray,
                        mean_field: np.ndarray | None = None) -> np.ndarray:
    """Fluctuation field u' = u - ū → (nt, ny, nx, 2)"""
    if mean_field is None:
        mean_field = compute_mean_field(data)
    return data - mean_field[np.newaxis, ...]


def build_data_matrix(field: np.ndarray) -> np.ndarray:
    """
    Flatten 4-D field (nt, ny, nx, 2) into 2-D data matrix (N, T).
    Each column = one time snapshot as an 8192-dimensional vector.
    Flattening order: all ux (ny×nx) followed by all uy (ny×nx).
    """
    nt = field.shape[0]
    # (nt, ny, nx, 2) → (nt, 2, ny, nx) → (nt, 2*ny*nx)
    mat = field.transpose(0, 3, 1, 2).reshape(nt, -1)  # (T, N)
    return mat.T  # (N, T)


# ── Derived quantities (periodic boundary finite differences) ──
def _periodic_diff(arr: np.ndarray, axis: int, dx: float = 1.0) -> np.ndarray:
    """Central difference (periodic boundary) along specified axis."""
    return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2.0 * dx)


def compute_vorticity(data: np.ndarray, L: float | None = None) -> np.ndarray:
    """
    Vorticity  ω = ∂u_y/∂x - ∂u_x/∂y  → (nt, ny, nx)
    data: (nt, ny, nx, 2)  component order [ux, uy]
    L: domain size for computing dx = L/NX (defaults to dx=1 if not given)
    """
    dx = (L / NX) if L is not None else 1.0
    dy = (L / NY) if L is not None else 1.0
    ux = data[..., 0]  # (nt, ny, nx)
    uy = data[..., 1]
    # ∂u_y/∂x  — x-direction corresponds to axis=2
    duy_dx = _periodic_diff(uy, axis=2, dx=dx)
    # ∂u_x/∂y  — y-direction corresponds to axis=1
    dux_dy = _periodic_diff(ux, axis=1, dx=dy)
    return duy_dx - dux_dy


def compute_divergence(data: np.ndarray, L: float | None = None) -> np.ndarray:
    """
    Divergence  ∇·u = ∂u_x/∂x + ∂u_y/∂y  → (nt, ny, nx)
    """
    dx = (L / NX) if L is not None else 1.0
    dy = (L / NY) if L is not None else 1.0
    ux = data[..., 0]
    uy = data[..., 1]
    dux_dx = _periodic_diff(ux, axis=2, dx=dx)
    duy_dy = _periodic_diff(uy, axis=1, dx=dy)
    return dux_dx + duy_dy


# ── Convenience entry point ──────────────────────────────────
def load_and_preprocess(fname: str = "vector_64.npy"):
    """
    One-step loading + mean/fluctuation separation + matrix construction.
    Returns dict:
        raw        : (nt, ny, nx, 2)
        mean_field : (ny, nx, 2)
        fluctuation: (nt, ny, nx, 2)
        data_matrix: (N, T)   based on fluctuation field
        vorticity  : (nt, ny, nx)
        divergence : (nt, ny, nx)
    """
    raw = load_raw(fname)
    mf  = compute_mean_field(raw)
    flu = compute_fluctuation(raw, mf)
    mat = build_data_matrix(flu)
    vor = compute_vorticity(raw)
    div = compute_divergence(raw)
    print(f"[data_loader] data_matrix shape = {mat.shape}")
    print(f"[data_loader] vorticity   shape = {vor.shape}")
    print(f"[data_loader] divergence  shape = {div.shape}")
    return dict(
        raw=raw, mean_field=mf, fluctuation=flu,
        data_matrix=mat, vorticity=vor, divergence=div,
    )
