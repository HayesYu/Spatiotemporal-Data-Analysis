"""
Author: Yu Huize (hayes_yu@163.com)
Date: 2026-02-11
Course: NUS ME5311 Project 1
"""

"""
svd_analysis.py — SVD / PCA Modal Analysis
==========================================
- Economy SVD decomposition
- Singular value spectrum & cumulative energy
- Spatial mode extraction and visualization
- Temporal coefficient extraction + temporal spectral analysis
"""

import numpy as np
from . import data_loader as dl
from . import visualization as viz


def perform_svd(data_matrix: np.ndarray, full_matrices: bool = False):
    """
    Perform economy SVD on data matrix (N, T).
    Returns U (N, K), sigma (K,), Vt (K, T)  where K = min(N, T)
    """
    print("[svd] Computing economy SVD …")
    U, sigma, Vt = np.linalg.svd(data_matrix, full_matrices=full_matrices)
    print(f"[svd] Done. U={U.shape}, sigma={sigma.shape}, Vt={Vt.shape}")
    return U, sigma, Vt


def energy_spectrum(sigma: np.ndarray):
    """
    Compute modal energy spectrum.
    Returns:
        energy       : σ_k²
        cum_energy   : cumulative energy fraction (0~1)
        n_95, n_99   : number of modes for 95% / 99% cumulative energy
    """
    energy = sigma ** 2
    cum = np.cumsum(energy) / energy.sum()
    n_95 = int(np.searchsorted(cum, 0.95)) + 1
    n_99 = int(np.searchsorted(cum, 0.99)) + 1
    print(f"[svd] Modes for 95% energy: {n_95},  99%: {n_99}")
    return energy, cum, n_95, n_99


def extract_spatial_modes(U: np.ndarray, ny: int = dl.NY, nx: int = dl.NX,
                          n_modes: int = 6):
    """
    Extract the first n_modes spatial modes, reshaped to (ny, nx) components.
    Returns a list [(ux_mode, uy_mode), ...]
    """
    half = ny * nx
    modes = []
    for i in range(n_modes):
        ux = U[:half, i].reshape(ny, nx)
        uy = U[half:, i].reshape(ny, nx)
        modes.append((ux, uy))
    return modes


def temporal_coefficients(sigma: np.ndarray, Vt: np.ndarray,
                          n_modes: int = 6):
    """Return temporal coefficients a_k(t) = σ_k * v_k(t) for the first n_modes."""
    return [sigma[i] * Vt[i, :] for i in range(n_modes)]


def temporal_coefficient_psd(sigma: np.ndarray, Vt: np.ndarray,
                             dt: float = dl.DT, n_modes: int = 6):
    """
    Compute FFT of the first n_modes SVD temporal coefficients.
    Returns (freqs, [psd_1, …, psd_K]).
    """
    nt = Vt.shape[1]
    freqs = np.fft.rfftfreq(nt, d=dt)
    psds = []
    for i in range(n_modes):
        coeff = sigma[i] * Vt[i, :]
        fhat = np.fft.rfft(coeff)
        psd = (np.abs(fhat) ** 2) / nt
        psds.append(psd)
    return freqs, psds


# ── Top-level run function ──────────────────────────────────
def run(data_matrix: np.ndarray, dt: float = dl.DT,
        ny: int = dl.NY, nx: int = dl.NX, n_modes: int = 6):
    """
    Full SVD analysis pipeline: decomposition → energy spectrum → spatial mode visualization
    → temporal coefficient visualization → temporal coefficient PSD.
    """
    U, sigma, Vt = perform_svd(data_matrix)

    # Energy spectrum
    energy, cum, n95, n99 = energy_spectrum(sigma)
    viz.plot_singular_values(sigma, n_show=min(100, len(sigma)))

    # Spatial modes
    viz.plot_spatial_modes(U, ny, nx, n_modes=n_modes)

    # Temporal coefficients
    viz.plot_temporal_coefficients(Vt, sigma, dt, n_modes=n_modes)

    # Temporal coefficient PSD
    freqs, psds = temporal_coefficient_psd(sigma, Vt, dt, n_modes=n_modes)
    viz.plot_mode_temporal_psd(freqs, psds, n_modes=n_modes)

    return dict(U=U, sigma=sigma, Vt=Vt,
                energy=energy, cum_energy=cum,
                n95=n95, n99=n99, freqs=freqs, psds=psds)
