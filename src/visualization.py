"""
Author: Yu Huize (hayes_yu@163.com)
Date: 2026-02-11
Course: NUS ME5311 Project 1
"""

"""
visualization.py — Unified Visualization Utilities
==================================================
Provides plotting functions for all analysis modules with
consistent color schemes, annotations, and export settings.
All figures are saved to the figures/ directory by default.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Global configuration ───────────────────────────────────────
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "image.cmap":       "RdBu_r",
    "figure.figsize":   (8, 6),
})


def savefig(fig, name: str, fmt: str = "png"):
    """Save figure to the figures/ directory."""
    path = FIG_DIR / f"{name}.{fmt}"
    fig.savefig(path)
    print(f"[viz] Saved → {path}")
    plt.close(fig)


# ── 1. Vector field snapshots ─────────────────────────────────
def plot_vector_snapshot(field_2d: np.ndarray, title: str = "Vector field",
                         save_name: str | None = None, step: int = 2):
    """
    Plot a single 2D vector field snapshot.
    field_2d: (ny, nx, 2)
    step: arrow spacing (downsampling to avoid overcrowding)
    """
    ny, nx, _ = field_2d.shape
    ux, uy = field_2d[..., 0], field_2d[..., 1]
    mag = np.sqrt(ux**2 + uy**2)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.pcolormesh(mag, cmap="viridis", shading="auto")
    Y, X = np.mgrid[0:ny, 0:nx]
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              ux[::step, ::step], uy[::step, ::step],
              color="k", scale=None, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Magnitude")
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_scalar_field(field_2d: np.ndarray, title: str = "",
                      cmap: str = "RdBu_r", save_name: str | None = None,
                      symmetric: bool = True):
    """Plot a scalar field (ny, nx) with optional symmetric colorbar."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    vmax = np.abs(field_2d).max() if symmetric else None
    vmin = -vmax if symmetric else None
    im = ax.imshow(field_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


# ── 2. SVD related ─────────────────────────────────────────
def plot_singular_values(sigma: np.ndarray, n_show: int = 100,
                         save_name: str = "svd_singular_values"):
    """Singular value decay curve + cumulative energy fraction."""
    energy = sigma**2
    cum_energy = np.cumsum(energy) / energy.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: singular values (log scale)
    ax = axes[0]
    ax.semilogy(np.arange(1, n_show + 1), sigma[:n_show], "o-", ms=3)
    ax.set_xlabel("Mode index $k$")
    ax.set_ylabel(r"Singular value $\sigma_k$")
    ax.set_title("Singular value spectrum")
    ax.grid(True, alpha=0.3)

    # Right panel: cumulative energy
    ax = axes[1]
    ax.plot(np.arange(1, n_show + 1), cum_energy[:n_show] * 100, "s-", ms=3)
    ax.axhline(95, color="r", ls="--", lw=1, label="95%")
    ax.axhline(99, color="orange", ls="--", lw=1, label="99%")
    ax.set_xlabel("Number of modes $K$")
    ax.set_ylabel("Cumulative energy (%)")
    ax.set_title("Cumulative energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, save_name)
    return fig


def plot_spatial_modes(U: np.ndarray, ny: int, nx: int,
                       n_modes: int = 6,
                       save_name: str = "svd_spatial_modes"):
    """
    Visualize the first n_modes SVD spatial modes.
    U: (N, K)  N = 2*ny*nx, first N//2 is ux, last N//2 is uy.
    """
    half = ny * nx
    fig, axes = plt.subplots(n_modes, 2, figsize=(10, 3 * n_modes))
    for i in range(n_modes):
        mode = U[:, i]
        ux_mode = mode[:half].reshape(ny, nx)
        uy_mode = mode[half:].reshape(ny, nx)
        for j, (comp, label) in enumerate([(ux_mode, "$u_x$"), (uy_mode, "$u_y$")]):
            ax = axes[i, j]
            vmax = np.abs(comp).max()
            im = ax.imshow(comp, origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_title(f"Mode {i+1} — {label}")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(fig, save_name)
    return fig


def plot_temporal_coefficients(Vt: np.ndarray, sigma: np.ndarray,
                                dt: float, n_modes: int = 6,
                                save_name: str = "svd_temporal_coeff"):
    """Plot temporal coefficients σ_k * v_k(t) for the first n_modes."""
    nt = Vt.shape[1]
    t = np.arange(nt) * dt

    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 2.5 * n_modes),
                             sharex=True)
    for i in range(n_modes):
        ax = axes[i]
        coeff = sigma[i] * Vt[i, :]
        ax.plot(t, coeff, lw=0.5)
        ax.set_ylabel(f"Mode {i+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    axes[0].set_title("Temporal coefficients $\\sigma_k v_k(t)$")
    fig.tight_layout()
    savefig(fig, save_name)
    return fig


# ── 3. Spectral analysis related ───────────────────────────────
def plot_2d_spectrum(psd_2d: np.ndarray, title: str = "2D Power Spectrum",
                     save_name: str | None = None, log: bool = True):
    """Plot 2D power spectrum (wavenumber domain), centered display."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    display = np.log10(psd_2d + 1e-30) if log else psd_2d
    ny, nx = psd_2d.shape
    extent = [-nx // 2, nx // 2, -ny // 2, ny // 2]
    im = ax.imshow(np.fft.fftshift(display), origin="lower",
                   cmap="inferno", extent=extent)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log₁₀(PSD)" if log else "PSD")
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_radial_spectrum(k_bins: np.ndarray, psd_radial: np.ndarray,
                         title: str = "Radial Power Spectrum",
                         save_name: str | None = None):
    """Plot radial (1D) power spectrum."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(k_bins, psd_radial, "o-", ms=3)
    ax.set_xlabel("Radial wavenumber $k$")
    ax.set_ylabel("PSD($k$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_temporal_psd(freqs: np.ndarray, psd: np.ndarray,
                      title: str = "Temporal PSD (spatially averaged)",
                      save_name: str | None = None):
    """Temporal frequency power spectrum."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(freqs, psd, lw=0.8)
    ax.set_xlabel("Frequency $f$")
    ax.set_ylabel("PSD")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_name:
        savefig(fig, save_name)
    return fig, ax


def plot_mode_temporal_psd(freqs: np.ndarray, psds: list[np.ndarray],
                           n_modes: int = 6,
                           save_name: str = "svd_mode_temporal_psd"):
    """PSD of the first K SVD modal temporal coefficients."""
    fig, axes = plt.subplots(n_modes, 1, figsize=(10, 2.5 * n_modes),
                             sharex=True)
    for i in range(n_modes):
        ax = axes[i]
        ax.semilogy(freqs, psds[i], lw=0.8)
        ax.set_ylabel(f"Mode {i+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frequency $f$")
    axes[0].set_title("PSD of SVD temporal coefficients")
    fig.tight_layout()
    savefig(fig, save_name)
    return fig


# ── 4. Symmetry / Anisotropy ────────────────────────────────
def plot_anisotropy_comparison(psd_kx: np.ndarray, psd_ky: np.ndarray,
                                k_1d: np.ndarray,
                                save_name: str = "anisotropy_kx_ky"):
    """Compare kx-direction vs ky-direction 1D spectral slices."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(k_1d, psd_kx, "o-", ms=3, label="PSD along $k_x$ ($k_y=0$)")
    ax.semilogy(k_1d, psd_ky, "s-", ms=3, label="PSD along $k_y$ ($k_x=0$)")
    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("PSD")
    ax.set_title("Anisotropy diagnostic")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, save_name)
    return fig, ax
