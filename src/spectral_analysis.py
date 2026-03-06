"""
Author: Yu Huize (hayes_yu@163.com)
Date: 2026-02-11
Course: NUS ME5311 Project 1
"""

"""
spectral_analysis.py — Fourier / Power Spectral Analysis
========================================================
- 2D spatial FFT + time-averaged PSD
- Radial power spectrum (1D reduction)
- Temporal frequency PSD (spatially averaged)
- Peak wavenumber detection (external forcing inference)
"""

import numpy as np
from . import data_loader as dl
from . import visualization as viz


# ── Spatial spectral analysis ──────────────────────────────────
def spatial_psd_2d(data: np.ndarray, component: int = 0) -> np.ndarray:
    """
    Compute 2D spatial FFT for all time snapshots, then time-averaged power spectrum.
    data: (nt, ny, nx, 2)
    component: 0=ux, 1=uy
    Returns: psd_2d (ny, nx)
    """
    field = data[..., component]            # (nt, ny, nx)
    fhat = np.fft.fft2(field, axes=(1, 2))  # (nt, ny, nx)
    psd = np.mean(np.abs(fhat) ** 2, axis=0) / (dl.NX * dl.NY)
    return psd


def radial_spectrum(psd_2d: np.ndarray):
    """
    Convert 2D PSD to radial (1D) power spectrum.
    Returns (k_bins, psd_radial).
    """
    ny, nx = psd_2d.shape
    # Wavenumber grid
    kx = np.fft.fftfreq(nx, d=1.0) * nx   # integer wavenumbers
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_max = int(np.floor(K.max()))
    k_bins = np.arange(0, k_max + 1)
    psd_radial = np.zeros(len(k_bins))
    for i, k in enumerate(k_bins):
        mask = (K >= k - 0.5) & (K < k + 0.5)
        if mask.any():
            psd_radial[i] = psd_2d[mask].sum()
    return k_bins, psd_radial


def detect_peak_wavenumbers(k_bins: np.ndarray, psd_radial: np.ndarray,
                            n_peaks: int = 5):
    """
    Find the most prominent n_peaks wavenumber peaks in the radial spectrum (excluding k=0 DC component).
    Returns [(k, psd_value), ...] sorted by energy in descending order.
    """
    psd_copy = psd_radial.copy()
    psd_copy[0] = 0  # Exclude DC
    idx = np.argsort(psd_copy)[::-1][:n_peaks]
    peaks = [(k_bins[i], psd_radial[i]) for i in idx]
    print("[spectral] Peak wavenumbers:")
    for k, p in peaks:
        print(f"    k = {k:.0f},  PSD = {p:.4e}")
    return peaks


# ── Temporal spectral analysis ─────────────────────────────────
def temporal_psd_avg(data: np.ndarray, component: int = 0,
                     dt: float = dl.DT):
    """
    Compute temporal FFT at each grid point, then spatially average to get spatially-averaged temporal PSD.
    data: (nt, ny, nx, 2)
    Returns (freqs, psd_avg)
    """
    field = data[..., component]             # (nt, ny, nx)
    nt = field.shape[0]
    fhat = np.fft.rfft(field, axis=0)        # (nt//2+1, ny, nx)
    psd = np.mean(np.abs(fhat) ** 2, axis=(1, 2)) / nt
    freqs = np.fft.rfftfreq(nt, d=dt)
    return freqs, psd


def detect_peak_frequencies(freqs: np.ndarray, psd: np.ndarray,
                            n_peaks: int = 5):
    """Find the most prominent n_peaks frequency peaks in the temporal PSD (excluding f=0)."""
    psd_copy = psd.copy()
    psd_copy[0] = 0
    idx = np.argsort(psd_copy)[::-1][:n_peaks]
    peaks = [(freqs[i], psd[i]) for i in idx]
    print("[spectral] Peak frequencies:")
    for f, p in peaks:
        print(f"    f = {f:.6f},  PSD = {p:.4e}")
    return peaks


# ── Component comparison ─────────────────────────────────────
def compare_components_spatial(data: np.ndarray):
    """
    Compute radial spectra for ux and uy separately, for anisotropy diagnostics.
    Returns dict(k_bins, psd_ux, psd_uy)
    """
    psd_ux = spatial_psd_2d(data, component=0)
    psd_uy = spatial_psd_2d(data, component=1)
    k_bins, rad_ux = radial_spectrum(psd_ux)
    _,      rad_uy = radial_spectrum(psd_uy)
    return dict(k_bins=k_bins, psd_ux_radial=rad_ux, psd_uy_radial=rad_uy,
                psd_ux_2d=psd_ux, psd_uy_2d=psd_uy)


# ── Top-level run function ──────────────────────────────────
def run(data: np.ndarray, dt: float = dl.DT):
    """
    Full spatial + temporal spectral analysis pipeline.
    data: (nt, ny, nx, 2)
    """
    results = {}

    # ---- Spatial spectrum: ux ----
    psd_ux_2d = spatial_psd_2d(data, component=0)
    k_bins, rad_ux = radial_spectrum(psd_ux_2d)
    viz.plot_2d_spectrum(psd_ux_2d, title="2D PSD — $u_x$",
                         save_name="spectral_2d_ux")
    viz.plot_radial_spectrum(k_bins, rad_ux,
                            title="Radial PSD — $u_x$",
                            save_name="spectral_radial_ux")

    # ---- Spatial spectrum: uy ----
    psd_uy_2d = spatial_psd_2d(data, component=1)
    _, rad_uy = radial_spectrum(psd_uy_2d)
    viz.plot_2d_spectrum(psd_uy_2d, title="2D PSD — $u_y$",
                         save_name="spectral_2d_uy")
    viz.plot_radial_spectrum(k_bins, rad_uy,
                            title="Radial PSD — $u_y$",
                            save_name="spectral_radial_uy")

    # ---- Combined radial spectrum ----
    psd_total_2d = psd_ux_2d + psd_uy_2d
    _, rad_total = radial_spectrum(psd_total_2d)
    viz.plot_radial_spectrum(k_bins, rad_total,
                            title="Radial PSD — total energy",
                            save_name="spectral_radial_total")

    # ---- Peak detection ----
    peaks_spatial = detect_peak_wavenumbers(k_bins, rad_total)

    # ---- Temporal PSD ----
    freqs_ux, tpsd_ux = temporal_psd_avg(data, component=0, dt=dt)
    freqs_uy, tpsd_uy = temporal_psd_avg(data, component=1, dt=dt)
    viz.plot_temporal_psd(freqs_ux, tpsd_ux,
                         title="Temporal PSD (spatially averaged) — $u_x$",
                         save_name="spectral_temporal_ux")
    viz.plot_temporal_psd(freqs_uy, tpsd_uy,
                         title="Temporal PSD (spatially averaged) — $u_y$",
                         save_name="spectral_temporal_uy")
    peaks_temporal = detect_peak_frequencies(freqs_ux, tpsd_ux)

    results.update(
        psd_ux_2d=psd_ux_2d, psd_uy_2d=psd_uy_2d,
        psd_total_2d=psd_total_2d,
        k_bins=k_bins,
        rad_ux=rad_ux, rad_uy=rad_uy, rad_total=rad_total,
        peaks_spatial=peaks_spatial,
        freqs_ux=freqs_ux, tpsd_ux=tpsd_ux,
        freqs_uy=freqs_uy, tpsd_uy=tpsd_uy,
        peaks_temporal=peaks_temporal,
    )
    return results
