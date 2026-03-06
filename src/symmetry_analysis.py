"""
Author: Yu Huize (hayes_yu@163.com)
Date: 2026-02-11
Course: NUS ME5311 Project 1
"""

"""
symmetry_analysis.py — Symmetry & Anisotropy Diagnostics
=======================================================
- 2D Fourier spectrum slices along kx / ky axes comparison
- Spectral mirror symmetry test
- SVD mode spatial symmetry test
- ux / uy component radial spectrum difference quantification
"""

import numpy as np
from . import data_loader as dl
from . import visualization as viz


# ── 1. Fourier spectrum anisotropy ─────────────────────────────
def axis_slices(psd_2d: np.ndarray):
    """
    Slice along kx axis (ky=0) and ky axis (kx=0).
    Returns (k_1d, psd_kx_slice, psd_ky_slice)
    psd_2d should be in original layout before fftshift (ny, nx).
    """
    ny, nx = psd_2d.shape
    psd_kx = psd_2d[0, :]          # ky=0 row → along kx
    psd_ky = psd_2d[:, 0]          # kx=0 column → along ky
    k_1d = np.arange(nx // 2 + 1)  # non-negative wavenumbers
    # FFT output: first half is [0..N/2], second half is negative frequency symmetric
    psd_kx = psd_kx[:nx // 2 + 1]
    psd_ky = psd_ky[:ny // 2 + 1]
    return k_1d, psd_kx, psd_ky


def anisotropy_ratio(psd_kx: np.ndarray, psd_ky: np.ndarray):
    """
    Compute anisotropy ratio  R(k) = PSD_kx(k) / PSD_ky(k).
    R ≈ 1 → isotropic; deviation from 1 → anisotropic.
    """
    eps = 1e-30
    ratio = psd_kx / (psd_ky + eps)
    return ratio


# ── 2. Mirror symmetry test ─────────────────────────────────
def mirror_symmetry(psd_2d: np.ndarray):
    """
    Test mirror symmetry of 2D PSD:
      x-symmetry: PSD(kx, ky) vs PSD(-kx, ky)
      y-symmetry: PSD(kx, ky) vs PSD(kx, -ky)
    Returns relative errors (err_x, err_y); smaller values indicate better symmetry.
    """
    # PSD(-kx, ky) = np.flip(PSD, axis=1) (FFT symmetry)
    flip_x = np.flip(psd_2d, axis=1)
    flip_y = np.flip(psd_2d, axis=0)

    norm = np.sum(psd_2d ** 2)
    err_x = np.sum((psd_2d - flip_x) ** 2) / norm
    err_y = np.sum((psd_2d - flip_y) ** 2) / norm
    print(f"[symmetry] Mirror-x relative error: {err_x:.6e}")
    print(f"[symmetry] Mirror-y relative error: {err_y:.6e}")
    return err_x, err_y


def rotational_symmetry_90(psd_2d: np.ndarray):
    """
    Test 90° rotational symmetry: PSD(kx,ky) vs PSD(ky,kx).
    Returns relative error; smaller values indicate better rotational symmetry (necessary condition for isotropy).
    """
    rotated = psd_2d.T
    norm = np.sum(psd_2d ** 2)
    err = np.sum((psd_2d - rotated) ** 2) / norm
    print(f"[symmetry] 90°-rotation relative error: {err:.6e}")
    return err


# ── 3. SVD mode symmetry ───────────────────────────────────
def mode_symmetry_check(mode_2d: np.ndarray):
    """
    Check a single spatial mode (ny, nx) for:
      - x-direction mirror symmetry  mode(y, x) vs mode(y, nx-1-x)
      - y-direction mirror symmetry  mode(y, x) vs mode(ny-1-y, x)
    Returns (corr_x, corr_y)  correlation coefficients: +1=symmetric, -1=antisymmetric, 0=uncorrelated.
    """
    flip_x = np.flip(mode_2d, axis=1)
    flip_y = np.flip(mode_2d, axis=0)

    def _corr(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.corrcoef(a_flat, b_flat)[0, 1]

    corr_x = _corr(mode_2d, flip_x)
    corr_y = _corr(mode_2d, flip_y)
    return corr_x, corr_y


# ── 4. Component difference quantification ─────────────────────
def component_energy_ratio(data: np.ndarray):
    """
    Compute global energy ratio of ux and uy.
    Returns (E_ux, E_uy, ratio = E_ux/E_uy)
    """
    E_ux = np.mean(data[..., 0] ** 2)
    E_uy = np.mean(data[..., 1] ** 2)
    ratio = E_ux / E_uy
    print(f"[symmetry] Energy: E_ux={E_ux:.4e}, E_uy={E_uy:.4e}, ratio={ratio:.4f}")
    return E_ux, E_uy, ratio


# ── Top-level run function ──────────────────────────────────
def run(data: np.ndarray, psd_total_2d: np.ndarray,
        svd_modes: list | None = None):
    """
    Full symmetry/anisotropy diagnostic pipeline.
    data         : (nt, ny, nx, 2)  raw or fluctuation field
    psd_total_2d : (ny, nx)  total energy 2D PSD (from spectral_analysis)
    svd_modes    : [(ux_2d, uy_2d), ...] from svd_analysis
    """
    results = {}

    # Component energy ratio
    E_ux, E_uy, ratio = component_energy_ratio(data)
    results["energy_ratio"] = ratio

    # Fourier spectrum symmetry
    err_x, err_y = mirror_symmetry(psd_total_2d)
    err_rot = rotational_symmetry_90(psd_total_2d)
    results["mirror_err_x"] = err_x
    results["mirror_err_y"] = err_y
    results["rotation_err"] = err_rot

    # kx / ky axis slice anisotropy
    k_1d, psd_kx, psd_ky = axis_slices(psd_total_2d)
    viz.plot_anisotropy_comparison(psd_kx, psd_ky, k_1d)
    results["anisotropy_ratio"] = anisotropy_ratio(psd_kx, psd_ky)

    # SVD mode symmetry
    if svd_modes is not None:
        sym_results = []
        for i, (ux_m, uy_m) in enumerate(svd_modes):
            cx_ux, cy_ux = mode_symmetry_check(ux_m)
            cx_uy, cy_uy = mode_symmetry_check(uy_m)
            sym_results.append(dict(
                mode=i + 1,
                ux_corr_x=cx_ux, ux_corr_y=cy_ux,
                uy_corr_x=cx_uy, uy_corr_y=cy_uy,
            ))
            print(f"[symmetry] Mode {i+1}: "
                  f"ux(mirror-x={cx_ux:+.3f}, mirror-y={cy_ux:+.3f}), "
                  f"uy(mirror-x={cx_uy:+.3f}, mirror-y={cy_uy:+.3f})")
        results["mode_symmetry"] = sym_results

    return results
