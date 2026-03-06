"""
Author: Yu Huize (hayes_yu@163.com)
Date: 2026-02-11
Course: NUS ME5311 Project 1
"""

"""
main.py — ME5311 Project 1 Main Analysis Script
=================================================
Runs all analysis modules in sequence through the complete analysis pipeline.
All figures are automatically saved to the figures/ directory.

Usage:
    cd <project_root>
    python main.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from src import data_loader as dl
from src import svd_analysis as svd
from src import spectral_analysis as spectral
from src import symmetry_analysis as symmetry
from src import visualization as viz


def main():
    print("=" * 60)
    print("  ME5311 Project 1 — Spatio-temporal Data Analysis")
    print("=" * 60)

    # ── Step 0: Data loading & preprocessing ─────────────────────────
    print("\n▶ Step 0: Loading & preprocessing …")
    bundle = dl.load_and_preprocess()
    raw         = bundle["raw"]           # (15000, 64, 64, 2)
    mean_field  = bundle["mean_field"]    # (64, 64, 2)
    fluctuation = bundle["fluctuation"]   # (15000, 64, 64, 2)
    data_matrix = bundle["data_matrix"]   # (8192, 15000)
    vorticity   = bundle["vorticity"]     # (15000, 64, 64)
    divergence  = bundle["divergence"]    # (15000, 64, 64)

    # Visualization: mean field, example snapshot, vorticity/divergence snapshots
    viz.plot_vector_snapshot(mean_field, title="Time-averaged mean field",
                            save_name="step0_mean_field")
    viz.plot_vector_snapshot(raw[0], title="Snapshot t=0",
                            save_name="step0_snapshot_t0")
    viz.plot_scalar_field(vorticity[0], title="Vorticity $\\omega$ at t=0",
                          save_name="step0_vorticity_t0")
    viz.plot_scalar_field(divergence[0], title="Divergence $\\nabla\\cdot u$ at t=0",
                          save_name="step0_divergence_t0")

    # ── Step 1: SVD / PCA analysis ─────────────────────────────
    print("\n▶ Step 1: SVD analysis on fluctuation field …")
    svd_results = svd.run(data_matrix, dt=dl.DT, ny=dl.NY, nx=dl.NX,
                          n_modes=6)

    # ── Step 2: Spatial + temporal spectral analysis ────────────────────
    print("\n▶ Step 2: Spectral analysis …")
    # Run spectral analysis on fluctuation field
    spec_results = spectral.run(fluctuation, dt=dl.DT)

    # Supplementary: vorticity field spatial spectrum
    print("\n  [extra] Vorticity spatial spectrum …")
    psd_vor_2d = np.mean(np.abs(np.fft.fft2(vorticity, axes=(1, 2))) ** 2,
                         axis=0) / (dl.NX * dl.NY)
    k_bins, rad_vor = spectral.radial_spectrum(psd_vor_2d)
    viz.plot_radial_spectrum(k_bins, rad_vor,
                            title="Radial PSD — vorticity",
                            save_name="spectral_radial_vorticity")
    spectral.detect_peak_wavenumbers(k_bins, rad_vor)

    # ── Step 3: Symmetry & anisotropy diagnostics ────────────────────
    print("\n▶ Step 3: Symmetry & anisotropy diagnostics …")
    spatial_modes = svd.extract_spatial_modes(
        svd_results["U"], n_modes=6)
    sym_results = symmetry.run(
        fluctuation, spec_results["psd_total_2d"], svd_modes=spatial_modes)

    # Supplementary: ux vs uy component radial spectrum comparison
    comp = spectral.compare_components_spatial(fluctuation)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(comp["k_bins"], comp["psd_ux_radial"], "o-", ms=3,
                label="$u_x$")
    ax.semilogy(comp["k_bins"], comp["psd_uy_radial"], "s-", ms=3,
                label="$u_y$")
    ax.set_xlabel("Radial wavenumber $k$")
    ax.set_ylabel("PSD($k$)")
    ax.set_title("Component comparison: $u_x$ vs $u_y$ radial spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    viz.savefig(fig, "spectral_component_comparison")

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Analysis complete. Figures saved to:", viz.FIG_DIR)
    print("=" * 60)

    # Return all results (for further exploration in interactive environments)
    return dict(bundle=bundle, svd=svd_results,
                spectral=spec_results, symmetry=sym_results)


if __name__ == "__main__":
    results = main()
