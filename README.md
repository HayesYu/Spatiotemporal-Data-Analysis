# ME5311 Project 1 — Spatio‑temporal Data Analysis

**Author**: Yu Huize

This repository contains the analysis code for **NUS ME5311** course **Project 1**.

---

## Overview

Data‑driven analysis of a large spatio‑temporal vector field dataset using **SVD/PCA**, **Fourier spectral analysis**, and **symmetry/anisotropy diagnostics**. The goal is to extract dominant spatial structures, energy distributions, and spatio‑temporal features directly from the data. No system dynamics modeling or prediction is performed (that is the scope of Project 2).

## Dataset

| Property | Value |
|----------|-------|
| Shape | `(15000, 64, 64, 2)` |
| Description | Two‑component vector field on a periodic 2D domain |
| Time step | `Δt = 0.2` |
| Total duration | `3000` simulation time units |
| File | `data/vector_64.npy` |

## Project Structure

```
Spatiotemporal-Data-Analysis/
├── data/                    # dataset (gitignored)
├── docs/                    # project documents (gitignored)
├── figures/                 # output figures (gitignored)
├── notebooks/               # Jupyter notebooks
├── src/                     # analysis modules
│   ├── data_loader.py       #   data loading & preprocessing
│   ├── svd_analysis.py      #   SVD/PCA modal analysis
│   ├── spectral_analysis.py #   Fourier / power spectral analysis
│   ├── symmetry_analysis.py #   symmetry & anisotropy diagnostics
│   └── visualization.py     #   unified visualization utilities
├── main.py                  # one‑click analysis pipeline
├── pyproject.toml           # project dependencies
├── .gitignore
└── README.md
```

## Features

- **Mean‑field / fluctuation separation**: remove temporal mean to highlight dynamics
- **SVD analysis**: singular value energy spectrum, dominant spatial modes, temporal coefficient PSD
- **Spatial spectral analysis**: 2D FFT power spectrum, radial spectrum, peak wavenumber detection
- **Temporal spectral analysis**: temporal PSD (spatially averaged), peak frequency detection
- **Symmetry diagnostics**: mirror/rotational symmetry tests, anisotropy quantification
- **Derived quantities**: vorticity ω, divergence ∇·u computation and analysis

## Jupyter Notebooks

One interactive Jupyter notebook is provided in the `notebooks/` directory for exploring the analysis step-by-step:

### English Version
- **File**: `notebooks/main_analysis_en.ipynb`
- **Content**: Complete data-driven analysis workflow with detailed interpretations
- **Sections**:
  - Step 0: Data Loading & Preprocessing
  - Step 1: SVD Analysis (Leading Spatial Structures)
  - Step 2: Fourier Spectral Analysis (Energy Distribution & Periodicity)
  - Step 3: Symmetry & Anisotropy Diagnosis
  - Summary: Comprehensive analysis results for Q1–Q4

**Usage**: Open notebook with Jupyter Lab/Notebook and run cells sequentially:
```bash
jupyter lab notebooks/main_analysis_en.ipynb
```

All generated figures are automatically saved to `figures/` and displayed inline.

## Dependencies

- Python ≥ 3.10
- `numpy` ≥ 1.24
- `matplotlib` ≥ 3.7
- `jupyter` (optional, for interactive notebook use)

## Install & Run

### Basic Installation (Command Line)

```bash
# Install minimal dependencies
pip install .

# Run the full analysis pipeline
python main.py
```

### Full Installation (With Jupyter Notebook Support)

If you want to use the interactive Jupyter notebooks:

```bash
# Install with optional notebook dependencies
pip install .[notebook]

# Launch notebook
jupyter lab notebooks/main_analysis_en.ipynb
```

All figures are automatically saved to the `figures/` directory.
