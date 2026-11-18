# KKT Atlas: Learning Compact Local Scenario Models for Contextual Stochastic Optimization

Official code and data replication package for **"KKT Atlas: Learning Compact Local Scenario Models for Contextual Stochastic Optimization"** (*Operations Research*, double-blind review archive).

---

## 1. Overview

**KKT Atlas** learns compact, decision-aware local scenario models for repeated contextual stochastic optimization:
1. **Context Localization**: Partitions the context space $\mathcal{X}$ into $B$ Voronoi charts.
2. **Gradient-Informed Scenario Subspace**: Within each chart $b$, constructs the empirical cross-covariance operator $\widehat{C}_b$ between standardized context features and scenario cost gradients $\phi(z, \xi) = \nabla_z Q(z, \xi)$, retaining $d_b$ directions via the spectral energy rule ($\alpha_{\mathrm{ret}} = 0.9999$) and setting local support size $K_b = d_b + 1$.
3. **End-to-End Operational Fitting**: Jointly optimizes chart-local scenario coordinates and affine-softmax probability weights via implicit differentiation through the KKT conditions of the downstream optimization problem.

This repository provides all source code, processed data panels, raw-data reconstruction pipelines, experiment runners, audited numerical outputs, and automated tests required to reproduce every table and figure in Section 6 and the Electronic Companion (`EC.2`--`EC.3`).

### Repository Layout
```text
├── README.md                          # Replication guide and exhibit-to-script mapping
├── LICENSE                            # MIT License
├── pyproject.toml                     # Python package metadata (editable install)
├── requirements.txt                   # Runtime and test dependencies
├── data/
│   ├── README.md                      # Data provenance, licensing, and raw-to-processed pipeline docs
│   ├── configs/
│   │   └── synthetic_protocol.json    # Frozen DGP, solver, and hyperparameter configuration
│   └── processed/
│       ├── opsd_grid64_2015_2020q3.npz    # 2,098 daily European grid observations (2015--2020Q3, OPSD)
│       └── entsoe_grid64_2017q4_2024q4.npz # 2,647 daily European grid observations (2017Q4--2024Q4, ENTSO-E)
├── src/
│   ├── kkt_atlas/                     # Core KKT Atlas implementation, 6 benchmarks, and synthetic problems
│   └── grid_reserve/                  # 8-zone European grid reserve problem, solver, and data builders
├── scripts/
│   ├── run_synthetic_experiments.py   # Reproduces Tables 3, 4, 5, 6, 9, 10, and EC.4 (synthetic)
│   ├── run_grid_experiments.py        # Reproduces Tables 7, 9, EC.2, EC.3, and EC.4 (European grid)
│   ├── run_mechanism_experiments.py   # Reproduces Figures 1--2 and Table 8
│   └── render_manuscript_tables.py    # Formats and verifies Tables 3--10 and EC.2--EC.4 from results/
├── results/                           # Pre-computed JSON summaries and PDF figures matching the manuscript
└── tests/
    └── test_smoke.py                  # Unit and end-to-end smoke tests (pytest)
```

---

## 2. Data Availability Statement

All datasets used in the manuscript are publicly available, non-proprietary, and reproducible out of the box:

1. **Synthetic Applications (`CAP`, `PORT`, `NET`) and Mechanism Studies**: Generated deterministically in memory from fixed random seeds (`260911`--`260920` for `CAP` and `PORT`; `260911`--`260913` for `NET`, timing, sample-size scaling, and mechanism studies; `241011`--`241013` for the controlled basis study) under `data/configs/synthetic_protocol.json`.
2. **Interconnected European Grid Reserve Study (`2018Q1`--`2024Q4`, 28 rolling quarterly windows, 2,557 operational days)**:
   - **Early Horizon (`2018Q1`--`2020Q3`, 11 quarters, 1,004 days)**: Extracted from the Open Power System Data (OPSD) `2020-10-06` compilation ([Wiese et al., 2019](https://doi.org/10.25832/time_series/2020-10-06), licensed under CC-BY-4.0). Included ready to run at `data/processed/opsd_grid64_2015_2020q3.npz`.
   - **Later Horizon (`2020Q4`--`2024Q4`, 17 quarters, 1,553 days)**: Extracted from the ENTSO-E Transparency Platform ([Hirth et al., 2018](https://transparency.entsoe.eu/)). Included ready to run at `data/processed/entsoe_grid64_2017q4_2024q4.npz`.
   - Full raw-to-processed build instructions and scripts (`grid_reserve.build_opsd_panel` and `grid_reserve.build_entsoe_panel`) are documented in `data/README.md`.

---

## 3. Computational Requirements

### Software Dependencies
- **Python**: `>= 3.11` (tested on Python 3.12 / 3.13 across Linux, macOS, and Windows)
- **Core Open-Source Stack**: `numpy`, `scipy`, `torch` (CPU), `clarabel` (Apache-2.0 interior-point conic solver), `scikit-learn`, `threadpoolctl`
- **Optional Commercial Solver (`NET` only)**: `gurobipy >= 11.0` was used as the primary convex QP solver for the multicommodity network reservation (`NET`) timing benchmarks in Tables 4--5. When `gurobipy` is not installed or no license is present, `kkt_atlas` **automatically falls back to the open-source `clarabel` solver**, so all four problem families (`CAP`, `PORT`, `NET`, and `GRID`) execute without commercial software.

### Hardware Notes
- Primary wall-clock and CPU timing benchmarks in Tables 4, 5, 7, and EC.2 were measured on a single 32-physical-core CPU processor group (`workers=32`, `CUDA_VISIBLE_DEVICES=""`).
- Smoke tests (`pytest` and `--quick-smoke`) complete in under **30 seconds** on a standard laptop.

---

## 4. Instructions for Replicators

### Step 1: Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

### Step 2: Run Automated Verification & Render Manuscript Tables (< 15 seconds)
```bash
# 1. Run unit and end-to-end smoke tests across CAP, PORT, NET, and GRID
pytest -q

# 2. Verify and print all numerical tables from pre-computed results/
python scripts/render_manuscript_tables.py
```

### Step 3: Fast End-to-End Pipeline Smoke Runs (< 1 minute each)
Every experiment script provides a `--quick-smoke` flag to verify end-to-end training, checkpoint selection, and out-of-sample evaluation on a laptop:
```bash
python scripts/run_synthetic_experiments.py --experiment main --quick-smoke
python scripts/run_grid_experiments.py --experiment early --quick-smoke
python scripts/run_mechanism_experiments.py --experiment controlled-basis --quick-smoke
```

### Step 4: Full Manuscript Replication Commands
```bash
# Synthetic main comparison, structures, and timing (Tables 3, 4, 5, 6)
python scripts/run_synthetic_experiments.py --experiment main --workers 32

# Rolling European grid: 11 early quarters + 2019Q1 deployment timing (Table 7 & Table EC.2)
python scripts/run_grid_experiments.py --experiment early --workers 32

# Rolling European grid: 17 later quarters 2020Q4--2024Q4 (Table 7 & Table EC.3)
python scripts/run_grid_experiments.py --experiment later --workers 32

# Mechanism studies: support size (Fig. 1), chart angle (Fig. 2), controlled basis (Table 8)
python scripts/run_mechanism_experiments.py --experiment all --workers 32

# Sensitivity & ablation studies: local basis (Table EC.4), alpha_ret (Table 9), sample size (Table 10)
python scripts/run_synthetic_experiments.py --experiment basis --workers 32
python scripts/run_grid_experiments.py --experiment basis --workers 32
python scripts/run_synthetic_experiments.py --experiment alpha --workers 32
python scripts/run_grid_experiments.py --experiment alpha --workers 32
python scripts/run_synthetic_experiments.py --experiment sample-scaling --workers 32
```

---

## 5. List of Exhibits

| Manuscript Exhibit | Description | Reproduction Script & Flag | Pre-Computed Output in `results/` |
| :--- | :--- | :--- | :--- |
| **Table 3** (`tab:capPortAnchorQuality`) | Out-of-sample risk & relative excess (`CAP`, `PORT`, `NET`) | `scripts/run_synthetic_experiments.py --experiment main` | `results/synthetic_main_summary.json` |
| **Table 4** (`tab:capPortConstructionCost`) | Model construction wall & CPU time (`CAP`, `PORT`, `NET`) | `scripts/run_synthetic_experiments.py --experiment main` | `results/synthetic_construction_timing.json` |
| **Table 5** (`tab:deploymentTiming`) | Deployment wall time (1 vs. 128 contexts, 1 vs. 32 cores) | `src/kkt_atlas/symmetric_deployment_v1.py` | `results/synthetic_main_summary.json` |
| **Table 6** (`tab:capPortStructures`) | Selected $(\widehat{B}, d_b, K_b)$, parameters, and reduction | `scripts/run_synthetic_experiments.py --experiment main` | `results/synthetic_main_summary.json` |
| **Table 7** (`tab:grid64Rolling`) | Rolling European grid summary (`2018Q1`--`2024Q4`, 28 qtrs) | `scripts/run_grid_experiments.py --experiment early` & `later` | `results/grid_early_2018q1_2020q3.json`, `results/grid_later_2020q4_2024q4.json` |
| **Figure 1** (`fig:fixedKCapCurve`) | Decision quality vs. local support size $K$ in `CAP` | `scripts/run_mechanism_experiments.py --experiment support-size` | `results/figures/fixed_k_cap_curve.pdf` |
| **Figure 2** (`fig:chartSensitivity`) | Decision quality across mode plane angles $\omega$ and $B$ | `scripts/run_mechanism_experiments.py --experiment chart-angle` | `results/figures/theory_chart_sensitivity.pdf` |
| **Table 8** (`tab:controlledBasisMeans`) | Cross-covariance vs. PCA under irrelevant variation $\sigma$ | `scripts/run_mechanism_experiments.py --experiment controlled-basis` | `outputs/mechanism/controlled_basis_run.json` |
| **Table 9** (`tab:ecAlphaSensitivity`) | Sensitivity to spectral retention threshold $\alpha_{\mathrm{ret}}$ | `run_synthetic_experiments.py --experiment alpha` & `run_grid_experiments.py --experiment alpha` | `results/alpha_sensitivity_summary.json` |
| **Table 10** (`tab:ecSampleScaling`) | Training sample size scaling ($\lvert I_{\mathrm{tr}}\rvert \in \{1600,\dots,12800\}$) | `scripts/run_synthetic_experiments.py --experiment sample-scaling` | `results/sample_scaling_summary.json` |
| **Table EC.2** (`tab:grid64EarlyDetails`) | Detailed first-period European grid metrics (`2018Q1`--`2020Q3`) | `scripts/run_grid_experiments.py --experiment early` | `results/grid_early_2018q1_2020q3.json` |
| **Table EC.3** (`tab:grid64Later`) | Detailed later-period European grid metrics (`2020Q4`--`2024Q4`) | `scripts/run_grid_experiments.py --experiment later` | `results/grid_later_2020q4_2024q4.json` |
| **Table EC.4** (`tab:ecBasisAblation`) | Local basis ablation (`KKT` vs. `PCA` vs. `Full`) across all studies | `run_synthetic_experiments.py --experiment basis` & `run_grid_experiments.py --experiment basis` | `results/basis_ablation_summary.json` |

---

## License

This replication package is released under the [MIT License](LICENSE).
