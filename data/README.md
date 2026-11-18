# Data Documentation

## 1. Synthetic Benchmark Applications (`CAP`, `PORT`, `NET`, and Mechanism Studies)
All synthetic datasets are generated deterministically on the fly from random seeds (`260911`--`260920` for `CAP` and `PORT`; `260911`--`260913` for `NET`, timing, sample-size scaling, and mechanism studies; `241011`--`241013` for the controlled basis study) using the frozen configuration in `data/configs/synthetic_protocol.json`. No external download is required.

## 2. European Interconnected Grid Reserve Study (`2018Q1`--`2024Q4`)
To enable immediate out-of-the-box replication of all 28 rolling quarterly windows (`2,557` operational days), `data/processed/` includes the two cleaned, strictly causal daily panels:

- `opsd_grid64_2015_2020q3.npz`: `2,098` complete UTC operational days (`2015-01-03` through `2020-09-30` after the two-day lag), used for the 11 early quarterly windows (`2018Q1`--`2020Q3`, `1,004` out-of-sample days). Derived from the Open Power System Data (OPSD) `2020-10-06` hourly time-series compilation (Wiese et al., 2019, CC-BY-4.0).
- `entsoe_grid64_2017q4_2024q4.npz`: `2,647` complete UTC operational days (`2017-10-03` through `2024-12-31` after the two-day lag), used for the 17 later quarterly windows (`2020Q4`--`2024Q4`, `1,553` out-of-sample days). Derived from the ENTSO-E Transparency Platform (Hirth et al., 2018) using the fixed 2015--2016 per-unit load bases `med_gw`.

Each `.npz` archive contains:
- `dates`: `N`-vector of ISO date strings (`YYYY-MM-DD`)
- `med_gw`: `8`-vector of 2015--2016 zonal median hourly loads in GW (`DE_50hertz`, `DE_amprion`, `DE_tennet`, `DE_transnetbw`, `FR`, `AT`, `DK_1`, `IT_NORD`)
- `contexts`: `(N, 16)` matrix of pre-gate-closure context vectors $x_t \in \mathbb{R}^{16}$ (Eq. EC.24)
- `outcomes`: `(N, 64)` matrix of 3-hour per-unit residual net loads $\xi_t \in (-2.2, 2.2)^{64}$

### Rebuilding Processed Panels from Raw Public Sources
1. **OPSD (2015--2020Q3)**: Download `time_series_60min_singleindex.csv` from <https://doi.org/10.25832/time_series/2020-10-06> into `data/raw/` and run:
   ```bash
   python -m grid_reserve.build_opsd_panel --raw-csv data/raw/time_series_60min_singleindex.csv --output data/processed/opsd_grid64_2015_2020q3.npz
   ```
2. **ENTSO-E (2017Q4--2024Q4)**: Export an ENTSO-E Web API security token (`export ENTSOE_SECURITY_TOKEN=...`, free registration at <https://transparency.entsoe.eu/>) and run:
   ```bash
   python -m grid_reserve.build_entsoe_panel --raw-dir data/raw/entsoe_xml --output data/processed/entsoe_grid64_2017q4_2024q4.npz
   ```
