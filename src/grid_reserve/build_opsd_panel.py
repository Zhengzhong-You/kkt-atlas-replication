"""Build the 2015--2020Q3 European grid panel (.npz) from raw Open Power System Data CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from pathlib import Path

import numpy as np

from grid_reserve.problem import ZONES


def build_opsd_panel(raw_csv: Path, output_npz: Path) -> None:
    start_ts = "2015-01-01T00:00:00Z"
    end_ts = "2020-09-30T23:00:00Z"
    zones = list(ZONES)
    r_cnt = len(zones)

    with raw_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        hdr = next(reader)
        idx = {c: i for i, c in enumerate(hdr)}
        load_cols = [idx[f"{z}_load_actual_entsoe_transparency"] for z in zones]
        wind_cols = [
            idx[f"{z}_wind_generation_actual"]
            if f"{z}_wind_generation_actual" in idx
            else idx[f"{z}_wind_onshore_generation_actual"]
            for z in zones
        ]
        solar_cols = [idx[f"{z}_solar_generation_actual"] for z in zones]

        timestamps: list[str] = []
        raw_load: list[list[float | None]] = []
        raw_wind: list[list[float | None]] = []
        raw_solar: list[list[float | None]] = []
        for row in reader:
            ts = row[0]
            if ts < start_ts or ts > end_ts:
                continue
            timestamps.append(ts)
            raw_load.append([float(row[c]) / 1000.0 if row[c].strip() != "" else None for c in load_cols])
            raw_wind.append([max(0.0, float(row[c]) / 1000.0) if row[c].strip() != "" else None for c in wind_cols])
            raw_solar.append([max(0.0, float(row[c]) / 1000.0) if row[c].strip() != "" else None for c in solar_cols])

    n_hours = len(timestamps)
    if n_hours != 50400:
        raise ValueError(f"Expected 50,400 UTC hours in 2015-01-01..2020-09-30, got {n_hours}")
    n_days = n_hours // 24

    # 2015--2016 initial-history zonal median hourly loads in GW
    n_init_hours = 2 * 365 * 24
    med_list = []
    for z_idx in range(r_cnt):
        vals = sorted(
            raw_load[h][z_idx]
            for h in range(n_init_hours)
            if raw_load[h][z_idx] is not None and raw_load[h][z_idx] > 0.0
        )
        med_list.append(float(vals[len(vals) // 2]))
    med_gw_raw = np.asarray(med_list, dtype=np.float64)
    med_gw = np.round(med_gw_raw, 4)

    clean_load = np.zeros((n_hours, r_cnt), dtype=np.float64)
    clean_wind = np.zeros((n_hours, r_cnt), dtype=np.float64)
    clean_solar = np.zeros((n_hours, r_cnt), dtype=np.float64)

    for h in range(n_hours):
        for z_idx in range(r_cnt):
            med = float(med_gw_raw[z_idx])
            lv = raw_load[h][z_idx]
            if lv is None or lv < 0.25 * med or lv > 2.20 * med:
                clean_load[h, z_idx] = (
                    clean_load[h - 24, z_idx] if h >= 24 else (clean_load[h - 1, z_idx] if h >= 1 else med)
                )
            else:
                clean_load[h, z_idx] = float(lv)
            wv = raw_wind[h][z_idx]
            clean_wind[h, z_idx] = (clean_wind[h - 24, z_idx] if h >= 24 else 0.0) if wv is None else float(wv)
            sv = raw_solar[h][z_idx]
            clean_solar[h, z_idx] = (clean_solar[h - 24, z_idx] if h >= 24 else 0.0) if sv is None else float(sv)

    clean_net_pu = (clean_load - clean_wind - clean_solar) / med_gw[None, :]
    load_3h_gw = np.zeros((n_days, r_cnt, 8), dtype=np.float64)
    ren_3h_gw = np.zeros((n_days, r_cnt, 8), dtype=np.float64)
    dates_all: list[str] = []
    weekdays = np.zeros(n_days, dtype=np.int64)
    doy = np.zeros(n_days, dtype=np.float64)

    for d in range(n_days):
        h0 = d * 24
        ds = timestamps[h0][:10]
        dt_obj = dt.date.fromisoformat(ds)
        dates_all.append(ds)
        weekdays[d] = dt_obj.weekday()
        doy[d] = float(dt_obj.timetuple().tm_yday)
        for z_idx in range(r_cnt):
            for blk in range(8):
                bh0 = h0 + blk * 3
                load_3h_gw[d, z_idx, blk] = round(sum(float(clean_load[bh0 + k, z_idx]) for k in range(3)) / 3.0, 5)
                ren_3h_gw[d, z_idx, blk] = round(
                    sum(float(clean_wind[bh0 + k, z_idx] + clean_solar[bh0 + k, z_idx]) for k in range(3)) / 3.0, 5
                )

    net_pu_3h = (load_3h_gw - ren_3h_gw) / med_gw[None, :, None]
    idx_de = [zones.index(z) for z in ("DE_50hertz", "DE_amprion", "DE_tennet", "DE_transnetbw")]
    idx_fr = zones.index("FR")
    idx_other = [zones.index(z) for z in ("AT", "DK_1", "IT_NORD")]

    valid_dates: list[str] = []
    outcomes_list: list[np.ndarray] = []
    contexts_list: list[np.ndarray] = []
    for t in range(2, n_days):
        valid_dates.append(dates_all[t])
        outcomes_list.append(net_pu_3h[t].reshape(-1))
        h_cut = (t - 1) * 24 + 8  # strictly < t-1 08:00 UTC (ahead of 12:00 CET/CEST gate closure)
        m_net = np.mean(clean_net_pu[h_cut - 3 : h_cut], axis=0)
        eve_tm2 = np.array(
            [
                float(np.mean(net_pu_3h[t - 2, idx_de, 6])),
                float(net_pu_3h[t - 2, idx_fr, 6]),
                float(np.mean(net_pu_3h[t - 2, idx_other, 6])),
            ],
            dtype=np.float64,
        )
        ang = 2.0 * math.pi * float(doy[t]) / 365.25
        harmonics = np.array([math.sin(ang), math.cos(ang), math.sin(2.0 * ang), math.cos(2.0 * ang)], dtype=np.float64)
        wknd = np.array([1.0 if int(weekdays[t]) >= 5 else 0.0], dtype=np.float64)
        contexts_list.append(np.concatenate([m_net, eve_tm2, harmonics, wknd]))

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        dates=np.asarray(valid_dates),
        med_gw=med_gw,
        outcomes=np.vstack(outcomes_list),
        contexts=np.vstack(contexts_list),
    )
    print(f"Wrote {len(valid_dates)} operational days to {output_npz}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True, help="Path to OPSD time_series_60min_singleindex.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/opsd_grid64_2015_2020q3.npz"),
        help="Output .npz path",
    )
    args = parser.parse_args()
    build_opsd_panel(args.raw_csv, args.output)


if __name__ == "__main__":
    main()
