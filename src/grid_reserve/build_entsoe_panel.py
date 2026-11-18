"""Fetch 2017Q4--2024Q4 ENTSO-E Transparency Platform XML series and build entsoe_grid64_2017q4_2024q4.npz.

Requires `ENTSOE_SECURITY_TOKEN` in the environment when downloading missing raw XML chunks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import math
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from grid_reserve.problem import ZONE_EIC_CODES, ZONES

API_ENDPOINT = "https://web-api.tp.entsoe.eu/api"
FIXED_2015_2016_MED_GW = np.array(
    [9.8325, 20.7318, 17.3440, 6.8907, 52.6545, 6.8964, 2.1705, 17.3970],
    dtype=np.float64,
)
SERIES_SPECS: tuple[tuple[str, dict[str, str]], ...] = (
    ("load", {"documentType": "A65", "processType": "A16"}),
    ("solar", {"documentType": "A75", "processType": "A16", "psrType": "B16"}),
    ("wind_onshore", {"documentType": "A75", "processType": "A16", "psrType": "B19"}),
    ("wind_offshore", {"documentType": "A75", "processType": "A16", "psrType": "B18"}),
)
RES_MINUTES = {"PT15M": 15, "PT30M": 30, "PT60M": 60, "PT1H": 60}


def _half_year_chunks() -> list[tuple[dt.datetime, dt.datetime]]:
    out: list[tuple[dt.datetime, dt.datetime]] = []
    for yr in range(2017, 2025):
        c0 = dt.datetime(yr, 10, 1, 0, 0, tzinfo=dt.timezone.utc)
        c1 = dt.datetime(yr + 1, 4, 1, 0, 0, tzinfo=dt.timezone.utc)
        c2 = dt.datetime(yr + 1, 10, 1, 0, 0, tzinfo=dt.timezone.utc)
        out.append((c0, c1))
        if c1 < dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc):
            out.append((c1, min(c2, dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc))))
    return out


def _parse_entsoe_xml_into_hourly(
    xml_bytes: bytes,
    hourly_sum: np.ndarray,
    hourly_wt: np.ndarray,
    base_utc: dt.datetime,
) -> None:
    root = ET.fromstring(xml_bytes)
    ns = {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    p_ts = "ns:TimeSeries" if ns else "TimeSeries"
    p_per = "ns:Period" if ns else "Period"
    p_ti = "ns:timeInterval" if ns else "timeInterval"
    p_st = "ns:start" if ns else "start"
    p_res = "ns:resolution" if ns else "resolution"
    p_pt = "ns:Point" if ns else "Point"
    p_pos = "ns:position" if ns else "position"
    p_qty = "ns:quantity" if ns else "quantity"

    n_hours = len(hourly_sum)
    for ts_el in root.findall(p_ts, ns):
        for per_el in ts_el.findall(p_per, ns):
            ti_el = per_el.find(p_ti, ns)
            res_el = per_el.find(p_res, ns)
            if ti_el is None or res_el is None or res_el.text not in RES_MINUTES:
                continue
            step_m = RES_MINUTES[res_el.text]
            st_txt = ti_el.findtext(p_st, default="", namespaces=ns)
            t_start = dt.datetime.fromisoformat(st_txt.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
            for pt_el in per_el.findall(p_pt, ns):
                pos = int(pt_el.findtext(p_pos, default="0", namespaces=ns))
                qty_txt = pt_el.findtext(p_qty, default="", namespaces=ns)
                if pos <= 0 or not qty_txt:
                    continue
                t_pt = t_start + dt.timedelta(minutes=(pos - 1) * step_m)
                h_idx = int((t_pt - base_utc).total_seconds() // 3600)
                if 0 <= h_idx < n_hours:
                    w = step_m / 60.0
                    hourly_sum[h_idx] += float(qty_txt) * w
                    hourly_wt[h_idx] += w


def build_entsoe_panel(raw_dir: Path, output_npz: Path, med_gw: np.ndarray = FIXED_2015_2016_MED_GW) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    base_utc = dt.datetime(2017, 10, 1, 0, 0, tzinfo=dt.timezone.utc)
    end_utc = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    n_hours = int((end_utc - base_utc).total_seconds() // 3600)
    n_days = n_hours // 24
    zones = list(ZONES)
    r_cnt = len(zones)
    token = os.environ.get("ENTSOE_SECURITY_TOKEN", "").strip()

    arrays_mw = {
        name: np.full((n_hours, r_cnt), np.nan, dtype=np.float64)
        for name, _ in SERIES_SPECS
    }

    for z_idx, zone in enumerate(zones):
        eic = ZONE_EIC_CODES[zone]
        for s_name, params in SERIES_SPECS:
            h_sum = np.zeros(n_hours, dtype=np.float64)
            h_wt = np.zeros(n_hours, dtype=np.float64)
            for c0, c1 in _half_year_chunks():
                tag = f"{zone}__{s_name}__{c0.strftime('%Y%m%d')}_{c1.strftime('%Y%m%d')}.xml.gz"
                cache_file = raw_dir / tag
                if not cache_file.exists():
                    if not token:
                        raise RuntimeError(
                            f"Missing cached {cache_file.name}; set ENTSOE_SECURITY_TOKEN to download from ENTSO-E."
                        )
                    q = {
                        **params,
                        "in_Domain": eic if s_name != "load" else eic,
                        "outBiddingZone_Domain": eic,
                        "periodStart": c0.strftime("%Y%m%d%H%M"),
                        "periodEnd": c1.strftime("%Y%m%d%H%M"),
                        "securityToken": token,
                    }
                    if s_name != "load":
                        q.pop("outBiddingZone_Domain", None)
                    url = API_ENDPOINT + "?" + urllib.parse.urlencode(q)
                    time.sleep(0.45)
                    try:
                        with urllib.request.urlopen(url, timeout=60) as resp:
                            payload = resp.read()
                    except urllib.error.HTTPError as exc:
                        if exc.code == 400 and s_name == "wind_offshore":
                            continue
                        raise RuntimeError(f"ENTSO-E HTTP {exc.code} for {zone} {s_name}") from None
                    if payload[:2] == b"PK":
                        with zipfile.ZipFile(io.BytesIO(payload)) as zf_in:
                            payload = b"\n".join(zf_in.read(n) for n in zf_in.namelist())
                    with gzip.open(cache_file, "wb") as gz:
                        gz.write(payload)
                with gzip.open(cache_file, "rb") as gz:
                    xml_data = gz.read()
                if b"Acknowledgement_MarketDocument" in xml_data:
                    continue
                _parse_entsoe_xml_into_hourly(xml_data, h_sum, h_wt, base_utc)
            valid = h_wt > 0.0
            arrays_mw[s_name][valid, z_idx] = h_sum[valid] / h_wt[valid]

    raw_load_gw = arrays_mw["load"] / 1000.0
    raw_solar_gw = np.where(np.isnan(arrays_mw["solar"]), np.nan, np.maximum(0.0, arrays_mw["solar"] / 1000.0))
    raw_wind_gw = np.full((n_hours, r_cnt), np.nan, dtype=np.float64)
    for z_idx in range(r_cnt):
        on_mw = arrays_mw["wind_onshore"][:, z_idx]
        off_mw = arrays_mw["wind_offshore"][:, z_idx]
        off_valid = np.flatnonzero(~np.isnan(off_mw))
        first_off = int(off_valid[0]) if len(off_valid) > 0 else None
        for h in range(n_hours):
            has_on = not math.isnan(float(on_mw[h]))
            has_off = not math.isnan(float(off_mw[h]))
            if first_off is None or h < first_off:
                if has_on:
                    raw_wind_gw[h, z_idx] = max(0.0, float(on_mw[h]) / 1000.0)
            elif has_on and has_off:
                raw_wind_gw[h, z_idx] = max(0.0, float(on_mw[h]) / 1000.0) + max(0.0, float(off_mw[h]) / 1000.0)

    clean_load = np.zeros((n_hours, r_cnt), dtype=np.float64)
    clean_wind = np.zeros((n_hours, r_cnt), dtype=np.float64)
    clean_solar = np.zeros((n_hours, r_cnt), dtype=np.float64)
    for h in range(n_hours):
        for z_idx in range(r_cnt):
            med = float(med_gw[z_idx])
            lv = float(raw_load_gw[h, z_idx])
            if math.isnan(lv) or lv < 0.25 * med or lv > 2.20 * med:
                clean_load[h, z_idx] = (
                    clean_load[h - 24, z_idx] if h >= 24 else (clean_load[h - 1, z_idx] if h >= 1 else med)
                )
            else:
                clean_load[h, z_idx] = lv
            wv = float(raw_wind_gw[h, z_idx])
            clean_wind[h, z_idx] = (clean_wind[h - 24, z_idx] if h >= 24 else 0.0) if math.isnan(wv) else wv
            sv = float(raw_solar_gw[h, z_idx])
            clean_solar[h, z_idx] = (clean_solar[h - 24, z_idx] if h >= 24 else 0.0) if math.isnan(sv) else sv

    clean_net_pu = (clean_load - clean_wind - clean_solar) / med_gw[None, :]
    load_3h_gw = np.zeros((n_days, r_cnt, 8), dtype=np.float64)
    ren_3h_gw = np.zeros((n_days, r_cnt, 8), dtype=np.float64)
    dates_all: list[str] = []
    weekdays = np.zeros(n_days, dtype=np.int64)
    doy = np.zeros(n_days, dtype=np.float64)
    for d in range(n_days):
        dt_day = (base_utc + dt.timedelta(days=d)).date()
        dates_all.append(dt_day.isoformat())
        weekdays[d] = dt_day.weekday()
        doy[d] = float(dt_day.timetuple().tm_yday)
        h0 = d * 24
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
        h_cut = (t - 1) * 24 + 8
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
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/entsoe_xml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/entsoe_grid64_2017q4_2024q4.npz"))
    args = parser.parse_args()
    build_entsoe_panel(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
