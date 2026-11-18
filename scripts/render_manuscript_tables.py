#!/usr/bin/env python3
"""Render and verify all numerical manuscript tables from results/*.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_LABELS = (
    ("atlas", "Atlas"),
    ("iceo_a", "ICEO"),
    ("iceo_sr", "ICEO-red"),
    ("sof", "SOF"),
    ("knn", "kNN"),
    ("knn_sr", "kNN-red"),
    ("saa", "SAA"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    rdir = args.results_dir

    syn = json.loads((rdir / "synthetic_main_summary.json").read_text(encoding="utf-8"))
    tim = json.loads((rdir / "synthetic_construction_timing.json").read_text(encoding="utf-8"))
    ge = json.loads((rdir / "grid_early_2018q1_2020q3.json").read_text(encoding="utf-8"))
    gl = json.loads((rdir / "grid_later_2020q4_2024q4.json").read_text(encoding="utf-8"))
    alpha = json.loads((rdir / "alpha_sensitivity_summary.json").read_text(encoding="utf-8"))
    samp = json.loads((rdir / "sample_scaling_summary.json").read_text(encoding="utf-8"))
    basis = json.loads((rdir / "basis_ablation_summary.json").read_text(encoding="utf-8"))

    print("=== Table 3: Out-of-Sample Decision Quality (Synthetic Applications) ===")
    for app, label in (("capacity", "CAP (10 seeds)"), ("portfolio", "PORT (10 seeds)"), ("network", "NET (3 seeds)")):
        print(f"\n[{label}]")
        m_dict = syn["applications"][app]["methods"]
        for key, name in METHOD_LABELS:
            row = m_dict[key]
            r_m, r_se = row["risk"]["mean"], row["risk"]["se"]
            e_m, e_se = row["relative_excess_percent"]["mean"], row["relative_excess_percent"]["se"]
            k_q = row["query_support_mean"]
            k_str = f"{k_q:6.1f}" if k_q is not None else "   n/a"
            print(f"  {name:<10} | K_query={k_str} | Risk={r_m:.4f} ({r_se:.4f}) | Excess={e_m:.2f}% ({e_se:.2f})")

    print("\n=== Table 4: Primary Model Construction Time (32-Core Wall / CPU Hours) ===")
    for row in tim:
        print(
            f"  {row['application']:<10} | {row['method']:<8} | "
            f"Wall={row['wall_hours']:.2f} h | CPU={row['cpu_hours']:.2f} h"
        )

    print("\n=== Table 7, Table EC.2 & Table EC.3: Rolling European Grid (28 Quarters, 2,557 Days) ===")
    ge_m = ge["a_methods_summary"]
    gl_m = gl["methods_summary"]
    ge_dep = ge["g_timing"]["deployment_2019Q1"]["methods"]
    for key, name in METHOD_LABELS:
        e_loss = ge_m[key]["pooled_1004d_mean_loss"]
        e_se = ge_m[key]["pooled_1004d_nw_se_lag14"]
        l_loss = gl_m[key]["pooled_mean_loss"]
        l_se = gl_m[key]["pooled_nw_se_lag14"]
        dep_32_ms = ge_dep[key]["ctx128_core32_s"]["median"] * 1000.0
        print(
            f"  {name:<10} | K={ge_m[key]['mean_scenarios']:5.1f} | "
            f"18Q1-20Q3={e_loss:.2f} ({e_se:.2f}) | "
            f"20Q4-24Q4={l_loss:.2f} ({l_se:.2f}) | "
            f"Deploy(128,32c)={dep_32_ms:.2f} ms"
        )

    print("\n=== Table 9: Spectral Retention Threshold Sensitivity (alpha_ret) ===")
    for a_key, a_lbl in (
        ("alpha_0p9", "0.9"),
        ("alpha_0p99", "0.99"),
        ("alpha_0p999", "0.999"),
        ("alpha_0p9999_v0", "0.9999 (default)"),
    ):
        cap_e = alpha["synthetic"]["capacity"][a_key]["relative_excess_percent"]["mean"]
        port_e = alpha["synthetic"]["portfolio"][a_key]["relative_excess_percent"]["mean"]
        if a_key == "alpha_0p9999_v0":
            g_all = basis["grid64"]["pca"]["combined_28q"]["v0_daily_mean_loss"]
        else:
            g_all = alpha["grid64"][a_key]["combined_28q"]["variant_daily_mean_loss"]
        print(f"  alpha_ret={a_lbl:<16} | CAP={cap_e:5.2f}% | PORT={port_e:6.2f}% | Grid(28q)={g_all:.2f}")

    print("\n=== Table 10: Training Sample Size Scaling (Mean Relative Excess %) ===")
    for app, label in (("capacity", "CAP"), ("portfolio", "PORT")):
        print(f"\n[{label}]")
        for key, name in METHOD_LABELS:
            vals = [
                f"{samp['applications'][app][f'n_{n}']['methods'][key]['relative_excess_percent']['mean']:.2f}"
                for n in (1600, 3200, 6400, 12800)
            ]
            print(f"  {name:<10} | 1600={vals[0]:>6}% | 3200={vals[1]:>6}% | 6400={vals[2]:>6}% | 12800={vals[3]:>6}%")

    print("\n=== Table EC.4: Local Basis Ablation ===")
    for app, label in (("capacity", "CAP"), ("portfolio", "PORT"), ("network", "NET")):
        b_app = basis["synthetic"][app]
        print(
            f"  {label:<6} | KKT={b_app['v0_kkt']['relative_excess_percent']['mean']:.2f}% | "
            f"PCA={b_app['pca']['relative_excess_percent']['mean']:.2f}% | "
            f"Full={b_app['full']['relative_excess_percent']['mean']:.2f}%"
        )
    g_pca = basis["grid64"]["pca"]["combined_28q"]
    g_full = basis["grid64"]["full"]["combined_28q"]
    print(
        f"  Grid28 | KKT={g_pca['v0_daily_mean_loss']:.2f}  | "
        f"PCA={g_pca['variant_daily_mean_loss']:.2f}  | "
        f"Full={g_full['variant_daily_mean_loss']:.2f}"
    )


if __name__ == "__main__":
    main()
