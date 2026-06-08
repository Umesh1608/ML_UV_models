#!/usr/bin/env python3
"""
R2-1: Quantify experimental lambda_max measurement uncertainty.

Two complementary measurements:
  (A) Inter-source disagreement: same (canon, solvent) reported in both
      Joung and Beard. Difference distribution = inter-source variability.
  (B) Within-source duplicate spread (pre-cleaning): groups of records
      sharing (canon, solvent) within a single source. Spread distribution =
      intra-source variability + true biological reproducibility noise.

Outputs:
  results/r2_1_uncertainty.json
  results/r2_1_uncertainty_summary.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results"
PREV = SCRIPT_DIR / "previous_code"

JOUNG = PREV / "UV_canonical_dataset1.csv"   # Joung 2020 cleaned
BEARD = PREV / "UV_canonical_dataset2.csv"   # Beard 2019 cleaned
V2    = PREV / "UV_canonical_full_dataset.csv"  # combined v2


def load(path):
    df = pd.read_csv(path)
    df = df[["canon", "solvents", "lambda_max"]].dropna().reset_index(drop=True)
    df["lambda_max"] = pd.to_numeric(df["lambda_max"], errors="coerce")
    df = df.dropna(subset=["lambda_max"]).reset_index(drop=True)
    return df


def main():
    print(f"  loading sources")
    joung = load(JOUNG)
    beard = load(BEARD)
    v2    = load(V2)
    print(f"  Joung: {len(joung)} records, {joung['canon'].nunique()} unique chrom")
    print(f"  Beard: {len(beard)} records, {beard['canon'].nunique()} unique chrom")
    print(f"  v2 combined: {len(v2)} records")

    # ---- (A) Inter-source disagreement -----------------------------------
    # For each (canon, solvent), aggregate to one lambda_max per source
    # (averaging within-source duplicates first to isolate inter-source noise).
    joung_agg = joung.groupby(["canon", "solvents"])["lambda_max"].mean().reset_index()
    beard_agg = beard.groupby(["canon", "solvents"])["lambda_max"].mean().reset_index()
    inter = joung_agg.merge(beard_agg, on=["canon", "solvents"], suffixes=("_J", "_B"))
    inter["abs_diff"] = (inter["lambda_max_J"] - inter["lambda_max_B"]).abs()
    n_inter = len(inter)
    print(f"  inter-source paired observations: {n_inter}")

    if n_inter > 0:
        diffs = inter["abs_diff"].values
        inter_stats = {
            "n_paired": int(n_inter),
            "median_abs_diff_nm":  round(float(np.median(diffs)), 3),
            "mean_abs_diff_nm":    round(float(np.mean(diffs)),   3),
            "p75_abs_diff_nm":     round(float(np.percentile(diffs, 75)), 3),
            "p90_abs_diff_nm":     round(float(np.percentile(diffs, 90)), 3),
            "p95_abs_diff_nm":     round(float(np.percentile(diffs, 95)), 3),
            "max_abs_diff_nm":     round(float(np.max(diffs)),    3),
            "frac_within_5nm":     round(float((diffs <= 5).mean()),  3),
            "frac_within_10nm":    round(float((diffs <= 10).mean()), 3),
            "frac_within_20nm":    round(float((diffs <= 20).mean()), 3),
            "frac_within_50nm":    round(float((diffs <= 50).mean()), 3),
            "rmsd_inter_source":   round(float(np.sqrt(np.mean(diffs**2))), 3),
        }
    else:
        inter_stats = {"n_paired": 0, "note": "no (canon, solvent) pairs in both sources"}

    # ---- (B) Within-source duplicate spread (pre-cleaning) ---------------
    # For each (canon, solvent) within each source, compute spread = max - min
    def within_source_spread(df, name):
        grouped = df.groupby(["canon", "solvents"])["lambda_max"].agg(["count", "max", "min"])
        multi = grouped[grouped["count"] > 1]
        spreads = (multi["max"] - multi["min"]).values
        return {
            "source": name,
            "n_total_records": int(len(df)),
            "n_unique_pairs": int(len(grouped)),
            "n_pairs_multiplicity_gt1": int(len(multi)),
            "n_records_in_multi_pairs": int(multi["count"].sum()),
            "median_spread_nm": round(float(np.median(spreads)), 3) if len(spreads) else None,
            "p75_spread_nm":    round(float(np.percentile(spreads, 75)), 3) if len(spreads) else None,
            "p95_spread_nm":    round(float(np.percentile(spreads, 95)), 3) if len(spreads) else None,
            "frac_spread_leq_5nm":  round(float((spreads <= 5).mean()),  3) if len(spreads) else None,
            "frac_spread_leq_10nm": round(float((spreads <= 10).mean()), 3) if len(spreads) else None,
            "frac_spread_leq_20nm": round(float((spreads <= 20).mean()), 3) if len(spreads) else None,
            "frac_spread_gt_50nm":  round(float((spreads > 50).mean()),  3) if len(spreads) else None,
        }

    within_joung = within_source_spread(joung, "Joung")
    within_beard = within_source_spread(beard, "Beard")
    within_v2    = within_source_spread(v2,    "v2_combined")

    # ---- (C) Comparison to model error -----------------------------------
    # Headline model errors from the v3 cleaned dataset (Table 1):
    #   RF (tuned)        RMSE = 31.50 nm, MAE = 15.20 nm
    #   Chemprop (def)    RMSE = 33.15 nm, MAE = ...
    #   BiGRU (tuned)     RMSE = 36.20 nm, MAE = ...
    # The reviewer R2-1 wants a "headroom" comparison: how much of the model
    # error is data noise vs model error.
    model_errors = {
        "RF_RMSE_nm":           31.50,
        "Chemprop_default_RMSE_nm": 33.15,
        "Chemprop_tuned_RMSE_nm":   28.83,
        "BiGRU_tuned_RMSE_nm":      36.20,
        "ChemBERTa_RMSE_nm":        54.09,
    }

    out = {
        "inter_source": inter_stats,
        "within_source": {
            "joung": within_joung,
            "beard": within_beard,
            "v2_combined": within_v2,
        },
        "model_errors_v3": model_errors,
    }

    (RESULTS / "r2_1_uncertainty.json").write_text(json.dumps(out, indent=2))
    print(f"  wrote results/r2_1_uncertainty.json")

    # ---- Markdown summary -------------------------------------------------
    md = []
    md.append("# R2-1 Step 0: Experimental lambda_max Uncertainty")
    md.append("")
    md.append("## (A) Inter-Source (Joung vs Beard) Disagreement")
    if n_inter > 0:
        md.append(f"- Paired (canon, solvent) observations in both sources: **{n_inter}**")
        md.append(f"- Median |Joung - Beard|: **{inter_stats['median_abs_diff_nm']:.2f} nm**")
        md.append(f"- Mean   |Joung - Beard|: {inter_stats['mean_abs_diff_nm']:.2f} nm")
        md.append(f"- 75th pct: {inter_stats['p75_abs_diff_nm']:.2f} nm")
        md.append(f"- 90th pct: {inter_stats['p90_abs_diff_nm']:.2f} nm")
        md.append(f"- 95th pct: **{inter_stats['p95_abs_diff_nm']:.2f} nm**")
        md.append(f"- Maximum:  {inter_stats['max_abs_diff_nm']:.2f} nm")
        md.append(f"- Fraction within  5 nm: {inter_stats['frac_within_5nm']*100:.1f}%")
        md.append(f"- Fraction within 10 nm: {inter_stats['frac_within_10nm']*100:.1f}%")
        md.append(f"- Fraction within 20 nm: {inter_stats['frac_within_20nm']*100:.1f}%")
        md.append(f"- Inter-source RMSD: **{inter_stats['rmsd_inter_source']:.2f} nm**")
    else:
        md.append("- No (canon, solvent) pairs are present in both Joung and Beard sources.")
    md.append("")
    md.append("## (B) Within-Source Duplicate Spread (max - min)")
    for s in (within_joung, within_beard, within_v2):
        md.append(f"### {s['source']}")
        md.append(f"- Total records: {s['n_total_records']:,}")
        md.append(f"- Unique (canon, solvent) pairs: {s['n_unique_pairs']:,}")
        md.append(f"- Pairs with multiplicity >1: {s['n_pairs_multiplicity_gt1']:,} "
                  f"({s['n_records_in_multi_pairs']:,} records)")
        if s["median_spread_nm"] is not None:
            md.append(f"- Median spread: **{s['median_spread_nm']:.2f} nm**")
            md.append(f"- 75th pct: {s['p75_spread_nm']:.2f} nm")
            md.append(f"- 95th pct: **{s['p95_spread_nm']:.2f} nm**")
            md.append(f"- Fraction spread <=  5 nm: {s['frac_spread_leq_5nm']*100:.1f}%")
            md.append(f"- Fraction spread <= 10 nm: {s['frac_spread_leq_10nm']*100:.1f}%")
            md.append(f"- Fraction spread <= 20 nm: {s['frac_spread_leq_20nm']*100:.1f}%")
            md.append(f"- Fraction spread >  50 nm: {s['frac_spread_gt_50nm']*100:.1f}%")
        md.append("")
    md.append("## (C) Comparison to Model Errors (v3 cleaned dataset, Table~1)")
    for k, v in model_errors.items():
        md.append(f"- {k}: {v} nm")
    md.append("")
    md.append("**Headroom analysis** (if inter-source RMSD is interpretable):")
    if n_inter > 0:
        for label, rmse in [
            ("RF",         model_errors["RF_RMSE_nm"]),
            ("Chemprop",   model_errors["Chemprop_default_RMSE_nm"]),
            ("ChemBERTa",  model_errors["ChemBERTa_RMSE_nm"]),
        ]:
            data_noise = inter_stats["rmsd_inter_source"]
            model_noise = (rmse**2 - data_noise**2)**0.5 if rmse > data_noise else 0
            md.append(f"- {label}: total RMSE = {rmse:.1f} nm, "
                      f"inter-source noise floor = {data_noise:.1f} nm, "
                      f"residual (model-attributable) = {model_noise:.1f} nm "
                      f"({model_noise/rmse*100:.0f}% of total)")
    (RESULTS / "r2_1_uncertainty_summary.md").write_text("\n".join(md))
    print(f"  wrote results/r2_1_uncertainty_summary.md")

    # Print to stdout
    print("\n  === KEY NUMBERS ===")
    if n_inter > 0:
        print(f"  Inter-source: {n_inter} pairs, median |diff| = "
              f"{inter_stats['median_abs_diff_nm']:.2f} nm, "
              f"p95 = {inter_stats['p95_abs_diff_nm']:.2f} nm, "
              f"RMSD = {inter_stats['rmsd_inter_source']:.2f} nm")
    print(f"  Within-Joung: {within_joung['n_pairs_multiplicity_gt1']} multi-pairs, "
          f"median spread = {within_joung.get('median_spread_nm', 'N/A')} nm")
    print(f"  Within-Beard: {within_beard['n_pairs_multiplicity_gt1']} multi-pairs, "
          f"median spread = {within_beard.get('median_spread_nm', 'N/A')} nm")
    print(f"  v2 combined: {within_v2['n_pairs_multiplicity_gt1']} multi-pairs, "
          f"median spread = {within_v2.get('median_spread_nm', 'N/A')} nm")


if __name__ == "__main__":
    main()
