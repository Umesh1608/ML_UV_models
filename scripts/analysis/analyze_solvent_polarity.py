#!/usr/bin/env python3
"""
R2-5: Polarity-grouped prediction variability.

Groups v3 solvents by polarity tier (using ET(30) and similar standard polarity
classifications). For each (model, tier) cell, computes test-fold RMSE and MAE.
Reveals whether any tier is systematically harder for the models.

Outputs:
  results/r2_5_polarity_breakdown.{json,csv,png}
  results/r2_5_summary.md
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results"
DATA = SCRIPT_DIR / "previous_code" / "UV_canonical_v3_dedup.csv"
FOLD_NPZ = RESULTS / "cv_fold_indices_v3.npz"

# Polarity classification by canonical SMILES.
# Tier ordering (low->high polarity): nonpolar -> moderate -> polar aprotic -> polar protic
# Based on standard ET(30) groupings (Reichardt 2011).
POLARITY_TIERS = {
    "nonpolar": {
        "CCCCCC":       "n-hexane",
        "CCCCCCC":      "n-heptane",
        "CCCCCCCC":     "n-octane",
        "C1CCCCC1":     "cyclohexane",
        "c1ccccc1":     "benzene",
        "Cc1ccccc1":    "toluene",
        "Cc1ccccc1C":   "o-xylene",
        "ClC(Cl)(Cl)Cl":"CCl4",
        "CCCCCCCCCC":   "n-decane",
    },
    "moderate": {
        "ClCCl":             "DCM",
        "ClC(Cl)Cl":         "CHCl3",
        "ClCCCl":            "1,2-DCE",
        "C1CCOC1":           "THF",
        "C1CCOC(C)C1":       "2-Me-THF",
        "CCOC(C)=O":         "EtOAc",
        "CC(=O)OC":          "MeOAc",
        "C1COCCO1":          "1,4-dioxane",
        "CC(C)=O":           "acetone",
        "COC(=O)C":          "MeOAc",
        "C(=O)(OCC)C":       "EtOAc",
    },
    "polar_aprotic": {
        "CC#N":              "MeCN",
        "CN(C)C=O":          "DMF",
        "CS(C)=O":           "DMSO",
        "CCC#N":             "propionitrile",
        "CC(C)=O":           "acetone (also moderate, kept here for canonical use)",
        "CN(C)C(C)=O":       "DMA",
        "O=C1OCCC1":         "GBL",
        "CCN(CC)C=O":        "DEF",
        "CN1CCCC1=O":        "NMP",
        "CN1CC[NH+](C)CC1":  "[NMM]",
    },
    "polar_protic": {
        "O":                 "water",
        "CO":                "MeOH",
        "CCO":               "EtOH",
        "CCCO":              "n-PrOH",
        "CC(C)O":            "i-PrOH",
        "CCCCO":             "n-BuOH",
        "CCCCCO":            "n-PenOH",
        "OCCO":              "ethylene glycol",
        "OCC(O)CO":          "glycerol",
    },
}


def build_tier_map():
    """Inverse map: canonical SMILES -> tier label."""
    m = {}
    for tier, entries in POLARITY_TIERS.items():
        for smi in entries:
            m[smi] = tier
    return m


def compute_rmse_mae(y_true, y_pred):
    err = y_pred - y_true
    return (float(np.sqrt(np.mean(err**2))),
            float(np.mean(np.abs(err))))


def main():
    print(f"  loading {DATA}")
    df = pd.read_csv(DATA)
    df["lambda_max"] = pd.to_numeric(df["lambda_max"], errors="coerce")
    df = df.dropna(subset=["canon", "solvents", "lambda_max"]).reset_index(drop=True)

    tier_map = build_tier_map()
    df["tier"] = df["solvents"].map(tier_map).fillna("other")

    coverage = df["tier"].value_counts().to_dict()
    print(f"  tier coverage (records):")
    for tier in ["nonpolar", "moderate", "polar_aprotic", "polar_protic", "other"]:
        n = coverage.get(tier, 0)
        print(f"    {tier:15s}: {n:6d} ({n/len(df)*100:5.1f}%)")

    # Models with v3 per-fold predictions available
    MODELS = [
        ("RF (tuned)",    "rf_tuned_v3"),
        ("Chemprop",      "chemprop_v3"),
        ("BiGRU (tuned)", "bigru_tuned_v3"),
        ("ChemBERTa",     "chemberta_v3"),
    ]
    N_FOLDS = 5

    # Per-(model, tier) results
    out_rows = []
    for model_label, prefix in MODELS:
        # Collect all 5 folds
        all_y, all_p, all_idx = [], [], []
        for f in range(N_FOLDS):
            p_path = RESULTS / f"{prefix}_fold{f}_predictions.npy"
            y_path = RESULTS / f"{prefix}_fold{f}_y_test.npy"
            t_path = RESULTS / f"{prefix}_fold{f}_test_indices.npy"
            if not (p_path.exists() and y_path.exists() and t_path.exists()):
                print(f"    [skip] {prefix} fold {f}")
                continue
            all_y.append(np.load(y_path).flatten())
            all_p.append(np.load(p_path).flatten())
            all_idx.append(np.load(t_path))
        if not all_y:
            print(f"    {model_label}: no folds found")
            continue
        y    = np.concatenate(all_y)
        p    = np.concatenate(all_p)
        idx  = np.concatenate(all_idx)
        tiers = df.iloc[idx]["tier"].values

        for tier in ["nonpolar", "moderate", "polar_aprotic", "polar_protic", "other", "ALL"]:
            mask = (tiers == tier) if tier != "ALL" else np.ones(len(tiers), dtype=bool)
            n = int(mask.sum())
            if n == 0:
                continue
            rmse, mae = compute_rmse_mae(y[mask], p[mask])
            out_rows.append({
                "model":    model_label,
                "tier":     tier,
                "n_test":   n,
                "RMSE":     round(rmse, 3),
                "MAE":      round(mae,  3),
                "mean_lmax": round(float(np.mean(y[mask])), 2),
            })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(RESULTS / "r2_5_polarity_breakdown.csv", index=False)
    print(f"\n  wrote results/r2_5_polarity_breakdown.csv")

    # JSON aggregate
    out_json = {"coverage_counts": coverage, "results": out_rows}
    (RESULTS / "r2_5_polarity_breakdown.json").write_text(json.dumps(out_json, indent=2))
    print(f"  wrote results/r2_5_polarity_breakdown.json")

    # Print pivot table
    pivot_rmse = out_df.pivot_table(index="tier", columns="model", values="RMSE")
    pivot_n    = out_df.pivot_table(index="tier", columns="model", values="n_test")
    print("\n  === RMSE pivot (tier x model) ===")
    print(pivot_rmse.round(2).to_string())
    print("\n  === N per cell ===")
    print(pivot_n.fillna(0).astype(int).to_string())

    # Plot
    tier_order = ["nonpolar", "moderate", "polar_aprotic", "polar_protic", "other"]
    pivot_for_plot = pivot_rmse.reindex(tier_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tier_order))
    width = 0.18
    colors = ["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]
    for i, (model_label, _) in enumerate(MODELS):
        if model_label not in pivot_for_plot.columns:
            continue
        vals = pivot_for_plot[model_label].values
        ax.bar(x + (i - 1.5) * width, vals, width, label=model_label, color=colors[i],
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ") for t in tier_order], fontsize=10)
    ax.set_ylabel("RMSE (nm) on pooled 5-fold test sets", fontsize=11)
    ax.set_title("Test-fold RMSE by solvent polarity tier (v3 cleaned dataset)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_5_polarity_breakdown.png", dpi=180, bbox_inches="tight")
    print(f"\n  wrote results/r2_5_polarity_breakdown.png")

    # Markdown summary
    md = []
    md.append("# R2-5 Step 0: Polarity-Grouped Prediction Variability")
    md.append("")
    md.append("## Tier coverage (v3 records)")
    for tier in tier_order:
        n = coverage.get(tier, 0)
        md.append(f"- {tier}: {n:,} records ({n/len(df)*100:.1f}%)")
    md.append(f"- TOTAL: {len(df):,} records")
    md.append("")
    md.append("## Test-fold RMSE by (model, tier)")
    md.append("")
    md.append("| Tier | " + " | ".join(m for m, _ in MODELS) + " |")
    md.append("|---|" + "---|" * len(MODELS))
    for tier in tier_order + ["ALL"]:
        row = [tier.replace("_", " ")]
        for model_label, _ in MODELS:
            sub = out_df[(out_df["tier"] == tier) & (out_df["model"] == model_label)]
            if not sub.empty:
                r = sub.iloc[0]
                row.append(f"{r['RMSE']:.1f} (n={int(r['n_test'])})")
            else:
                row.append("--")
        md.append("| " + " | ".join(row) + " |")
    (RESULTS / "r2_5_summary.md").write_text("\n".join(md))
    print(f"  wrote results/r2_5_summary.md")


if __name__ == "__main__":
    main()
