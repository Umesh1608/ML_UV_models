#!/usr/bin/env python3
"""
R1-16: Diagnose the ChemBERTa ~600 nm prediction ceiling.

The reviewer flagged that ChemBERTa's parity plot shows a horizontal cluster of
predictions around 600 nm even when the experimental value extends well beyond
that. This script tests three hypotheses:

  (H1) Training-distribution imbalance: are >600 nm chromophores rare in train?
  (H2) Output-range truncation: is the ceiling near the empirical 99th
       percentile of the training distribution (consistent with the regression
       head learning to "average out" rather than extrapolate)?
  (H3) SMILES-length / MAX_LEN truncation: do >600 nm chromophores have
       longer SMILES that get truncated at MAX_LEN=256, removing
       conjugation tokens needed for accurate prediction?

Outputs:
  results/r1_16_chemberta_diagnostic.{json,csv,png}
  results/r1_16_summary.md
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results"
DATA = SCRIPT_DIR / "previous_code" / "UV_canonical_v3_dedup.csv"

CHEMBERTA_PREFIX = "chemberta_v3"
N_FOLDS = 5
MAX_LEN = 256   # ChemBERTa max_length (run_chemberta.py default)


def load_concatenated(prefix):
    y_list, p_list, idx_list = [], [], []
    for f in range(N_FOLDS):
        y_path = RESULTS / f"{prefix}_fold{f}_y_test.npy"
        p_path = RESULTS / f"{prefix}_fold{f}_predictions.npy"
        t_path = RESULTS / f"{prefix}_fold{f}_test_indices.npy"
        if not (y_path.exists() and p_path.exists() and t_path.exists()):
            print(f"    [skip] {prefix} fold {f}")
            continue
        y_list.append(np.load(y_path).flatten())
        p_list.append(np.load(p_path).flatten())
        idx_list.append(np.load(t_path))
    return (np.concatenate(y_list),
            np.concatenate(p_list),
            np.concatenate(idx_list))


def main():
    print(f"  loading v3 dataset")
    df = pd.read_csv(DATA)
    df["lambda_max"] = pd.to_numeric(df["lambda_max"], errors="coerce")
    df = df.dropna(subset=["lambda_max"]).reset_index(drop=True)

    # Combined-input length matches run_baselines.py / run_chemberta.py:
    # "<solute>!<solvent>" or similar. Approximate as canon + "!" + solvents.
    df["combined"] = df["canon"].fillna(df["smiles"]) + "!" + df["solvents"].fillna("")
    df["combined_len"] = df["combined"].astype(str).str.len()

    print(f"  v3 size: {len(df)} records")

    # ==== (H1) Training-distribution imbalance ============================
    # Use fold-0 train slice to approximate the training distribution
    # ChemBERTa sees. ChemBERTa was trained on the same v3 folds as the
    # other models, so train_0 indices are the same.
    import numpy as np
    z = np.load(RESULTS / "cv_fold_indices_v3.npz")
    if "train_0" in z:
        train_idx = z["train_0"]
    else:
        train_idx = np.arange(len(df))  # fallback: all
    train_lmax = df.iloc[train_idx]["lambda_max"].values

    h1_bins = {
        "n_train":                int(len(train_lmax)),
        "n_train_gt_600nm":       int((train_lmax > 600).sum()),
        "n_train_gt_700nm":       int((train_lmax > 700).sum()),
        "n_train_gt_800nm":       int((train_lmax > 800).sum()),
        "frac_train_gt_600":      round(float((train_lmax > 600).mean()), 4),
        "frac_train_gt_700":      round(float((train_lmax > 700).mean()), 4),
        "frac_train_gt_800":      round(float((train_lmax > 800).mean()), 4),
        "train_median_lmax":      round(float(np.median(train_lmax)), 1),
        "train_p95_lmax":         round(float(np.percentile(train_lmax, 95)), 1),
        "train_p99_lmax":         round(float(np.percentile(train_lmax, 99)), 1),
        "train_max_lmax":         round(float(np.max(train_lmax)), 1),
    }
    print(f"\n  (H1) Training distribution (fold 0):")
    for k, v in h1_bins.items():
        print(f"    {k}: {v}")

    # ==== (H2) Output range / prediction distribution =====================
    y_all, p_all, idx_all = load_concatenated(CHEMBERTA_PREFIX)
    print(f"\n  ChemBERTa concat: |y_test|={len(y_all)}")
    h2 = {
        "pred_min":           round(float(np.min(p_all)),    1),
        "pred_max":           round(float(np.max(p_all)),    1),
        "pred_median":        round(float(np.median(p_all)), 1),
        "pred_p95":           round(float(np.percentile(p_all, 95)), 1),
        "pred_p99":           round(float(np.percentile(p_all, 99)), 1),
        "frac_pred_gt_600":   round(float((p_all > 600).mean()), 4),
        "frac_pred_gt_700":   round(float((p_all > 700).mean()), 4),
        # On test records where y_true > 600 nm, what fraction of predictions
        # are clamped at "near 600" (e.g., within [550, 650])?
    }
    mask_gt600 = y_all > 600
    if mask_gt600.sum() > 0:
        p_for_gt600 = p_all[mask_gt600]
        h2.update({
            "n_test_y_gt_600":              int(mask_gt600.sum()),
            "median_pred_when_y_gt_600":    round(float(np.median(p_for_gt600)), 1),
            "p25_pred_when_y_gt_600":       round(float(np.percentile(p_for_gt600, 25)), 1),
            "p75_pred_when_y_gt_600":       round(float(np.percentile(p_for_gt600, 75)), 1),
            "frac_pred_in_550_650_when_y_gt_600": round(float(
                ((p_for_gt600 >= 550) & (p_for_gt600 <= 650)).mean()), 4),
            "frac_pred_lt_y_minus_50_when_y_gt_600": round(float(
                (p_for_gt600 < (y_all[mask_gt600] - 50)).mean()), 4),
            "median_underprediction_when_y_gt_600_nm": round(float(
                np.median(y_all[mask_gt600] - p_for_gt600)), 1),
        })
    print(f"\n  (H2) ChemBERTa prediction distribution:")
    for k, v in h2.items():
        print(f"    {k}: {v}")

    # ==== (H3) SMILES-length vs lambda_max correlation ====================
    # For test records, get SMILES length and check correlation with
    # underprediction magnitude.
    test_df = df.iloc[idx_all].reset_index(drop=True)
    test_df["pred"]     = p_all
    test_df["y_true"]   = y_all
    test_df["error"]    = p_all - y_all  # predicted - true; negative = underprediction

    h3 = {
        "smiles_len_p50":            round(float(test_df["combined_len"].median()),    1),
        "smiles_len_p95":            round(float(test_df["combined_len"].quantile(0.95)), 1),
        "smiles_len_max":            int(test_df["combined_len"].max()),
        "frac_truncated_at_MAX_LEN": round(float((test_df["combined_len"] > MAX_LEN).mean()), 4),
        "MAX_LEN": MAX_LEN,
    }
    # Mean SMILES length by lambda_max bin
    bins = [(150, 400), (400, 500), (500, 600), (600, 700), (700, 1100)]
    h3["smiles_len_by_y_bin"] = {}
    for lo, hi in bins:
        sub = test_df[(test_df["y_true"] >= lo) & (test_df["y_true"] < hi)]
        if len(sub) == 0:
            continue
        h3["smiles_len_by_y_bin"][f"y_{lo}-{hi}"] = {
            "n":                  int(len(sub)),
            "mean_smiles_len":    round(float(sub["combined_len"].mean()),    1),
            "median_smiles_len":  round(float(sub["combined_len"].median()),  1),
            "frac_truncated":     round(float((sub["combined_len"] > MAX_LEN).mean()), 4),
        }
    print(f"\n  (H3) SMILES length vs lambda_max:")
    for k, v in h3.items():
        if isinstance(v, dict):
            print(f"    {k}:")
            for kk, vv in v.items():
                print(f"      {kk}: {vv}")
        else:
            print(f"    {k}: {v}")

    out = {"H1_training_distribution": h1_bins,
           "H2_prediction_distribution": h2,
           "H3_smiles_length_by_y": h3}
    (RESULTS / "r1_16_chemberta_diagnostic.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote results/r1_16_chemberta_diagnostic.json")

    # Per-test-record diagnostic CSV
    diag = test_df[["canon", "solvents", "y_true", "pred", "error",
                    "combined_len"]].copy()
    diag["truncated"] = diag["combined_len"] > MAX_LEN
    diag.sort_values("y_true", ascending=False).head(200).to_csv(
        RESULTS / "r1_16_chemberta_diagnostic_top_y.csv", index=False)
    print(f"  wrote results/r1_16_chemberta_diagnostic_top_y.csv "
          f"(top 200 test records by y_true)")

    # Plot: (a) train + test histograms of lambda_max; (b) pred vs y_true
    # scatter colored by whether the SMILES would be truncated; (c) bar
    # chart of mean SMILES length by y_true bin.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Train+test histogram
    ax = axes[0]
    ax.hist(train_lmax, bins=60, alpha=0.55, color="#5B8DBF", label=f"train n={len(train_lmax):,}",
            edgecolor="#2D4D6E", linewidth=0.3)
    ax.hist(y_all,      bins=60, alpha=0.55, color="#CE5B5B", label=f"test  n={len(y_all):,}",
            edgecolor="#732D2D", linewidth=0.3)
    ax.axvline(600, color="black", linestyle="--", linewidth=1.2, label="600 nm")
    ax.set_xlim(150, 1000)
    ax.set_xlabel(r"$\lambda_{\max}$ (nm)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"(a) v3 fold-0 distribution: {h1_bins['n_train_gt_600nm']:,} train + "
                 f"{int(mask_gt600.sum())} test above 600 nm",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")

    # (b) ChemBERTa parity, color = truncated
    ax = axes[1]
    trunc = (test_df["combined_len"] > MAX_LEN).values
    ax.scatter(y_all[~trunc], p_all[~trunc], alpha=0.25, s=10,
               c="steelblue", label=f"not truncated (n={(~trunc).sum():,})", rasterized=True)
    ax.scatter(y_all[trunc],  p_all[trunc],  alpha=0.60, s=14,
               c="crimson",   label=f"truncated >{MAX_LEN} chars (n={trunc.sum()})",
               rasterized=True)
    ax.plot([150, 1000], [150, 1000], "k--", linewidth=1.0, alpha=0.7)
    ax.axhline(600, color="orange", linestyle=":", linewidth=1.2, label="pred = 600 nm")
    ax.set_xlim(150, 1000); ax.set_ylim(150, 1000)
    ax.set_xlabel(r"Experimental $\lambda_{\max}$ (nm)", fontsize=11)
    ax.set_ylabel(r"ChemBERTa predicted $\lambda_{\max}$ (nm)", fontsize=11)
    ax.set_title("(b) Parity plot, coloured by SMILES truncation",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle=":")

    # (c) Mean SMILES length by y_true bin
    ax = axes[2]
    bin_labels = list(h3["smiles_len_by_y_bin"].keys())
    means      = [h3["smiles_len_by_y_bin"][k]["mean_smiles_len"] for k in bin_labels]
    fracs_tr   = [h3["smiles_len_by_y_bin"][k]["frac_truncated"] * 100 for k in bin_labels]
    ns         = [h3["smiles_len_by_y_bin"][k]["n"]               for k in bin_labels]
    x = np.arange(len(bin_labels))
    bars = ax.bar(x, means, color="#7BAACE", edgecolor="#2D4D6E", linewidth=0.5)
    ax.axhline(MAX_LEN, color="crimson", linestyle="--", linewidth=1.4,
               label=f"MAX_LEN = {MAX_LEN}")
    for xi, n, f in zip(x, ns, fracs_tr):
        ax.text(xi, max(means) * 0.08, f"n={n}\n{f:.0f}% trunc",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("y_", "").replace("-", " - ") + "nm" for b in bin_labels],
                       fontsize=10, rotation=15)
    ax.set_ylabel("Mean combined-SMILES length (chars)", fontsize=11)
    ax.set_title("(c) Combined-SMILES length by $\\lambda_{max}$ bin",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(RESULTS / "r1_16_chemberta_diagnostic.png", dpi=180, bbox_inches="tight")
    print(f"  wrote results/r1_16_chemberta_diagnostic.png")

    # Markdown summary
    md = []
    md.append("# R1-16 Step 0: ChemBERTa 600 nm Prediction-Ceiling Diagnostic")
    md.append("")
    md.append("## (H1) Training-distribution imbalance (fold 0)")
    md.append(f"- Train records: {h1_bins['n_train']:,}")
    md.append(f"- Train records with lambda_max > 600 nm: {h1_bins['n_train_gt_600nm']:,} "
              f"({h1_bins['frac_train_gt_600']*100:.2f}%)")
    md.append(f"- Train records with lambda_max > 700 nm: {h1_bins['n_train_gt_700nm']:,} "
              f"({h1_bins['frac_train_gt_700']*100:.2f}%)")
    md.append(f"- Train records with lambda_max > 800 nm: {h1_bins['n_train_gt_800nm']:,}")
    md.append(f"- Train median: {h1_bins['train_median_lmax']} nm; p95: {h1_bins['train_p95_lmax']} nm; "
              f"p99: {h1_bins['train_p99_lmax']} nm")
    md.append("")
    md.append("## (H2) ChemBERTa prediction distribution")
    md.append(f"- Pred min/median/max: {h2['pred_min']} / {h2['pred_median']} / {h2['pred_max']} nm")
    md.append(f"- Frac pred > 600 nm: {h2['frac_pred_gt_600']*100:.2f}%")
    md.append(f"- Frac pred > 700 nm: {h2['frac_pred_gt_700']*100:.2f}%")
    if "n_test_y_gt_600" in h2:
        md.append(f"- Test records with y_true > 600 nm: **{h2['n_test_y_gt_600']}**")
        md.append(f"- Median ChemBERTa prediction when y > 600: **{h2['median_pred_when_y_gt_600']} nm**")
        md.append(f"- Frac of those predictions in [550, 650]: "
                  f"**{h2['frac_pred_in_550_650_when_y_gt_600']*100:.1f}%**")
        md.append(f"- Frac underprediction (pred < y - 50): "
                  f"**{h2['frac_pred_lt_y_minus_50_when_y_gt_600']*100:.1f}%**")
        md.append(f"- Median underprediction (y - pred): "
                  f"**{h2['median_underprediction_when_y_gt_600_nm']} nm**")
    md.append("")
    md.append("## (H3) SMILES length vs lambda_max bin")
    md.append(f"- Test set median combined-SMILES length: {h3['smiles_len_p50']} chars")
    md.append(f"- p95: {h3['smiles_len_p95']} chars; max: {h3['smiles_len_max']} chars")
    md.append(f"- Frac of test records truncated at MAX_LEN={MAX_LEN}: "
              f"**{h3['frac_truncated_at_MAX_LEN']*100:.2f}%**")
    md.append("")
    md.append("| y_true bin (nm) | n | mean SMILES len | frac truncated |")
    md.append("|---|---:|---:|---:|")
    for k, v in h3["smiles_len_by_y_bin"].items():
        md.append(f"| {k.replace('y_', '')} | {v['n']} | {v['mean_smiles_len']:.1f} | "
                  f"{v['frac_truncated']*100:.1f}% |")
    md.append("")
    (RESULTS / "r1_16_summary.md").write_text("\n".join(md))
    print(f"  wrote results/r1_16_summary.md")


if __name__ == "__main__":
    main()
