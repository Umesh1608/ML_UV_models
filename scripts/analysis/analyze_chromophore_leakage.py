#!/usr/bin/env python3
"""
R1-9 Step 0 — Chromophore-solvent leakage analysis.

Greenman et al.\ (2022, Chem.\ Sci., DOI 10.1039/D1SC05677H) raised the concern
that the standard random/stratified CV split can leak the same chromophore
between train and test under different solvents, giving the model an easier
prediction target than a true scaffold split would.

This script quantifies the magnitude of that leakage on the v3 cleaned
Joung+Beard primary dataset and the existing v3 5-fold stratified splits.

For each fold and each model:
  - Identify unique canonical solute SMILES in test
  - Identify unique canonical solute SMILES in train (the 64% slice that
    the fingerprint models actually fit on; DL models also see val)
  - Test records are partitioned by whether their solute *also* appears in
    train under a *different* solvent ("leaked") or not ("novel solute")
  - Compute RMSE / MAE on each subset for all 5 v3 model families using the
    saved per-fold .npy predictions

Outputs:
  results/r1_9_leakage_stats.json
  results/r1_9_leakage_per_model.csv
  results/r1_9_leakage_per_fold.csv
  results/r1_9_summary.md

NO manuscript edits; pure analysis.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

DATA_PATH = SCRIPT_DIR / "previous_code" / "UV_canonical_v3_dedup.csv"
RESULTS = SCRIPT_DIR / "results"
FOLD_NPZ = RESULTS / "cv_fold_indices_v3.npz"
SEED = 7
N_FOLDS = 5

# Model keys whose per-fold v3 predictions live in results/
MODELS = ["rf_tuned", "xgboost", "chemprop", "bigru_tuned", "chemberta"]


def load_v3_data():
    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "solvents", "lambda_max"]].dropna().reset_index(drop=True)
    print(f"  v3 rows: {len(df)}, unique canon solutes: {df['canon'].nunique()}, "
          f"unique solvents: {df['solvents'].nunique()}")
    return df


def load_folds():
    if not FOLD_NPZ.exists():
        sys.exit(f"missing {FOLD_NPZ} — run retrain_v3.py first")
    splits = np.load(FOLD_NPZ)
    folds = []
    for i in range(N_FOLDS):
        folds.append((splits[f"train_{i}"], splits[f"val_{i}"], splits[f"test_{i}"]))
    return folds


def load_pred(model_key, fold):
    name = f"{model_key}_v3_fold{fold}"
    y_pred = np.load(RESULTS / f"{name}_predictions.npy")
    y_test = np.load(RESULTS / f"{name}_y_test.npy")
    test_idx = np.load(RESULTS / f"{name}_test_indices.npy")
    return y_pred, y_test, test_idx


def metrics(y_true, y_pred):
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    return {
        "n": int(len(y_true)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def main():
    print("[LOAD] v3 dataset and fold splits...")
    df = load_v3_data()
    folds = load_folds()
    canon = df["canon"].values

    # For each fold, identify which test records' solute appears in train
    # under a different solvent.
    fold_leakage = []
    for i, (tr, va, te) in enumerate(folds):
        # "Train pool" = the 64% train slice (matches what RF/XGBoost fit on).
        # We also report the 80% train+val slice (matches what DL models see).
        train_solutes = set(canon[tr])
        trainval_solutes = set(canon[tr]) | set(canon[va])
        n_test = len(te)
        # Boolean masks aligned with test_idx in the original fold order
        in_train = np.array([canon[idx] in train_solutes for idx in te])
        in_trainval = np.array([canon[idx] in trainval_solutes for idx in te])
        # "Leaked" = solute appears in train (under any solvent) — same chromophore
        # is therefore not strictly novel to the model.
        n_leaked_train = int(in_train.sum())
        n_leaked_trainval = int(in_trainval.sum())
        n_novel_train = n_test - n_leaked_train
        n_novel_trainval = n_test - n_leaked_trainval
        # Also report unique-solute counts
        test_solutes = set(canon[te])
        n_test_unique_solutes = len(test_solutes)
        n_leaked_unique = len(test_solutes & train_solutes)
        n_novel_unique = n_test_unique_solutes - n_leaked_unique

        print(f"  fold {i}: test_records={n_test}, "
              f"unique_solutes={n_test_unique_solutes}, "
              f"leaked vs train={n_leaked_train}/{n_test} "
              f"({100*n_leaked_train/n_test:.1f}%), "
              f"leaked vs train+val={n_leaked_trainval}/{n_test} "
              f"({100*n_leaked_trainval/n_test:.1f}%), "
              f"unique-solute leakage={n_leaked_unique}/{n_test_unique_solutes} "
              f"({100*n_leaked_unique/n_test_unique_solutes:.1f}%)")

        fold_leakage.append({
            "fold": i,
            "n_test_records": n_test,
            "n_test_unique_solutes": n_test_unique_solutes,
            "n_leaked_records_train": n_leaked_train,
            "n_novel_records_train": n_novel_train,
            "leak_rate_train": round(n_leaked_train / n_test, 4),
            "n_leaked_records_trainval": n_leaked_trainval,
            "n_novel_records_trainval": n_novel_trainval,
            "leak_rate_trainval": round(n_leaked_trainval / n_test, 4),
            "n_leaked_unique_solutes": n_leaked_unique,
            "n_novel_unique_solutes": n_novel_unique,
            "in_train_mask": in_train,
            "in_trainval_mask": in_trainval,
        })

    # For each model, compute (leaked, novel) RMSE/MAE per fold using the
    # appropriate train-pool mask (train-only for RF/XGB; train+val for DL).
    print("\n[METRICS] per model, per fold, per leakage-split...")
    rows = []
    for m in MODELS:
        # Pick mask based on whether the model uses val (DL) or not (FP)
        mask_kind = "in_train_mask" if m in ("rf_tuned", "xgboost") else "in_trainval_mask"
        for fl in fold_leakage:
            i = fl["fold"]
            try:
                y_pred, y_test, test_idx = load_pred(m, i)
            except FileNotFoundError:
                print(f"  [SKIP] {m} fold {i} prediction files missing")
                continue
            assert len(test_idx) <= fl["n_test_records"], "test_idx mismatch"
            # Some models (RF, Chemprop) skip a handful of invalid SMILES; their
            # test_idx is a strict subset of the fold's test indices. Recompute
            # the leakage mask aligned with the actual test_idx the model used.
            tr_idx = folds[i][0]
            va_idx = folds[i][1]
            train_pool = set(canon[tr_idx]) if mask_kind == "in_train_mask" \
                         else (set(canon[tr_idx]) | set(canon[va_idx]))
            mask = np.array([canon[ix] in train_pool for ix in test_idx])
            n_leaked = int(mask.sum())
            n_novel = int((~mask).sum())
            m_leaked = metrics(y_test[mask], y_pred[mask]) if n_leaked else {"n": 0}
            m_novel  = metrics(y_test[~mask], y_pred[~mask]) if n_novel else {"n": 0}
            m_all    = metrics(y_test, y_pred)
            rows.append({
                "model": m, "fold": i,
                "mask_used": "train" if mask_kind == "in_train_mask" else "train+val",
                "n_total_test": len(test_idx),
                "n_leaked": n_leaked, "leak_rate": round(n_leaked / len(test_idx), 4),
                "rmse_leaked": m_leaked.get("RMSE", np.nan),
                "mae_leaked": m_leaked.get("MAE", np.nan),
                "r2_leaked": m_leaked.get("R2", np.nan),
                "n_novel": n_novel,
                "rmse_novel": m_novel.get("RMSE", np.nan),
                "mae_novel": m_novel.get("MAE", np.nan),
                "r2_novel": m_novel.get("R2", np.nan),
                "rmse_all": m_all["RMSE"], "mae_all": m_all["MAE"], "r2_all": m_all["R2"],
            })

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(RESULTS / "r1_9_leakage_per_fold.csv", index=False)
    print(f"\n[OUT] {RESULTS/'r1_9_leakage_per_fold.csv'}")

    # Aggregate per model
    agg = (df_rows.groupby("model")
                  .agg(leak_rate=("leak_rate", "mean"),
                       rmse_leaked_mean=("rmse_leaked", "mean"),
                       rmse_leaked_std=("rmse_leaked", "std"),
                       rmse_novel_mean=("rmse_novel", "mean"),
                       rmse_novel_std=("rmse_novel", "std"),
                       mae_leaked_mean=("mae_leaked", "mean"),
                       mae_novel_mean=("mae_novel", "mean"),
                       rmse_all_mean=("rmse_all", "mean"))
                  .reset_index())
    agg["delta_rmse"] = (agg["rmse_novel_mean"] - agg["rmse_leaked_mean"]).round(2)
    agg["delta_mae"]  = (agg["mae_novel_mean"]  - agg["mae_leaked_mean"]).round(2)
    for c in agg.columns:
        if c.endswith("_mean") or c.endswith("_std") or c.startswith("rmse_") or c.startswith("mae_"):
            agg[c] = agg[c].round(2)
        if c == "leak_rate":
            agg[c] = agg[c].round(4)
    agg.to_csv(RESULTS / "r1_9_leakage_per_model.csv", index=False)
    print(f"[OUT] {RESULTS/'r1_9_leakage_per_model.csv'}")

    # Summary stats for JSON
    overall_leak_train = float(np.mean([fl["leak_rate_train"] for fl in fold_leakage]))
    overall_leak_trainval = float(np.mean([fl["leak_rate_trainval"] for fl in fold_leakage]))
    summary = {
        "v3_total_records": int(len(df)),
        "v3_unique_solutes": int(df["canon"].nunique()),
        "v3_unique_solvents": int(df["solvents"].nunique()),
        "n_folds": N_FOLDS,
        "overall_leak_rate_train_only": round(overall_leak_train, 4),
        "overall_leak_rate_trainval": round(overall_leak_trainval, 4),
        "per_fold": [
            {k: v for k, v in fl.items() if not k.endswith("_mask")}
            for fl in fold_leakage
        ],
        "per_model": agg.to_dict(orient="records"),
    }
    (RESULTS / "r1_9_leakage_stats.json").write_text(json.dumps(summary, indent=2))
    print(f"[OUT] {RESULTS/'r1_9_leakage_stats.json'}")

    # Markdown summary
    md = []
    md.append("# R1-9 Step 0 — Chromophore-Solvent Leakage Audit\n")
    md.append("_v3 cleaned Joung+Beard primary dataset, 5-fold stratified CV_\n")
    md.append(f"## Headline numbers\n")
    md.append(f"- v3 unique solutes: **{df['canon'].nunique()}**, unique solvents: **{df['solvents'].nunique()}**, records: **{len(df)}**")
    md.append(f"- mean leakage rate (test records whose solute also appears in train *only* slice): **{100*overall_leak_train:.1f}%**")
    md.append(f"- mean leakage rate (test records whose solute also appears in train *or* val slice): **{100*overall_leak_trainval:.1f}%**")
    md.append("")
    md.append("## Per-fold breakdown\n")
    md.append("| fold | n test | unique solutes in test | leaked vs train | leaked vs train+val |")
    md.append("|---|---|---|---|---|")
    for fl in fold_leakage:
        md.append(
            f"| {fl['fold']} | {fl['n_test_records']} | {fl['n_test_unique_solutes']} | "
            f"{fl['n_leaked_records_train']} ({100*fl['leak_rate_train']:.1f}%) | "
            f"{fl['n_leaked_records_trainval']} ({100*fl['leak_rate_trainval']:.1f}%) |"
        )
    md.append("")
    md.append("## Per-model leakage effect (RMSE)\n")
    md.append("| model | leak rate | leaked RMSE | novel RMSE | ΔRMSE (novel − leaked) |")
    md.append("|---|---|---|---|---|")
    for _, r in agg.iterrows():
        md.append(
            f"| {r['model']} | {100*r['leak_rate']:.1f}% | "
            f"{r['rmse_leaked_mean']:.2f} $\\pm$ {r['rmse_leaked_std']:.2f} | "
            f"{r['rmse_novel_mean']:.2f} $\\pm$ {r['rmse_novel_std']:.2f} | "
            f"{r['delta_rmse']:+.2f} |"
        )
    md.append("")
    # Decision bucket per user spec
    md.append("## Decision-rule bucketing (per-model)\n")
    md.append("Per user's spec: ΔRMSE = (novel − leaked); large → ≥ 2 nm; bounded → 0.5–2 nm; small → < 0.5 nm.\n")
    md.append("| model | ΔRMSE | bucket |")
    md.append("|---|---|---|")
    for _, r in agg.iterrows():
        d = float(r["delta_rmse"])
        if d >= 2.0:
            bucket = "**LARGE**"
        elif 0.5 <= d < 2.0:
            bucket = "BOUNDED"
        else:
            bucket = "SMALL"
        md.append(f"| {r['model']} | {d:+.2f} | {bucket} |")
    (RESULTS / "r1_9_summary.md").write_text("\n".join(md))
    print(f"[OUT] {RESULTS/'r1_9_summary.md'}")


if __name__ == "__main__":
    main()
