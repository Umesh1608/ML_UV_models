#!/usr/bin/env python3
"""
RF hyperparameter search using fold 0 validation set from v2 stratified splits.

Searches over: n_estimators, max_depth, min_samples_leaf, max_features,
               Morgan FP radius, Morgan FP n_bits.

Usage:
  python3 tune_rf.py              # Run grid search
  python3 tune_rf.py --full-cv    # After search: run best config on all 5 folds
"""

import argparse
import itertools
import json
import os
import time
import warnings

# Suppress RDKit deprecation warnings
warnings.filterwarnings("ignore", message=".*DEPRECATION.*MorganGenerator.*")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from paper1_new_cl.models import compute_metrics, compute_morgan_fps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_PATH = os.path.join(SCRIPT_DIR, "previous_code", "UV_canonical_full_dataset.csv")
SEED = 42

# ─── Hyperparameter grid ─────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators": [300, 500, 1000],
    "max_depth": [None, 20, 30, 50],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2", 0.3],
}

FP_GRID = {
    "radius": [2, 3],
    "n_bits": [1024, 2048],
}


def load_data():
    """Load dataset and fold 0 splits."""
    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "solvents", "lambda_max"]].dropna().reset_index(drop=True)
    splits = np.load(os.path.join(RESULTS_DIR, "cv_fold_indices_stratified.npz"))
    print(f"  Dataset: {len(df)} rows")
    return df, splits


def compute_fps(smiles_solute, smiles_solvent, radius, n_bits):
    """Compute solute + solvent concatenated Morgan FPs."""
    solute_fps, solute_mask = compute_morgan_fps(smiles_solute, radius=radius, n_bits=n_bits)
    solvent_fps, solvent_mask = compute_morgan_fps(smiles_solvent, radius=radius, n_bits=n_bits)
    fp_all = np.hstack([solute_fps, solvent_fps])
    valid_mask = solute_mask & solvent_mask
    return fp_all, valid_mask


def run_search():
    """Grid search over RF hyperparameters using fold 0 train/val split."""
    df, splits = load_data()
    y = df["lambda_max"].values

    train_idx = splits["train_0"]
    val_idx = splits["val_0"]

    print(f"  Fold 0: {len(train_idx)} train, {len(val_idx)} val")
    print(f"  RF param grid: {sum(1 for _ in itertools.product(*PARAM_GRID.values()))} configs")
    print(f"  FP grid: {sum(1 for _ in itertools.product(*FP_GRID.values()))} configs")

    # Total configs
    rf_combos = list(itertools.product(*PARAM_GRID.values()))
    fp_combos = list(itertools.product(*FP_GRID.values()))
    total = len(rf_combos) * len(fp_combos)
    print(f"  Total configurations: {total}\n")

    results = []
    best_rmse = float("inf")
    best_config = None

    for fp_i, (radius, n_bits) in enumerate(fp_combos):
        print(f"{'='*60}")
        print(f"  FP: radius={radius}, n_bits={n_bits}")
        print(f"{'='*60}")

        fp_all, valid_mask = compute_fps(df["canon"].tolist(), df["solvents"].tolist(), radius, n_bits)

        # Get valid indices for this FP config
        train_valid = train_idx[valid_mask[train_idx]]
        val_valid = val_idx[valid_mask[val_idx]]

        X_train = fp_all[train_valid]
        y_train = y[train_valid]
        X_val = fp_all[val_valid]
        y_val = y[val_valid]

        for rf_i, (n_est, max_d, min_leaf, max_feat) in enumerate(rf_combos):
            config = {
                "radius": radius,
                "n_bits": n_bits,
                "n_estimators": n_est,
                "max_depth": max_d,
                "min_samples_leaf": min_leaf,
                "max_features": max_feat,
            }

            t0 = time.time()
            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_leaf=min_leaf,
                max_features=max_feat,
                n_jobs=-1,
                random_state=SEED,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            elapsed = time.time() - t0

            metrics = compute_metrics(y_val, y_pred)

            config_idx = fp_i * len(rf_combos) + rf_i + 1
            marker = ""
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_config = config.copy()
                marker = " *** BEST ***"

            print(f"  [{config_idx:3d}/{total}] n_est={n_est:4d} depth={str(max_d):>4s} "
                  f"leaf={min_leaf} feat={str(max_feat):>4s} | "
                  f"RMSE={metrics['RMSE']:6.2f} MAE={metrics['MAE']:6.2f} "
                  f"R²={metrics['R2']:.4f} ({elapsed:.0f}s){marker}")

            config["RMSE"] = metrics["RMSE"]
            config["MAE"] = metrics["MAE"]
            config["R2"] = metrics["R2"]
            config["Pearson_r"] = metrics["Pearson_r"]
            config["time_s"] = round(elapsed, 1)
            results.append(config)

    # Save results
    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(os.path.join(RESULTS_DIR, "rf_hyperparam_search.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"  BEST CONFIGURATION (val RMSE = {best_rmse:.2f})")
    print(f"{'='*60}")
    for k, v in best_config.items():
        print(f"    {k}: {v}")

    # Save best config
    with open(os.path.join(RESULTS_DIR, "rf_best_config.json"), "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  Saved to results/rf_best_config.json")

    # Show top 10
    print(f"\n  Top 10 configurations:")
    print(results_df.head(10).to_string(index=False))

    return best_config


def run_full_cv(config=None):
    """Run full 5-fold CV with the best config."""
    if config is None:
        config_path = os.path.join(RESULTS_DIR, "rf_best_config.json")
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Loaded config from {config_path}")

    print(f"\n  Config: {config}")

    df, splits = load_data()
    y = df["lambda_max"].values

    radius = config["radius"]
    n_bits = config["n_bits"]
    fp_all, valid_mask = compute_fps(df["canon"].tolist(), df["solvents"].tolist(), radius, n_bits)

    fold_metrics = []
    for fold in range(5):
        train_idx = splits[f"train_{fold}"]
        test_idx = splits[f"test_{fold}"]

        train_valid = train_idx[valid_mask[train_idx]]
        test_valid = test_idx[valid_mask[test_idx]]

        X_train = fp_all[train_valid]
        y_train = y[train_valid]
        X_test = fp_all[test_valid]
        y_test = y[test_valid]

        t0 = time.time()
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config.get("max_depth"),
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            n_jobs=-1,
            random_state=SEED,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - t0

        metrics = compute_metrics(y_test, y_pred)
        fold_metrics.append(metrics)

        print(f"  Fold {fold}: RMSE={metrics['RMSE']:6.2f} MAE={metrics['MAE']:6.2f} "
              f"R²={metrics['R2']:.4f} r={metrics['Pearson_r']:.4f} ({elapsed:.0f}s)")

        # Save fold results
        prefix = f"rf_tuned_fold{fold}"
        with open(os.path.join(RESULTS_DIR, f"{prefix}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        np.save(os.path.join(RESULTS_DIR, f"{prefix}_predictions.npy"), y_pred)
        np.save(os.path.join(RESULTS_DIR, f"{prefix}_y_test.npy"), y_test)
        np.save(os.path.join(RESULTS_DIR, f"{prefix}_test_indices.npy"), test_valid)

        import joblib
        joblib.dump(model, os.path.join(RESULTS_DIR, f"{prefix}_model.joblib"))

    # Aggregate
    agg = {}
    for key in ["RMSE", "MAE", "R2", "Pearson_r"]:
        vals = [m[key] for m in fold_metrics]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 2)
        agg[f"{key}_std"] = round(float(np.std(vals)), 2)

    agg["config"] = config
    with open(os.path.join(RESULTS_DIR, "rf_tuned_cv_aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  AGGREGATE (5-fold CV)")
    print(f"{'='*60}")
    print(f"  RMSE: {agg['RMSE_mean']:.2f} ± {agg['RMSE_std']:.2f}")
    print(f"  MAE:  {agg['MAE_mean']:.2f} ± {agg['MAE_std']:.2f}")
    print(f"  R²:   {agg['R2_mean']:.4f} ± {agg['R2_std']:.4f}")
    print(f"  r:    {agg['Pearson_r_mean']:.4f} ± {agg['Pearson_r_std']:.4f}")

    print(f"\n  Previous RF v2:  RMSE 32.18 ± 1.74, MAE 15.42 ± 0.47")
    print(f"  Tuned RF:        RMSE {agg['RMSE_mean']:.2f} ± {agg['RMSE_std']:.2f}, "
          f"MAE {agg['MAE_mean']:.2f} ± {agg['MAE_std']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF hyperparameter search")
    parser.add_argument("--full-cv", action="store_true",
                        help="Run full 5-fold CV with best config")
    args = parser.parse_args()

    if args.full_cv:
        run_full_cv()
    else:
        run_search()
