#!/usr/bin/env python3
"""
Experimental validation: predict UV λ_max for wetlab-measured molecules.

Loads trained BiGRU+Solvent v2 and RF v2 fold models, predicts λ_max for
16 molecules measured in EtOH and MeOH (32 data points total), and compares
against experimental measurements.

Usage:
  python3 eval_wetlab.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# ─── GPU / CUDA setup ────────────────────────────────────────────────────────
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/home/umesh/.local/cuda_compat"
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "wetlab_experimental.csv")
N_FOLDS = 5


def load_wetlab_data():
    """Load wetlab experimental CSV."""
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} wetlab measurements ({df['molecule'].nunique()} molecules)")
    return df


def predict_bigru(df):
    """Predict λ_max using BiGRU+Solvent v2 ensemble (average across folds)."""
    import tensorflow as tf

    # Load config
    config_path = os.path.join(RESULTS_DIR, "bigru_solvent_v2_config.json")
    with open(config_path) as f:
        config = json.load(f)

    char_to_int = config["char_to_int"]
    embed_len = config["embed_len"]

    from paper1_new_cl.models import vectorize_smiles

    # Construct combined SMILES
    combined = (df["smiles"] + "!" + df["solvent_smiles"]).values

    # Check charset coverage
    all_chars = set("".join(combined))
    missing = all_chars - set(char_to_int.keys())
    if missing:
        print(f"  WARNING: Characters not in charset: {missing}")
        return None

    # Vectorize
    X = vectorize_smiles(combined, char_to_int, embed_len)

    # Load and predict with each fold model
    all_preds = []
    for fold in range(N_FOLDS):
        model_path = os.path.join(RESULTS_DIR, f"bigru_solvent_v2_fold{fold}_model.keras")
        if not os.path.exists(model_path):
            print(f"  WARNING: Missing fold {fold} model: {model_path}")
            continue
        model = tf.keras.models.load_model(model_path)
        preds = model.predict(X, verbose=0).flatten()
        all_preds.append(preds)
        del model
        tf.keras.backend.clear_session()

    if not all_preds:
        print("  ERROR: No BiGRU fold models found!")
        return None

    # Ensemble average
    ensemble_pred = np.mean(all_preds, axis=0)
    print(f"  BiGRU: ensembled {len(all_preds)} fold models")
    return ensemble_pred


def predict_rf(df):
    """Predict λ_max using RF v2 ensemble (average across folds)."""
    import joblib
    from paper1_new_cl.models import compute_morgan_fps

    # Compute Morgan FPs: solute + solvent concatenated
    solute_fps, solute_mask = compute_morgan_fps(df["smiles"].tolist())
    solvent_fps, solvent_mask = compute_morgan_fps(df["solvent_smiles"].tolist())
    fp_all = np.hstack([solute_fps, solvent_fps])
    valid_mask = solute_mask & solvent_mask

    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  WARNING: {n_invalid} molecules failed FP computation")

    # Load and predict with each fold model
    all_preds = []
    for fold in range(N_FOLDS):
        model_path = os.path.join(RESULTS_DIR, f"rf_tuned_fold{fold}_model.joblib")
        if not os.path.exists(model_path):
            print(f"  WARNING: Missing fold {fold} model: {model_path}")
            continue
        model = joblib.load(model_path)
        preds = model.predict(fp_all)
        all_preds.append(preds)

    if not all_preds:
        print("  ERROR: No RF fold models found!")
        return None

    ensemble_pred = np.mean(all_preds, axis=0)
    print(f"  RF: ensembled {len(all_preds)} fold models")
    return ensemble_pred


def predict_chemberta(df):
    """Predict λ_max using ChemBERTa ensemble (average across folds)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Combined SMILES with "." separator
    combined = (df["smiles"] + "." + df["solvent_smiles"]).values.tolist()

    # Tokenize
    encodings = tokenizer(
        combined, add_special_tokens=True, max_length=256,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    all_preds = []
    for fold in range(N_FOLDS):
        model_path = os.path.join(RESULTS_DIR, f"chemberta_fold{fold}_best.pt")
        if not os.path.exists(model_path):
            print(f"  WARNING: Missing fold {fold} model: {model_path}")
            continue

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=1, problem_type="regression"
        )
        ckpt = torch.load(model_path, weights_only=False, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.squeeze(-1).cpu().numpy()

        all_preds.append(preds)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not all_preds:
        print("  ERROR: No ChemBERTa fold models found!")
        return None

    ensemble_pred = np.mean(all_preds, axis=0)
    print(f"  ChemBERTa: ensembled {len(all_preds)} fold models")
    return ensemble_pred


def check_training_overlap(df):
    """Check which wetlab molecules appear in the training dataset."""
    train_path = os.path.join(SCRIPT_DIR, "previous_code", "UV_canonical_full_dataset.csv")
    train_df = pd.read_csv(train_path)

    overlaps = []
    for _, row in df.drop_duplicates("smiles").iterrows():
        matches = train_df[train_df["canon"] == row["smiles"]]
        if len(matches) > 0:
            solvents = matches["solvents"].unique()
            lmax_vals = matches["lambda_max"].values
            overlaps.append({
                "molecule": row["molecule"],
                "smiles": row["smiles"],
                "n_train_entries": len(matches),
                "train_solvents": list(solvents),
                "train_lmax": list(lmax_vals),
            })

    if overlaps:
        print(f"\n  Training data overlap ({len(overlaps)} molecules):")
        for o in overlaps:
            print(f"    {o['molecule']}: {o['n_train_entries']} entries, "
                  f"solvents={o['train_solvents']}, λ_max={o['train_lmax']}")
    else:
        print("\n  No training data overlap — all molecules are out-of-distribution")

    return overlaps


def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, and per-sample errors."""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(abs_errors)
    return rmse, mae, errors, abs_errors


def main():
    print(f"\n{'=' * 70}")
    print(f"  WETLAB EXPERIMENTAL VALIDATION")
    print(f"{'=' * 70}")

    df = load_wetlab_data()

    # Check training overlap
    overlaps = check_training_overlap(df)

    y_true = df["lambda_max_exp"].values

    # Predict with all models
    print(f"\n{'─' * 70}")
    print("  Loading BiGRU+Solvent v2 models...")
    bigru_pred = predict_bigru(df)

    print(f"\n{'─' * 70}")
    print("  Loading RF v2 models...")
    rf_pred = predict_rf(df)

    print(f"\n{'─' * 70}")
    print("  Loading ChemBERTa models...")
    chemberta_pred = predict_chemberta(df)

    # Build results table
    results = df[["molecule", "solvent_name", "lambda_max_exp"]].copy()

    if bigru_pred is not None:
        results["BiGRU_pred"] = np.round(bigru_pred, 1)
        results["BiGRU_error"] = np.round(bigru_pred - y_true, 1)

    if rf_pred is not None:
        results["RF_pred"] = np.round(rf_pred, 1)
        results["RF_error"] = np.round(rf_pred - y_true, 1)

    if chemberta_pred is not None:
        results["ChemBERTa_pred"] = np.round(chemberta_pred, 1)
        results["ChemBERTa_error"] = np.round(chemberta_pred - y_true, 1)

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(results.to_string(index=False))

    # Aggregate metrics
    print(f"\n{'─' * 70}")
    print("  AGGREGATE METRICS")
    print(f"{'─' * 70}")

    for name, pred in [("BiGRU+Solvent v2", bigru_pred), ("RF v2", rf_pred),
                        ("ChemBERTa", chemberta_pred)]:
        if pred is None:
            continue
        rmse, mae, errors, abs_errors = compute_metrics(y_true, pred)
        print(f"\n  {name}:")
        print(f"    Overall:  RMSE={rmse:.2f} nm, MAE={mae:.2f} nm")

        # By solvent
        for solvent in ["EtOH", "MeOH"]:
            mask = df["solvent_name"].values == solvent
            s_rmse, s_mae, _, _ = compute_metrics(y_true[mask], pred[mask])
            print(f"    {solvent}:     RMSE={s_rmse:.2f} nm, MAE={s_mae:.2f} nm")

        # Exclude training overlap molecules
        overlap_smiles = {o["smiles"] for o in overlaps}
        ood_mask = ~df["smiles"].isin(overlap_smiles).values
        if ood_mask.sum() < len(ood_mask):
            ood_rmse, ood_mae, _, _ = compute_metrics(y_true[ood_mask], pred[ood_mask])
            print(f"    OOD only: RMSE={ood_rmse:.2f} nm, MAE={ood_mae:.2f} nm "
                  f"(N={ood_mask.sum()}, excl. training overlap)")

    # Solvent sensitivity analysis
    print(f"\n{'─' * 70}")
    print("  SOLVENT SENSITIVITY (|EtOH - MeOH|)")
    print(f"{'─' * 70}")

    molecules = df["molecule"].unique()
    sens_rows = []
    for mol in molecules:
        mol_data = df[df["molecule"] == mol]
        etoh = mol_data[mol_data["solvent_name"] == "EtOH"]
        meoh = mol_data[mol_data["solvent_name"] == "MeOH"]
        if len(etoh) == 1 and len(meoh) == 1:
            exp_diff = abs(etoh["lambda_max_exp"].values[0] - meoh["lambda_max_exp"].values[0])
            row = {"Molecule": mol, "Exp |Δ|": exp_diff}

            etoh_idx = etoh.index[0]
            meoh_idx = meoh.index[0]

            if bigru_pred is not None:
                bigru_diff = abs(bigru_pred[etoh_idx] - bigru_pred[meoh_idx])
                row["BiGRU |Δ|"] = round(bigru_diff, 1)

            if rf_pred is not None:
                rf_diff = abs(rf_pred[etoh_idx] - rf_pred[meoh_idx])
                row["RF |Δ|"] = round(rf_diff, 1)

            if chemberta_pred is not None:
                cb_diff = abs(chemberta_pred[etoh_idx] - chemberta_pred[meoh_idx])
                row["ChemBERTa |Δ|"] = round(cb_diff, 1)

            sens_rows.append(row)

    sens_df = pd.DataFrame(sens_rows)
    print(sens_df.to_string(index=False))

    # Save results
    out_path = os.path.join(RESULTS_DIR, "wetlab_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    # Save detailed JSON
    summary = {
        "n_molecules": int(df["molecule"].nunique()),
        "n_predictions": len(df),
        "solvents": ["EtOH", "MeOH"],
    }
    for name, pred in [("bigru", bigru_pred), ("rf", rf_pred),
                        ("chemberta", chemberta_pred)]:
        if pred is None:
            continue
        rmse, mae, _, _ = compute_metrics(y_true, pred)
        summary[f"{name}_rmse"] = round(float(rmse), 2)
        summary[f"{name}_mae"] = round(float(mae), 2)

    json_path = os.path.join(RESULTS_DIR, "wetlab_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")

    print(f"\n{'=' * 70}")
    print(f"  WETLAB VALIDATION COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
