#!/usr/bin/env python3
"""
External validation of trained UV λ_max prediction models (5-fold CV ensemble).

Evaluates saved Keras models (all 5 folds) on external datasets (ChemFluor, or any
user-provided CSV) that were NOT used during training. Handles:
  - Preprocessing raw external data (solvent name → SMILES mapping)
  - Deduplication against training set
  - SMILES vectorization using saved charset/config
  - Ensemble prediction (average of 5 fold models) + uncertainty (std)
  - Parity plot for external predictions

Usage:
  python3 eval_external.py                              # evaluate on all available
  python3 eval_external.py --dataset chemfluor           # ChemFluor only
  python3 eval_external.py --dataset dsscdb              # DSSCDB only (if available)
  python3 eval_external.py --dataset combined            # both combined
  python3 eval_external.py --model-key bigru_solvent     # specify model (default)
  python3 eval_external.py --skip-preprocess             # skip preprocessing step
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from paper1_new_cl.models import compute_metrics, vectorize_smiles

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "previous_code", "UV_canonical_full_dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

N_FOLDS = 5

# ─── Solvent name → canonical SMILES mapping ─────────────────────────────────
# Covers all 63 ChemFluor solvents + common variants

SOLVENT_NAME_TO_SMILES = {
    # Alcohols
    "methanol": "CO", "meoh": "CO",
    "ethanol": "CCO", "etoh": "CCO",
    "1-propanol": "CCCO", "n-propanol": "CCCO",
    "2-propanol": "CC(O)C", "isopropanol": "CC(O)C", "ipa": "CC(O)C",
    "1-butanol": "CCCCO", "n-butanol": "CCCCO",
    "1-hexanol": "CCCCCCO",
    "1-octanol": "CCCCCCCCO",
    "1-decanol": "CCCCCCCCCCO",
    "2-methyl-2-propanol": "CC(C)(C)O", "tert-butanol": "CC(C)(C)O",
    "tert-pentanol": "CCC(C)(C)O",
    "ethylene glycol": "OCCO",
    "1,2-propanediol": "CC(O)CO",
    "glycerol": "OCC(O)CO",
    "tfe": "OCC(F)(F)F", "2,2,2-trifluoroethanol": "OCC(F)(F)F",
    # Water
    "water": "O", "h2o": "O",
    # Ethers
    "diethyl ether": "CCOCC", "et2o": "CCOCC",
    "thf": "C1CCOC1", "tetrahydrofuran": "C1CCOC1",
    "me-thf": "CC1CCCO1", "2-methyltetrahydrofuran": "CC1CCCO1",
    "dioxane": "C1COCCO1", "1,4-dioxane": "C1COCCO1",
    "di-n-butyl ether": "CCCCOCCCC",
    "diisopropyl ether": "CC(C)OC(C)C",
    # Halogenated
    "ch2cl2": "ClCCl", "dcm": "ClCCl", "dichloromethane": "ClCCl",
    "chcl3": "ClC(Cl)Cl", "chloroform": "ClC(Cl)Cl",
    "ccl4": "ClC(Cl)(Cl)Cl", "carbon tetrachloride": "ClC(Cl)(Cl)Cl",
    "chlorobenzene": "Clc1ccccc1",
    "bromobenzene": "Brc1ccccc1",
    "dcb": "Clc1ccccc1Cl", "1,2-dichlorobenzene": "Clc1ccccc1Cl",
    # Aromatics
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "o-dimethoxybenzene": "COc1ccccc1OC", "1,2-dimethoxybenzene": "COc1ccccc1OC",
    # Ketones
    "acetone": "CC(C)=O",
    "2-pentanone": "CCCC(C)=O",
    # Nitriles
    "acetonitrile": "CC#N", "mecn": "CC#N", "acn": "CC#N",
    # Amides / polar aprotic
    "dmf": "CN(C)C=O", "dimethylformamide": "CN(C)C=O",
    "dmso": "CS(C)=O", "dimethyl sulfoxide": "CS(C)=O",
    "dma": "CC(=O)N(C)C", "dimethylacetamide": "CC(=O)N(C)C",
    "formamide": "NC=O",
    "n-methylformamide": "CNC=O",
    "1-methyl-2-pyrrolidinone": "CN1CCCC1=O", "nmp": "CN1CCCC1=O",
    # Esters
    "ethyl acetate": "CCOC(C)=O", "etoac": "CCOC(C)=O",
    "methyl acetate": "COC(C)=O",
    "methyl formate": "COC=O",
    "butyl acetate": "CCCCOC(C)=O",
    # Hydrocarbons
    "hexane": "CCCCCC", "n-hexane": "CCCCCC",
    "heptane": "CCCCCCC", "n-heptane": "CCCCCCC",
    "cyclohexane": "C1CCCCC1",
    "methylcyclohexane": "CC1CCCCC1",
    "2-methylbutane": "CCC(C)C", "isopentane": "CCC(C)C",
    # Amines
    "triethylamine": "CCN(CC)CC",
    "pyridine": "c1ccncc1",
}


def map_solvent_name_to_smiles(name):
    """Map a solvent name to its canonical SMILES. Returns None if unmappable."""
    key = name.strip().lower()
    return SOLVENT_NAME_TO_SMILES.get(key)


# ─── Preprocessing: ChemFluor ────────────────────────────────────────────────

def preprocess_chemfluor():
    """Read raw ChemFluor xlsx and produce a cleaned CSV."""
    raw_path = os.path.join(RAW_DIR, "chemfluor.xlsx")
    out_path = os.path.join(DATA_DIR, "chemfluor_processed.csv")

    if not os.path.exists(raw_path):
        print(f"  [SKIP] ChemFluor raw file not found at {raw_path}")
        print(f"  Download from: https://ndownloader.figshare.com/files/24007736")
        return None

    print("  Preprocessing ChemFluor...")
    df = pd.read_excel(raw_path, header=0)

    # Extract relevant columns (positions based on xlsx inspection)
    df_clean = pd.DataFrame({
        "smiles": df.iloc[:, 4],        # SMILES column (E)
        "absorption_nm": df.iloc[:, 1],  # Absorption/nm (B)
        "solvent_name": df.iloc[:, 5],   # solvent (F)
    })

    # Drop rows without absorption data
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=["absorption_nm", "smiles", "solvent_name"])
    print(f"  Dropped {before - len(df_clean)} rows with missing data")

    # Skip solvent mixtures (contain ":")
    mixture_mask = df_clean["solvent_name"].str.contains(":", na=False)
    n_mix = mixture_mask.sum()
    if n_mix > 0:
        df_clean = df_clean[~mixture_mask]
        print(f"  Dropped {n_mix} solvent mixture entries")

    # Map solvent names to SMILES
    df_clean["solvent_smiles"] = df_clean["solvent_name"].apply(map_solvent_name_to_smiles)
    unmapped = df_clean["solvent_smiles"].isna()
    if unmapped.any():
        unmapped_names = df_clean.loc[unmapped, "solvent_name"].unique()
        print(f"  WARNING: {unmapped.sum()} entries with unmapped solvents: {unmapped_names}")
        df_clean = df_clean[~unmapped]

    # Canonicalize SMILES with RDKit
    from rdkit import Chem
    canon_smiles = []
    valid_mask = []
    for smi in df_clean["smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            canon_smiles.append(Chem.MolToSmiles(mol))
            valid_mask.append(True)
        else:
            canon_smiles.append(None)
            valid_mask.append(False)

    df_clean["smiles"] = canon_smiles
    n_invalid = sum(not v for v in valid_mask)
    if n_invalid > 0:
        df_clean = df_clean[pd.Series(valid_mask, index=df_clean.index)]
        print(f"  Dropped {n_invalid} invalid SMILES")

    # Also canonicalize solvent SMILES for consistency
    canon_solvents = []
    for smi in df_clean["solvent_smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            canon_solvents.append(Chem.MolToSmiles(mol))
        else:
            canon_solvents.append(smi)
    df_clean["solvent_smiles"] = canon_solvents

    # Output
    result = df_clean[["smiles", "solvent_smiles", "absorption_nm"]].rename(
        columns={"absorption_nm": "lambda_max"}
    )
    result.to_csv(out_path, index=False)
    print(f"  Saved {len(result)} entries to {out_path}")
    return out_path


def preprocess_dsscdb():
    """Placeholder for DSSCDB preprocessing."""
    raw_path = os.path.join(RAW_DIR, "dsscdb.csv")
    out_path = os.path.join(DATA_DIR, "dsscdb_processed.csv")

    if not os.path.exists(raw_path):
        print("  [SKIP] DSSCDB raw file not found.")
        print("  Note: dyedb.com is offline. Contact original authors for data.")
        return None

    # If user provides a CSV with columns: smiles, solvent_smiles, lambda_max
    # we just validate and pass through
    print("  Preprocessing DSSCDB...")
    df = pd.read_csv(raw_path)
    required = {"smiles", "solvent_smiles", "lambda_max"}
    if not required.issubset(set(df.columns)):
        # Try to map from common column names
        print(f"  Expected columns: {required}")
        print(f"  Got columns: {set(df.columns)}")
        print("  Please provide a CSV with columns: smiles, solvent_smiles, lambda_max")
        return None

    from rdkit import Chem
    canon = []
    valid = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            canon.append(Chem.MolToSmiles(mol))
            valid.append(True)
        else:
            canon.append(None)
            valid.append(False)
    df["smiles"] = canon
    df = df[pd.Series(valid, index=df.index)]
    df = df.dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} entries to {out_path}")
    return out_path


# ─── Load external dataset ───────────────────────────────────────────────────

def load_external_dataset(name):
    """Load a preprocessed external dataset CSV."""
    path = os.path.join(DATA_DIR, f"{name}_processed.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


# ─── Deduplication against training data ──────────────────────────────────────

def remove_training_overlap(external_df, dataset_name):
    """Remove molecules that appear in the training dataset."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_smiles = set(train_df["canon"].dropna().values)

    before = len(external_df)
    mask = ~external_df["smiles"].isin(train_smiles)
    external_df = external_df[mask].copy()
    n_overlap = before - len(external_df)
    print(f"  Removed {n_overlap} overlapping molecules from {dataset_name} "
          f"({before} → {len(external_df)})")
    return external_df


# ─── Ensemble model loading + prediction ─────────────────────────────────────

def load_ensemble(model_key, n_folds=N_FOLDS):
    """Load all fold models + shared config. Returns (list_of_models, config)."""
    config_path = os.path.join(RESULTS_DIR, f"{model_key}_config.json")
    if not os.path.exists(config_path):
        print(f"  ERROR: No config found at {config_path}")
        print(f"  Run: python3 run_baselines.py --model {model_key}")
        return None, None

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "bigru")

    # GPU / CUDA setup
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_cuda_data_dir=/home/umesh/.local/cuda_compat"
    )
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    models = []
    if model_type in ("bigru", "bilstm", "cnn_bigru"):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        for i in range(n_folds):
            model_path = os.path.join(RESULTS_DIR, f"{model_key}_fold{i}_model.keras")
            if not os.path.exists(model_path):
                print(f"  [WARN] Missing fold {i} model: {model_path}")
                continue
            print(f"  Loading fold {i} model from {model_path}...")
            model = tf.keras.models.load_model(model_path)
            models.append(model)

    elif model_type in ("rf", "xgb"):
        import joblib
        for i in range(n_folds):
            model_path = os.path.join(RESULTS_DIR, f"{model_key}_fold{i}_model.joblib")
            if not os.path.exists(model_path):
                print(f"  [WARN] Missing fold {i} model: {model_path}")
                continue
            print(f"  Loading fold {i} model from {model_path}...")
            model = joblib.load(model_path)
            models.append(model)

    if not models:
        print(f"  ERROR: No fold models found for '{model_key}'")
        return None, None

    print(f"  Loaded {len(models)}/{n_folds} fold models")
    return models, config


def predict_external_ensemble(models, config, external_df):
    """Predict with each fold model, return mean prediction and per-sample std."""
    char_to_int = config["char_to_int"]
    embed_len = config["embed_len"]
    with_solvent = config["with_solvent"]
    model_type = config.get("model_type", "bigru")

    # Build combined SMILES
    if with_solvent:
        combined = (external_df["smiles"] + "!" + external_df["solvent_smiles"]).values
    else:
        combined = external_df["smiles"].values

    # Check for characters not in charset
    charset_set = set(config["charset"])
    oov_count = 0
    for smi in combined:
        for c in smi:
            if c not in charset_set:
                oov_count += 1
                break
    if oov_count > 0:
        print(f"  WARNING: {oov_count}/{len(combined)} sequences contain out-of-vocabulary characters")

    # Check for sequences that exceed embed_len
    max_len = max(len(s) for s in combined)
    if max_len + 2 > embed_len:
        print(f"  WARNING: {sum(len(s) + 2 > embed_len for s in combined)} sequences "
              f"exceed max training length ({embed_len}), will be truncated")

    if model_type in ("bigru", "bilstm", "cnn_bigru"):
        # Vectorize for DL models
        X = vectorize_smiles(combined, char_to_int, embed_len)

        # Predict with each fold model
        all_preds = []
        for i, model in enumerate(models):
            y_pred = model.predict(X, verbose=0).flatten()
            all_preds.append(y_pred)
            print(f"    Fold {i} predicted ({len(y_pred)} samples)")

    elif model_type in ("rf", "xgb"):
        # Compute fingerprints for FP models
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from tqdm import tqdm

        solute_list = external_df["smiles"].values
        solvent_list = external_df["solvent_smiles"].values if with_solvent else [""] * len(external_df)

        print("  Computing fingerprints for external data...")
        fps_solute = []
        for smi in tqdm(solute_list, desc="    Solute FP", leave=False):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps_solute.append(np.array(fp))
            else:
                fps_solute.append(np.zeros(2048))

        fps_solvent = []
        for smi in tqdm(solvent_list, desc="    Solvent FP", leave=False):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps_solvent.append(np.array(fp))
            else:
                fps_solvent.append(np.zeros(2048))

        X = np.hstack([np.array(fps_solute), np.array(fps_solvent)])

        all_preds = []
        for i, model in enumerate(models):
            y_pred = model.predict(X).flatten()
            all_preds.append(y_pred)
            print(f"    Fold {i} predicted ({len(y_pred)} samples)")

    all_preds = np.array(all_preds)  # shape: (n_folds, n_samples)
    y_pred_mean = all_preds.mean(axis=0)
    y_pred_std = all_preds.std(axis=0)

    print(f"  Ensemble: {len(models)} models, mean uncertainty (std): {y_pred_std.mean():.2f} nm")
    return y_pred_mean, y_pred_std


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_dataset(models, config, dataset_name, deduplicate=True):
    """Load, deduplicate, predict with ensemble, and score an external dataset."""
    df = load_external_dataset(dataset_name)
    if df is None:
        print(f"\n  [{dataset_name}] No processed data found. Skipping.")
        return None

    print(f"\n{'─'*60}")
    print(f"  Evaluating on: {dataset_name} ({len(df)} entries)")
    print(f"{'─'*60}")

    if deduplicate:
        df = remove_training_overlap(df, dataset_name)

    if len(df) == 0:
        print(f"  No entries remaining after deduplication!")
        return None

    y_true = df["lambda_max"].values
    y_pred_mean, y_pred_std = predict_external_ensemble(models, config, df)

    metrics = compute_metrics(y_true, y_pred_mean)
    print(f"  Results: RMSE={metrics['RMSE']}  MAE={metrics['MAE']}  "
          f"R²={metrics['R2']}  r={metrics['Pearson_r']}")
    print(f"  N={len(y_true)}, mean ensemble std={y_pred_std.mean():.2f} nm")

    # Save results
    result = {
        "dataset": dataset_name,
        "n_samples": len(y_true),
        "n_ensemble_models": len(models),
        "mean_ensemble_std": round(float(y_pred_std.mean()), 2),
        **metrics
    }
    result_path = os.path.join(RESULTS_DIR, f"external_{dataset_name}_metrics.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved metrics to {result_path}")

    # Save predictions with uncertainty
    np.savez(os.path.join(RESULTS_DIR, f"external_{dataset_name}_predictions.npz"),
             y_true=y_true, y_pred=y_pred_mean, y_pred_std=y_pred_std)

    return {
        "metrics": metrics, "y_true": y_true, "y_pred": y_pred_mean,
        "y_pred_std": y_pred_std, "n_samples": len(y_true), "name": dataset_name
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_external_parity(results_list):
    """Generate parity plot for external validation results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_plots = len(results_list)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5), squeeze=False)

    for idx, res in enumerate(results_list):
        ax = axes[0, idx]
        y_true, y_pred = res["y_true"], res["y_pred"]
        y_std = res.get("y_pred_std")
        rmse = res["metrics"]["RMSE"]
        name = res["name"]
        n = res["n_samples"]

        ax.scatter(y_true, y_pred, alpha=0.3, s=10, c="steelblue", edgecolors="none")
        if y_std is not None:
            # Show error bars for ensemble uncertainty (subsample for readability)
            step = max(1, len(y_true) // 200)
            ax.errorbar(y_true[::step], y_pred[::step], yerr=y_std[::step],
                        fmt="none", ecolor="gray", alpha=0.2, lw=0.5)

        lims = [min(y_true.min(), y_pred.min()) - 10,
                max(y_true.max(), y_pred.max()) + 10]
        ax.plot(lims, lims, "k--", lw=1, label="Ideal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Experimental $\\lambda_{\\max}$ (nm)", fontsize=11)
        ax.set_ylabel("Predicted $\\lambda_{\\max}$ (nm)", fontsize=11)
        ax.set_title(f"{name} (RMSE={rmse:.1f} nm, N={n})", fontsize=12)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, "external_parity_plot.png")
    out_pdf = os.path.join(RESULTS_DIR, "external_parity_plot.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"\n  Parity plot saved to {out_png}")


# ─── Summary table ────────────────────────────────────────────────────────────

def save_comparison_table(results_list):
    """Save comparison CSV with all external evaluation results."""
    if not results_list:
        return

    rows = []
    for res in results_list:
        rows.append({
            "Dataset": res["name"],
            "N": res["n_samples"],
            **res["metrics"]
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "external_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Comparison table saved to {out_path}")
    print(f"\n{df.to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="External validation of UV λ_max models (ensemble)")
    parser.add_argument("--dataset", type=str, choices=["chemfluor", "dsscdb", "combined"],
                        help="Which external dataset to evaluate on (default: all available)")
    parser.add_argument("--model-key", type=str, default="bigru_solvent",
                        help="Model key to load (default: bigru_solvent)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing step (use existing processed CSVs)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip deduplication against training data")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  EXTERNAL VALIDATION — UV λ_max Prediction (5-Fold Ensemble)")
    print("=" * 60)

    # Step 1: Preprocess external datasets
    if not args.skip_preprocess:
        print("\n[1/4] Preprocessing external datasets...")
        preprocess_chemfluor()
        preprocess_dsscdb()
    else:
        print("\n[1/4] Skipping preprocessing (--skip-preprocess)")

    # Step 2: Load ensemble (all fold models)
    print(f"\n[2/4] Loading ensemble for '{args.model_key}'...")
    models, config = load_ensemble(args.model_key)
    if models is None:
        sys.exit(1)
    print(f"  Model type: {config.get('model_type', 'unknown')}")
    print(f"  With solvent: {config.get('with_solvent', 'unknown')}")
    print(f"  Charset size: {len(config.get('charset', []))}")
    print(f"  Embed length: {config.get('embed_len', 'unknown')}")
    print(f"  Ensemble size: {len(models)} fold models")

    # Step 3: Evaluate
    print(f"\n[3/4] Evaluating on external data...")
    datasets_to_eval = []
    if args.dataset is None or args.dataset == "chemfluor":
        datasets_to_eval.append("chemfluor")
    if args.dataset is None or args.dataset == "dsscdb":
        datasets_to_eval.append("dsscdb")

    results = []
    for ds_name in datasets_to_eval:
        res = evaluate_dataset(models, config, ds_name, deduplicate=not args.no_dedup)
        if res is not None:
            results.append(res)

    # Combined evaluation (if both exist and requested)
    if (args.dataset is None or args.dataset == "combined") and len(results) >= 2:
        print(f"\n{'─'*60}")
        print("  Evaluating on: combined")
        print(f"{'─'*60}")
        combined_true = np.concatenate([r["y_true"] for r in results])
        combined_pred = np.concatenate([r["y_pred"] for r in results])
        combined_std = np.concatenate([r["y_pred_std"] for r in results])
        combined_metrics = compute_metrics(combined_true, combined_pred)
        combined_n = len(combined_true)
        print(f"  Results: RMSE={combined_metrics['RMSE']}  MAE={combined_metrics['MAE']}  "
              f"R²={combined_metrics['R2']}  r={combined_metrics['Pearson_r']}")
        print(f"  N={combined_n}")

        combined_result = {
            "metrics": combined_metrics, "y_true": combined_true, "y_pred": combined_pred,
            "y_pred_std": combined_std, "n_samples": combined_n, "name": "combined"
        }
        with open(os.path.join(RESULTS_DIR, "external_combined_metrics.json"), "w") as f:
            json.dump({"dataset": "combined", "n_samples": combined_n, **combined_metrics}, f, indent=2)
        results.append(combined_result)

    if not results:
        print("\n  No external datasets available for evaluation.")
        print("  Ensure data/chemfluor_processed.csv or data/dsscdb_processed.csv exists.")
        sys.exit(1)

    # Step 4: Generate plots and summary
    print(f"\n[4/4] Generating outputs...")
    plot_external_parity(results)
    save_comparison_table(results)

    print("\n" + "=" * 60)
    print("  EXTERNAL VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
