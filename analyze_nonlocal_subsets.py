#!/usr/bin/env python3
"""
R1-5 — Non-local subset analysis for the JCIM revision.

Identifies subsets of the Joung+Beard primary dataset where non-local effects
(donor-acceptor charge transfer, low-gap π systems) are expected to dominate,
and computes per-model error metrics on those subsets across the 5 CV folds.

Subsets:
  * full        — entire dataset (sanity check / reference)
  * lambda500   — λ_max > 500 nm (long-wavelength: low-gap chromophores)
  * lambda600   — λ_max > 600 nm (deep-coloured / NIR-pushing)
  * azo         — contains an aromatic azo bridge (Ar-N=N-Ar)
  * da          — molecule contains BOTH a strong π-donor AND a strong
                  π-acceptor on aromatic positions (donor-acceptor candidate)
  * dad         — ≥2 distinct donor matches AND ≥1 acceptor (D-A-D candidate)
  * bodipy      — BODIPY core (4,4-difluoro-4-bora-3a,4a-diaza-s-indacene)
  * cyanine     — open-chain cyanine-like motif
  * non_local   — union(lambda500, azo, da, dad, bodipy, cyanine)
                  (any compound where non-locality is plausibly important)

For each (model, subset) pair the script reports mean ± std RMSE / MAE / R²
across the 5 folds, restricted to test molecules that fall into the subset.

Per-fold predictions are loaded from
  results/{model_key}_fold{i}_predictions.npy
  results/{model_key}_fold{i}_y_test.npy
  results/{model_key}_fold{i}_test_indices.npy

For XGBoost the saved joblib models are scored against the deterministic
stratified fold splits.

Outputs (results/nonlocal_subsets/):
  - subset_membership.csv         per-molecule subset flags
  - nonlocal_metrics_full.csv     long-form metrics (model × subset × fold)
  - nonlocal_metrics_summary.csv  mean ± std table
  - subset_counts.csv             subset sizes
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")  # silence parse warnings

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "previous_code" / "UV_canonical_full_dataset.csv"
RESULTS_DIR = SCRIPT_DIR / "results"
OUT_DIR = RESULTS_DIR / "nonlocal_subsets"
OUT_DIR.mkdir(exist_ok=True)

SEED = 7
N_FOLDS = 5

# ──────────────────────────────────────────────────────────────────────────────
# SMARTS-based subset classification
# ──────────────────────────────────────────────────────────────────────────────

# π-Donor groups: lone-pair-bearing substituents on aromatic carbon that push
# electron density into the π system. Exclude amides (acyl-N), nitro-N, etc.
DONOR_SMARTS = [
    ("aryl_NH2",       "[c]-[NX3;H2;!$(N(C=O));!$(N-N=O)]"),
    ("aryl_NHR",       "[c]-[NX3;H1;!$(N-C=O);!$(N-N=O);!$(N-S(=O))]"),
    ("aryl_NR2",       "[c]-[NX3;H0;!$(N-C=O);!$(N-N=O);!$(N-S(=O))]([#6])[#6]"),
    ("aryl_OH",        "[c]-[OX2H1]"),
    ("aryl_OR",        "[c]-[OX2;H0]-[CX4]"),
    ("aryl_SH",        "[c]-[SX2H1]"),
    ("aryl_SR",        "[c]-[SX2;H0]-[CX4]"),
]

# π-Acceptor groups: strongly electron-withdrawing substituents conjugated
# into a π system (aromatic *or* alkene/aryl sp2 carbon). We accept attachment
# to either an aromatic carbon [c] or an sp2 carbon participating in a C=C
# double bond, so push-pull cinnamic acids (ferulic, coumaric) are captured.
SP2_ANCHOR = "[$([c]),$([CX3](=[#6,#7])[!#1])]"

ACCEPTOR_SMARTS = [
    ("nitro",          f"{SP2_ANCHOR}-[$([NX3+](=O)[O-]),$([NX3](=O)=O)]"),
    ("CN",             f"{SP2_ANCHOR}-[CX2]#[NX1]"),
    ("CHO",            f"{SP2_ANCHOR}-[CX3H1](=O)"),
    ("ketone",         f"{SP2_ANCHOR}-[CX3](=O)[#6]"),
    ("COOH",           f"{SP2_ANCHOR}-[CX3](=O)[OX2H1]"),
    ("COOR",           f"{SP2_ANCHOR}-[CX3](=O)[OX2H0][#6]"),
    ("amide",          f"{SP2_ANCHOR}-[CX3](=O)[NX3]"),
    ("SO2",            f"{SP2_ANCHOR}-[SX4](=O)(=O)"),
    ("CF3",            f"{SP2_ANCHOR}-[CX4](F)(F)F"),
    ("aromatic_N+",    "[c][n+,N+;X3]"),         # pyridinium / acridinium
    ("dicyanovinyl",   "[#6]=C([CX2]#[NX1])[CX2]#[NX1]"),
    ("malononitrile",  "[CX4H1]([CX2]#N)[CX2]#N"),
]

# Specific dye motifs known to require non-local description
MOTIF_SMARTS = {
    # Azo bridge between two aromatic systems (any geometry)
    "azo":     "[c]-[N;X2]=[N;X2]-[c]",
    # BODIPY core: BF2 bridging two ring nitrogens (very permissive — captures
    # aromatic and kekulised representations of the dipyrromethene scaffold)
    "bodipy":  "[B](F)(F)([#7;R])[#7;R]",
    # Open-chain cyanine: polyene between two amines, terminal positive charge
    "cyanine": "[NX3,nX3]-,=[CX3]=,-[CX3]-,=[CX3]=,-[CX3]-,=[NX3+,nX3+]",
    # Squaraine: 1,3-substituted cyclobutene-3,4-diiminium core
    "squaraine": "C1(=O)C(=*)C(=O)C1=*",
    # E-stilbene-like 1,2-disubstituted alkene between two aryl groups
    "stilbene": "[c]-[CH]=[CH]-[c]",
}


def _compile(smarts_list):
    out = []
    for name, smarts in smarts_list:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            print(f"  [WARN] failed to parse SMARTS for {name}: {smarts}", file=sys.stderr)
        else:
            out.append((name, patt))
    return out


def _compile_dict(d):
    out = {}
    for k, smarts in d.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            print(f"  [WARN] failed to parse motif SMARTS {k}: {smarts}", file=sys.stderr)
        else:
            out[k] = patt
    return out


DONOR_PATTERNS = _compile(DONOR_SMARTS)
ACCEPTOR_PATTERNS = _compile(ACCEPTOR_SMARTS)
MOTIF_PATTERNS = _compile_dict(MOTIF_SMARTS)


def _count_unique_matches(mol, patterns):
    """Return total count of distinct atom-set matches across all patterns.

    A donor match here is counted once per *match*; two methoxys on the same
    ring count as two donor matches (we want this for D-A-D where >=2 donors
    is meaningful)."""
    total = 0
    per_rule = {}
    for name, patt in patterns:
        matches = mol.GetSubstructMatches(patt, uniquify=True)
        per_rule[name] = len(matches)
        total += len(matches)
    return total, per_rule


def classify_molecule(smiles):
    """Return dict of subset flags + per-rule counts for one SMILES."""
    out = {
        "n_donor": 0, "n_acceptor": 0,
        "is_da": False, "is_dad": False,
        "has_azo": False, "has_bodipy": False, "has_cyanine": False,
        "has_squaraine": False, "has_stilbene": False,
        "valid": False,
    }
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return out
    out["valid"] = True
    n_d, _ = _count_unique_matches(mol, DONOR_PATTERNS)
    n_a, _ = _count_unique_matches(mol, ACCEPTOR_PATTERNS)
    out["n_donor"] = n_d
    out["n_acceptor"] = n_a
    out["is_da"] = (n_d >= 1) and (n_a >= 1)
    out["is_dad"] = (n_d >= 2) and (n_a >= 1)
    for motif, patt in MOTIF_PATTERNS.items():
        out[f"has_{motif}"] = mol.HasSubstructMatch(patt)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Subset assembly
# ──────────────────────────────────────────────────────────────────────────────

SUBSET_DEFS = [
    "full", "lambda500", "lambda600",
    "azo", "da", "dad", "bodipy", "cyanine", "squaraine", "stilbene",
    "non_local",
]


def build_subset_membership(df):
    """Return DataFrame indexed like df with a boolean column per subset."""
    print(f"\n[CLASSIFY] Computing SMARTS membership for {len(df)} molecules...")
    rows = []
    for smi in df["canon"].values:
        rows.append(classify_molecule(smi))
    cls = pd.DataFrame(rows, index=df.index)

    out = pd.DataFrame(index=df.index)
    out["lambda_max"] = df["lambda_max"].values
    out["canon"] = df["canon"].values

    out["full"]      = True
    out["lambda500"] = df["lambda_max"].values > 500
    out["lambda600"] = df["lambda_max"].values > 600
    out["azo"]       = cls["has_azo"].values
    out["da"]        = cls["is_da"].values
    out["dad"]       = cls["is_dad"].values
    out["bodipy"]    = cls["has_bodipy"].values
    out["cyanine"]   = cls["has_cyanine"].values
    out["squaraine"] = cls["has_squaraine"].values
    out["stilbene"]  = cls["has_stilbene"].values

    # Composite: ANY non-local indicator
    out["non_local"] = (
        out["lambda500"] | out["azo"] | out["da"] | out["dad"]
        | out["bodipy"] | out["cyanine"] | out["squaraine"]
    )

    # Persist for downstream inspection
    out.to_csv(OUT_DIR / "subset_membership.csv", index=False)

    # Per-subset counts and pairwise overlaps
    counts = []
    for s in SUBSET_DEFS:
        counts.append({"subset": s, "n": int(out[s].sum()),
                       "pct": round(100 * out[s].mean(), 2)})
    pd.DataFrame(counts).to_csv(OUT_DIR / "subset_counts.csv", index=False)

    print("[CLASSIFY] subset sizes:")
    for c in counts:
        print(f"  {c['subset']:>12s}: {c['n']:>6d}  ({c['pct']:>5.2f}%)")

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-fold metric computation
# ──────────────────────────────────────────────────────────────────────────────

def _safe_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    return {
        "n": int(len(y_true)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def compute_subset_metrics(model_key, fold_predictions, subset_member):
    """fold_predictions: list of (test_idx, y_test, y_pred) tuples (length N_FOLDS).

    subset_member: DataFrame with boolean subset columns indexed by molecule
                   index (0..N-1 in the full dataset).
    """
    rows = []
    for fold, (test_idx, y_test, y_pred) in enumerate(fold_predictions):
        assert len(test_idx) == len(y_test) == len(y_pred), \
            f"fold {fold} length mismatch"
        sub_local = subset_member.loc[test_idx]  # rows of subset flags
        for s in SUBSET_DEFS:
            mask = sub_local[s].values
            if mask.sum() < 1:
                rows.append({"model": model_key, "subset": s, "fold": fold,
                             "n": 0, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan})
                continue
            m = _safe_metrics(y_test[mask], y_pred[mask])
            m.update({"model": model_key, "subset": s, "fold": fold})
            rows.append(m)
    return rows


def load_fold_predictions(model_key, results_dir=RESULTS_DIR):
    """Load per-fold .npy files for a model. Returns list of (test_idx, y_test, y_pred)
    or None if any fold file is missing."""
    out = []
    for i in range(N_FOLDS):
        p_pred = results_dir / f"{model_key}_fold{i}_predictions.npy"
        p_yt = results_dir / f"{model_key}_fold{i}_y_test.npy"
        p_idx = results_dir / f"{model_key}_fold{i}_test_indices.npy"
        if not (p_pred.exists() and p_yt.exists() and p_idx.exists()):
            print(f"  [SKIP] {model_key}: missing fold {i} files")
            return None
        out.append((np.load(p_idx), np.load(p_yt), np.load(p_pred)))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# XGBoost inference path: regenerate predictions from saved joblibs
# ──────────────────────────────────────────────────────────────────────────────

def regen_xgboost_predictions(global_data, folds, variant="xgboost_v2"):
    """Score the saved {variant}_fold{i}_model.joblib on each test fold."""
    import joblib
    from paper1_new_cl.models import compute_morgan_fps

    print(f"\n[XGBOOST] Regenerating predictions from saved {variant}_fold*_model.joblib...")
    # Build full-dataset Morgan FP (solute + solvent, 2048 each)
    solute = global_data["smiles_solute"]
    solvent = global_data["solvents"]
    print(f"  computing solute FPs...")
    fp_solute, mask_s = compute_morgan_fps(list(solute), radius=2, n_bits=2048)
    print(f"  computing solvent FPs...")
    fp_solvent, mask_v = compute_morgan_fps(list(solvent), radius=2, n_bits=2048)
    fp_all = np.hstack([fp_solute, fp_solvent])
    print(f"  X shape: {fp_all.shape}")

    y_all = global_data["y"]
    out = []
    for i, (_train, _val, test_idx) in enumerate(folds):
        model_path = RESULTS_DIR / f"{variant}_fold{i}_model.joblib"
        if not model_path.exists():
            print(f"  [SKIP] {variant} fold {i}: {model_path.name} missing")
            return None
        model = joblib.load(model_path)
        y_pred = model.predict(fp_all[test_idx])
        out.append((test_idx, y_all[test_idx], y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_all[test_idx], y_pred)))
        print(f"  fold {i}: n_test={len(test_idx):>5d}  RMSE={rmse:.2f}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def load_global_data():
    print(f"\n[DATA] {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "solvents", "lambda_max"]].dropna().reset_index(drop=True)
    print(f"  rows: {len(df)}")
    return {
        "smiles_solute": df["canon"].values,
        "solvents": df["solvents"].values,
        "y": df["lambda_max"].values.astype(float),
        "df": df,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "rf_tuned", "xgboost_v2",
        "chemprop_v2", "bigru_tuned", "chemberta",
    ])
    ap.add_argument("--regen_xgboost", action="store_true",
                    help="Run XGBoost inference from saved joblibs")
    args = ap.parse_args()

    global_data = load_global_data()

    # Recreate the stratified fold splits deterministically
    sys.path.insert(0, str(SCRIPT_DIR))
    from paper1_new_cl.splits import create_stratified_folds
    print(f"\n[FOLDS] Recreating deterministic StratifiedKFold splits (seed={SEED})")
    folds = create_stratified_folds(
        n_samples=len(global_data["y"]),
        solvents=global_data["solvents"],
        n_folds=N_FOLDS, seed=SEED,
    )
    for i, (tr, va, te) in enumerate(folds):
        print(f"  fold {i}: train={len(tr):>5d}, val={len(va):>5d}, test={len(te):>5d}")

    # Sanity check: union of test folds should equal range(N)
    pooled_test = np.concatenate([te for _, _, te in folds])
    assert sorted(pooled_test.tolist()) == list(range(len(global_data["y"]))), \
        "Test folds do not partition dataset!"
    print(f"  test-fold partition OK ({len(pooled_test)} total)")

    # Compute subset membership for full dataset (indexed 0..N-1)
    sub = build_subset_membership(global_data["df"])

    # Score each requested model
    all_rows = []
    for model_key in args.models:
        print(f"\n[MODEL] {model_key}")
        if model_key.startswith("xgboost") and args.regen_xgboost:
            preds = regen_xgboost_predictions(global_data, folds, variant=model_key)
        else:
            preds = load_fold_predictions(model_key)
        if preds is None:
            print(f"  [WARN] {model_key}: predictions not available — skipping")
            continue
        rows = compute_subset_metrics(model_key, preds, sub)
        all_rows.extend(rows)

    if not all_rows:
        print("\n[ERROR] No model results to summarise. Did you rsync the prediction files?")
        sys.exit(1)

    full = pd.DataFrame(all_rows)
    full.to_csv(OUT_DIR / "nonlocal_metrics_full.csv", index=False)

    # Mean ± std across folds
    summary = (full.groupby(["model", "subset"])
                   .agg(n=("n", "mean"),
                        RMSE_mean=("RMSE", "mean"),
                        RMSE_std=("RMSE", "std"),
                        MAE_mean=("MAE", "mean"),
                        MAE_std=("MAE", "std"),
                        R2_mean=("R2", "mean"),
                        R2_std=("R2", "std"))
                   .reset_index())
    summary["n"] = summary["n"].round(0).astype(int)
    for c in summary.columns:
        if c.endswith("_mean") or c.endswith("_std"):
            summary[c] = summary[c].round(3)
    summary.to_csv(OUT_DIR / "nonlocal_metrics_summary.csv", index=False)

    print(f"\n[OUT] wrote:")
    print(f"  {OUT_DIR/'nonlocal_metrics_full.csv'}")
    print(f"  {OUT_DIR/'nonlocal_metrics_summary.csv'}")
    print(f"  {OUT_DIR/'subset_counts.csv'}")
    print(f"  {OUT_DIR/'subset_membership.csv'}")

    # Pretty print summary
    pivot_rmse = summary.pivot(index="subset", columns="model", values="RMSE_mean")
    pivot_n = summary.pivot(index="subset", columns="model", values="n").iloc[:, 0]
    print(f"\n[SUMMARY] RMSE (mean across folds), per (model × subset):")
    print(pivot_rmse.round(1).to_string())
    print(f"\nSubset sizes (test molecules per fold averaged):")
    print(pivot_n.to_string())


if __name__ == "__main__":
    main()
