#!/usr/bin/env python3
"""
Download and preprocess UV absorption datasets for cross-dataset benchmarking.

Deep4Chem (Joung 2020):
  ~20K samples, CSV from Figshare.
  Columns: Chromophore (SMILES), Solvent (SMILES), Absorption max (nm)

Jung et al. 2024:
  ~26K experimental samples, Excel from GitHub.
  Columns: smiles, solvent (SMILES), peakwavs_max

nablaColors (Potapov et al. 2026):
  ~24.6K absorption pairs, CSVs from Zenodo.
  Pre-defined scaffold split (train/val/test).
  Columns: key, smiles, solvent, peakwavs_max

Usage:
  python3 download_datasets.py                          # download + preprocess all
  python3 download_datasets.py --dataset deep4chem      # Deep4Chem only
  python3 download_datasets.py --dataset jung2024       # Jung 2024 only
  python3 download_datasets.py --dataset nablacolors    # nablaColors only
  python3 download_datasets.py --skip-download          # preprocess from cached files
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ─── Download URLs ──────────────────────────────────────────────────────────

DEEP4CHEM_URL = "https://ndownloader.figshare.com/files/49887156"
DEEP4CHEM_RAW = os.path.join(RAW_DIR, "deep4chem_raw.csv")

JUNG2024_URL = "https://raw.githubusercontent.com/Songyosk/UVVIS/main/Data/uvvis_datasets.xlsx"
JUNG2024_RAW = os.path.join(RAW_DIR, "jung2024_raw.xlsx")

NABLACOLORS_BASE_URL = "https://zenodo.org/records/18061300/files"
NABLACOLORS_DIR = os.path.join(RAW_DIR, "nablacolors")
NABLACOLORS_FILES = [
    "absorption_pairs_all.csv",
    "absorption_train.csv",
    "absorption_val.csv",
    "absorption_test.csv",
]


def download_file(url, dest, description="file"):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        print(f"  [CACHED] {description} already at {dest}")
        return True

    print(f"  Downloading {description}...")
    print(f"    URL: {url}")
    print(f"    Dest: {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"    Done ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


# ─── Deep4Chem preprocessing ───────────────────────────────────────────────

def preprocess_deep4chem():
    """Read raw Deep4Chem CSV and produce a cleaned CSV."""
    from rdkit import Chem

    out_path = os.path.join(DATA_DIR, "deep4chem_processed.csv")

    print("\n[Deep4Chem] Preprocessing...")
    df = pd.read_csv(DEEP4CHEM_RAW)
    print(f"  Raw: {len(df)} rows, columns: {list(df.columns)}")

    # Select relevant columns
    df = df[["Chromophore", "Solvent", "Absorption max (nm)"]].copy()
    df.columns = ["smiles", "solvent_smiles", "lambda_max"]

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    print(f"  After dropna: {len(df)} (dropped {before - len(df)})")

    # Filter out non-SMILES solvent entries (e.g., "Solid", "Gas", "Film")
    non_smiles = {"solid", "gas", "film", "neat", "crystal", "powder"}
    mask = ~df["solvent_smiles"].str.strip().str.lower().isin(non_smiles)
    n_removed = (~mask).sum()
    if n_removed > 0:
        df = df[mask]
        print(f"  Removed {n_removed} non-solution entries (Solid/Gas/Film/etc.)")

    # Canonicalize SMILES with RDKit
    canon_smiles = []
    canon_solvents = []
    valid = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        sol = Chem.MolFromSmiles(str(row["solvent_smiles"]))
        if mol is not None and sol is not None:
            canon_smiles.append(Chem.MolToSmiles(mol))
            canon_solvents.append(Chem.MolToSmiles(sol))
            valid.append(True)
        else:
            canon_smiles.append(None)
            canon_solvents.append(None)
            valid.append(False)

    n_invalid = sum(not v for v in valid)
    if n_invalid > 0:
        print(f"  Dropped {n_invalid} invalid SMILES")

    df["smiles"] = canon_smiles
    df["solvent_smiles"] = canon_solvents
    df = df[pd.Series(valid, index=df.index)]

    # Convert lambda_max to float
    df["lambda_max"] = pd.to_numeric(df["lambda_max"], errors="coerce")
    df = df.dropna(subset=["lambda_max"])

    # Save
    df = df.reset_index(drop=True)
    df.to_csv(out_path, index=False)

    # Stats
    print(f"\n  [Deep4Chem] Saved {len(df)} entries to {out_path}")
    print(f"  Unique solutes: {df['smiles'].nunique()}")
    print(f"  Unique solvents: {df['solvent_smiles'].nunique()}")
    print(f"  λ_max range: {df['lambda_max'].min():.1f} – {df['lambda_max'].max():.1f} nm")
    print(f"  λ_max mean: {df['lambda_max'].mean():.1f} nm, std: {df['lambda_max'].std():.1f} nm")

    return out_path


# ─── Jung 2024 preprocessing ──────────────────────────────────────────────

def preprocess_jung2024():
    """Read raw Jung 2024 Excel and produce a cleaned CSV (experimental only)."""
    from rdkit import Chem

    out_path = os.path.join(DATA_DIR, "jung2024_processed.csv")

    print("\n[Jung 2024] Preprocessing...")
    df = pd.read_excel(JUNG2024_RAW, sheet_name="Experimental")
    print(f"  Raw: {len(df)} rows, columns: {list(df.columns)}")

    # Select relevant columns
    df = df[["smiles", "solvent", "peakwavs_max"]].copy()
    df.columns = ["smiles", "solvent_smiles", "lambda_max"]

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    print(f"  After dropna: {len(df)} (dropped {before - len(df)})")

    # Canonicalize SMILES with RDKit
    canon_smiles = []
    canon_solvents = []
    valid = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        sol = Chem.MolFromSmiles(str(row["solvent_smiles"]))
        if mol is not None and sol is not None:
            canon_smiles.append(Chem.MolToSmiles(mol))
            canon_solvents.append(Chem.MolToSmiles(sol))
            valid.append(True)
        else:
            canon_smiles.append(None)
            canon_solvents.append(None)
            valid.append(False)

    n_invalid = sum(not v for v in valid)
    if n_invalid > 0:
        print(f"  Dropped {n_invalid} invalid SMILES")

    df["smiles"] = canon_smiles
    df["solvent_smiles"] = canon_solvents
    df = df[pd.Series(valid, index=df.index)]

    # Convert lambda_max to float
    df["lambda_max"] = pd.to_numeric(df["lambda_max"], errors="coerce")
    df = df.dropna(subset=["lambda_max"])

    # Save
    df = df.reset_index(drop=True)
    df.to_csv(out_path, index=False)

    # Stats
    print(f"\n  [Jung 2024] Saved {len(df)} entries to {out_path}")
    print(f"  Unique solutes: {df['smiles'].nunique()}")
    print(f"  Unique solvents: {df['solvent_smiles'].nunique()}")
    print(f"  λ_max range: {df['lambda_max'].min():.1f} – {df['lambda_max'].max():.1f} nm")
    print(f"  λ_max mean: {df['lambda_max'].mean():.1f} nm, std: {df['lambda_max'].std():.1f} nm")

    return out_path


# ─── nablaColors preprocessing ───────────────────────────────────────────────

def preprocess_nablacolors():
    """Read raw nablaColors CSVs and produce a cleaned CSV + split index mapping."""
    from rdkit import Chem

    out_path = os.path.join(DATA_DIR, "nablacolors_processed.csv")

    print("\n[nablaColors] Preprocessing...")

    # Load all pairs
    all_path = os.path.join(NABLACOLORS_DIR, "absorption_pairs_all.csv")
    df = pd.read_csv(all_path)
    print(f"  Raw (all pairs): {len(df)} rows, columns: {list(df.columns)}")

    # Rename columns to standard format
    df = df.rename(columns={"solvent": "solvent_smiles", "peakwavs_max": "lambda_max"})

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    print(f"  After dropna: {len(df)} (dropped {before - len(df)})")

    # Canonicalize SMILES with RDKit
    canon_smiles = []
    canon_solvents = []
    valid = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        sol = Chem.MolFromSmiles(str(row["solvent_smiles"]))
        if mol is not None and sol is not None:
            canon_smiles.append(Chem.MolToSmiles(mol))
            canon_solvents.append(Chem.MolToSmiles(sol))
            valid.append(True)
        else:
            canon_smiles.append(None)
            canon_solvents.append(None)
            valid.append(False)

    n_invalid = sum(not v for v in valid)
    if n_invalid > 0:
        print(f"  Dropped {n_invalid} invalid SMILES")

    df["smiles"] = canon_smiles
    df["solvent_smiles"] = canon_solvents
    df = df[pd.Series(valid, index=df.index)]

    # Convert lambda_max to float
    df["lambda_max"] = pd.to_numeric(df["lambda_max"], errors="coerce")
    df = df.dropna(subset=["lambda_max"])

    # Save processed dataset
    df = df.reset_index(drop=True)
    df.to_csv(out_path, index=False)

    # Build split index mapping from pre-defined train/val/test CSVs
    # Match rows by (smiles, solvent_smiles, lambda_max) tuple
    split_path = os.path.join(DATA_DIR, "nablacolors_splits.npz")
    train_df = pd.read_csv(os.path.join(NABLACOLORS_DIR, "absorption_train.csv"))
    val_df = pd.read_csv(os.path.join(NABLACOLORS_DIR, "absorption_val.csv"))
    test_df = pd.read_csv(os.path.join(NABLACOLORS_DIR, "absorption_test.csv"))

    # Create lookup key from processed df
    df["_key"] = df["smiles"] + "|" + df["solvent_smiles"] + "|" + df["lambda_max"].astype(str)

    def get_split_indices(split_df, processed_df):
        """Map split CSV rows to indices in the processed dataframe."""
        indices = []
        # Canonicalize the split SMILES to match processed df
        for _, row in split_df.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            sol = Chem.MolFromSmiles(str(row["solvent"]))
            if mol is None or sol is None:
                continue
            canon_smi = Chem.MolToSmiles(mol)
            canon_sol = Chem.MolToSmiles(sol)
            lmax = str(float(row["peakwavs_max"]))
            key = f"{canon_smi}|{canon_sol}|{lmax}"
            matches = processed_df.index[processed_df["_key"] == key].tolist()
            if matches:
                indices.append(matches[0])
        return np.array(indices, dtype=np.int64)

    print("  Mapping pre-defined splits to processed indices...")
    train_idx = get_split_indices(train_df, df)
    val_idx = get_split_indices(val_df, df)
    test_idx = get_split_indices(test_df, df)

    np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)
    print(f"  Split mapping: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  Coverage: {len(train_idx) + len(val_idx) + len(test_idx)}/{len(df)} "
          f"({(len(train_idx) + len(val_idx) + len(test_idx)) / len(df):.1%})")

    # Drop temporary key column from saved file
    df = df.drop(columns=["_key"])
    df.to_csv(out_path, index=False)

    # Stats
    print(f"\n  [nablaColors] Saved {len(df)} entries to {out_path}")
    print(f"  Unique solutes: {df['smiles'].nunique()}")
    print(f"  Unique solvents: {df['solvent_smiles'].nunique()}")
    print(f"  λ_max range: {df['lambda_max'].min():.1f} – {df['lambda_max'].max():.1f} nm")
    print(f"  λ_max mean: {df['lambda_max'].mean():.1f} nm, std: {df['lambda_max'].std():.1f} nm")
    print(f"  Split indices saved to {split_path}")

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Download + preprocess UV absorption datasets")
    parser.add_argument("--dataset", choices=["deep4chem", "jung2024", "nablacolors"],
                        help="Process a single dataset (default: all)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use cached raw files")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["deep4chem", "jung2024", "nablacolors"]

    print("\n" + "=" * 60)
    print("  UV Absorption Dataset Download & Preprocessing")
    print("=" * 60)

    # Download
    if not args.skip_download:
        print("\n[1/2] Downloading raw files...")
        if "deep4chem" in datasets:
            download_file(DEEP4CHEM_URL, DEEP4CHEM_RAW, "Deep4Chem (Joung 2020)")
        if "jung2024" in datasets:
            download_file(JUNG2024_URL, JUNG2024_RAW, "Jung et al. 2024")
        if "nablacolors" in datasets:
            os.makedirs(NABLACOLORS_DIR, exist_ok=True)
            for fname in NABLACOLORS_FILES:
                dest = os.path.join(NABLACOLORS_DIR, fname)
                download_file(f"{NABLACOLORS_BASE_URL}/{fname}", dest,
                              f"nablaColors {fname}")
    else:
        print("\n[1/2] Skipping download (--skip-download)")

    # Preprocess
    print("\n[2/2] Preprocessing...")
    if "deep4chem" in datasets:
        if os.path.exists(DEEP4CHEM_RAW):
            preprocess_deep4chem()
        else:
            print(f"  [SKIP] Deep4Chem raw file not found: {DEEP4CHEM_RAW}")

    if "jung2024" in datasets:
        if os.path.exists(JUNG2024_RAW):
            preprocess_jung2024()
        else:
            print(f"  [SKIP] Jung 2024 raw file not found: {JUNG2024_RAW}")

    if "nablacolors" in datasets:
        all_csv = os.path.join(NABLACOLORS_DIR, "absorption_pairs_all.csv")
        if os.path.exists(all_csv):
            preprocess_nablacolors()
        else:
            print(f"  [SKIP] nablaColors raw file not found: {all_csv}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
