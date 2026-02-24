#!/usr/bin/env python3
"""
Post-process raw Reaxys UV-Vis data into clean regression & classification datasets.

Takes the raw extraction output from reaxys_extract_uv.py and:
1. For multi-peak molecules, selects the peak with highest MEC
2. Filters by solvent (configurable)
3. Canonicalizes SMILES with RDKit
4. Merges with Mamede class labels
5. Deduplicates against our training data (Joung+Beard)
6. Outputs regression and classification datasets

Usage:
    python3 postprocess_reaxys.py                              # default: methanol
    python3 postprocess_reaxys.py --solvent all                # keep all solvents
    python3 postprocess_reaxys.py --solvent "methanol,ethanol" # custom filter
    python3 postprocess_reaxys.py --no-dedup                   # skip deduplication
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_UV_FILE = RAW_DIR / "reaxys_uv_raw.csv"
PARSED_MAMEDE = DATA_DIR / "mamede_parsed.csv"
TRAIN_DATA_PATH = Path("previous_code/UV_canonical_full_dataset.csv")

OUTPUT_REGRESSION = DATA_DIR / "mamede_regression_dataset.csv"
OUTPUT_CLASSIFICATION = DATA_DIR / "mamede_classification_dataset.csv"

# ---------------------------------------------------------------------------
# Solvent name → canonical SMILES mapping
# (reused from eval_external.py)
# ---------------------------------------------------------------------------
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


def map_solvent_to_smiles(name: str) -> str | None:
    """Map a solvent name to canonical SMILES. Returns None if unmappable."""
    if not name or not isinstance(name, str):
        return None
    key = name.strip().lower()
    # Direct lookup
    smi = SOLVENT_NAME_TO_SMILES.get(key)
    if smi:
        return smi
    # Try partial match (e.g. "methanol (99%)" → "methanol")
    for known_name, known_smi in SOLVENT_NAME_TO_SMILES.items():
        if known_name in key or key in known_name:
            return known_smi
    # Try as SMILES directly (if already a valid SMILES string)
    mol = Chem.MolFromSmiles(name)
    if mol:
        return Chem.MolToSmiles(mol)
    return None


def canonicalize_smiles(smi: str) -> str | None:
    """Canonicalize a SMILES string with RDKit. Returns None if invalid."""
    if not smi or not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
def load_raw_uv(path: Path) -> pd.DataFrame:
    """Load raw UV extraction results."""
    df = pd.read_csv(path)
    print(f"  Raw UV records loaded: {len(df)}")

    # Basic cleaning
    df["wavelength_nm"] = pd.to_numeric(df["wavelength_nm"], errors="coerce")
    df["mec"] = pd.to_numeric(df["mec"], errors="coerce")
    df = df.dropna(subset=["wavelength_nm"])

    # Sanity filter: wavelength should be 100-1000 nm
    before = len(df)
    df = df[(df["wavelength_nm"] >= 100) & (df["wavelength_nm"] <= 1000)]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} records with wavelength outside 100-1000 nm")

    print(f"  Valid UV records: {len(df)}")
    return df


def select_primary_peak(df: pd.DataFrame) -> pd.DataFrame:
    """For molecules with multiple UV peaks, select the one with highest MEC.

    If MEC is missing for all peaks of a molecule, take the peak with the
    longest wavelength (common convention for primary absorption).
    """
    # Group by (xrn, solvent) since same molecule may have data in different solvents
    groups = []
    for (xrn, solvent), group in df.groupby(["xrn", "solvent"]):
        if len(group) == 1:
            groups.append(group)
            continue
        # Prefer peak with highest MEC
        has_mec = group["mec"].notna()
        if has_mec.any():
            best_idx = group.loc[has_mec, "mec"].idxmax()
        else:
            # Fallback: longest wavelength
            best_idx = group["wavelength_nm"].idxmax()
        groups.append(group.loc[[best_idx]])

    result = pd.concat(groups, ignore_index=True)
    print(f"  After selecting primary peak: {len(result)} records "
          f"(from {len(df)} raw)")
    return result


def filter_solvents(df: pd.DataFrame, solvent_arg: str) -> pd.DataFrame:
    """Filter by solvent according to CLI flag.

    Args:
        solvent_arg: "all", single name, or comma-separated list
    """
    if solvent_arg.lower() == "all":
        print("  Solvent filter: keeping ALL solvents")
        # Map solvent names to SMILES
        df["solvent_smiles"] = df["solvent"].apply(map_solvent_to_smiles)
        unmapped = df["solvent_smiles"].isna().sum()
        if unmapped:
            print(f"  Warning: {unmapped} records have unmappable solvent names")
        return df

    # Parse solvent list
    target_solvents = [s.strip().lower() for s in solvent_arg.split(",")]
    target_smiles = set()
    for name in target_solvents:
        smi = SOLVENT_NAME_TO_SMILES.get(name)
        if smi:
            target_smiles.add(smi)
        else:
            print(f"  Warning: unknown solvent '{name}', skipping")

    if not target_smiles:
        print("  ERROR: No valid solvents in filter list")
        sys.exit(1)

    print(f"  Solvent filter: {target_solvents} → {target_smiles}")

    # Map and filter
    df["solvent_smiles"] = df["solvent"].apply(map_solvent_to_smiles)
    before = len(df)
    df = df[df["solvent_smiles"].isin(target_smiles)].copy()
    print(f"  After solvent filter: {len(df)} records (dropped {before - len(df)})")
    return df


def canonicalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize all SMILES in the dataset."""
    df["smiles_canonical"] = df["smiles"].apply(canonicalize_smiles)
    invalid = df["smiles_canonical"].isna().sum()
    if invalid:
        print(f"  Warning: {invalid} records have invalid SMILES (dropped)")
        df = df.dropna(subset=["smiles_canonical"]).copy()
    df["smiles"] = df["smiles_canonical"]
    df = df.drop(columns=["smiles_canonical"])
    return df


# ---------------------------------------------------------------------------
# Derive POS/NEG class labels from UV data (Mamede criteria)
# ---------------------------------------------------------------------------
# ICH S10 / Mamede criteria: POS = any absorption maximum in 290-700 nm
# with MEC >= 1000 L·mol⁻¹·cm⁻¹
POS_WAV_MIN = 290  # nm
POS_WAV_MAX = 700  # nm
POS_MEC_THRESHOLD = 1000  # L·mol⁻¹·cm⁻¹


def assign_class_labels(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Derive POS/NEG class labels from UV data using Mamede's ICH S10 criteria.

    A molecule is POS if it has ANY absorption peak in 290-700 nm with MEC >= 1000.
    The raw_df (before peak selection) is needed because POS/NEG depends on ALL peaks,
    not just the primary one selected for regression.

    Args:
        df: processed regression dataset (one row per molecule, primary peak)
        raw_df: raw UV data with ALL peaks per molecule
    """
    # Determine POS molecules from ALL their peaks (before peak selection)
    pos_mask = (
        (raw_df["wavelength_nm"] >= POS_WAV_MIN)
        & (raw_df["wavelength_nm"] <= POS_WAV_MAX)
        & (raw_df["mec"].fillna(0) >= POS_MEC_THRESHOLD)
    )
    pos_xrns = set(raw_df.loc[pos_mask, "xrn"].astype(str).unique())

    df["xrn"] = df["xrn"].astype(str)
    df["class_label"] = df["xrn"].apply(lambda x: "POS" if x in pos_xrns else "NEG")

    n_pos = (df["class_label"] == "POS").sum()
    n_neg = (df["class_label"] == "NEG").sum()
    print(f"  Assigned class labels: {n_pos} POS, {n_neg} NEG")
    print(f"  Criteria: absorption in {POS_WAV_MIN}-{POS_WAV_MAX} nm "
          f"with MEC >= {POS_MEC_THRESHOLD}")
    return df


# ---------------------------------------------------------------------------
# Deduplication against training data
# ---------------------------------------------------------------------------
def deduplicate_training(df: pd.DataFrame) -> pd.DataFrame:
    """Remove molecules that overlap with our Joung+Beard training data."""
    if not TRAIN_DATA_PATH.exists():
        print(f"  Warning: {TRAIN_DATA_PATH} not found, skipping deduplication")
        return df

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_smiles = set(train_df["canon"].dropna().values)

    before = len(df)
    mask = ~df["smiles"].isin(train_smiles)
    df = df[mask].copy()
    n_overlap = before - len(df)
    pct = n_overlap / before * 100 if before > 0 else 0
    print(f"  Deduplication: removed {n_overlap} overlapping molecules "
          f"({pct:.1f}%), {before} → {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_regression_dataset(df: pd.DataFrame):
    """Save the regression dataset."""
    cols = ["smiles", "solvent_smiles", "wavelength_nm", "mec", "class_label", "xrn"]
    # Rename wavelength_nm → lambda_max for consistency with our other datasets
    out = df[[c for c in cols if c in df.columns]].copy()
    out = out.rename(columns={"wavelength_nm": "lambda_max"})
    out.to_csv(OUTPUT_REGRESSION, index=False)
    print(f"\n  Regression dataset saved: {OUTPUT_REGRESSION}")
    print(f"    Rows: {len(out)}")
    print(f"    lambda_max range: {out['lambda_max'].min():.1f} - "
          f"{out['lambda_max'].max():.1f} nm")
    print(f"    lambda_max mean: {out['lambda_max'].mean():.1f} nm")
    if "solvent_smiles" in out.columns:
        print(f"    Unique solvents: {out['solvent_smiles'].nunique()}")
    if "class_label" in out.columns:
        vc = out["class_label"].value_counts()
        for label, count in vc.items():
            print(f"    Class '{label}': {count}")


def save_classification_dataset(raw_df: pd.DataFrame):
    """Save classification dataset with POS/NEG labels derived from UV data.

    Uses ALL extracted UV data (not just primary peak) to assign labels per
    Mamede's ICH S10 criteria, then joins with the full Mamede molecule list
    so molecules with no UV data in the extraction are labeled NEG.
    """
    # Determine POS XRNs from raw UV data
    pos_mask = (
        (raw_df["wavelength_nm"] >= POS_WAV_MIN)
        & (raw_df["wavelength_nm"] <= POS_WAV_MAX)
        & (raw_df["mec"].fillna(0) >= POS_MEC_THRESHOLD)
    )
    pos_xrns = set(raw_df.loc[pos_mask, "xrn"].astype(str).unique())

    if PARSED_MAMEDE.exists():
        mamede = pd.read_csv(PARSED_MAMEDE)
        mamede["smiles"] = mamede["smiles"].apply(canonicalize_smiles)
        mamede = mamede.dropna(subset=["smiles"])
        mamede["xrn"] = mamede["xrn"].astype(str)
        mamede["class_label"] = mamede["xrn"].apply(
            lambda x: "POS" if x in pos_xrns else "NEG"
        )
        out = mamede[["smiles", "class_label", "xrn"]]
    else:
        # Fall back to only molecules present in the raw UV extraction
        xrns_in_raw = raw_df[["xrn", "smiles"]].drop_duplicates(subset=["xrn"])
        xrns_in_raw["xrn"] = xrns_in_raw["xrn"].astype(str)
        xrns_in_raw["smiles"] = xrns_in_raw["smiles"].apply(canonicalize_smiles)
        xrns_in_raw = xrns_in_raw.dropna(subset=["smiles"])
        xrns_in_raw["class_label"] = xrns_in_raw["xrn"].apply(
            lambda x: "POS" if x in pos_xrns else "NEG"
        )
        out = xrns_in_raw[["smiles", "class_label", "xrn"]]

    out.to_csv(OUTPUT_CLASSIFICATION, index=False)
    print(f"\n  Classification dataset saved: {OUTPUT_CLASSIFICATION}")
    print(f"    Rows: {len(out)}")
    vc = out["class_label"].value_counts()
    for label, count in vc.items():
        print(f"    Class '{label}': {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Post-process Reaxys UV data into regression/classification datasets"
    )
    parser.add_argument(
        "--solvent", type=str, default="methanol",
        help='Solvent filter: "all", single name, or comma-separated '
             '(default: methanol)',
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Skip deduplication against training data",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help=f"Path to raw UV CSV (default: {RAW_UV_FILE})",
    )
    args = parser.parse_args()

    raw_path = Path(args.input) if args.input else RAW_UV_FILE
    if not raw_path.exists():
        print(f"ERROR: Raw UV data not found: {raw_path}")
        print("  Run reaxys_extract_uv.py first to extract UV data from Reaxys.")
        sys.exit(1)

    print("=" * 60)
    print("Post-processing Reaxys UV-Vis data")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n[1/6] Loading raw UV data...")
    df = load_raw_uv(raw_path)
    raw_df = df.copy()  # keep ALL peaks for class label derivation

    # Step 2: Select primary peak per molecule
    print("\n[2/6] Selecting primary absorption peak...")
    df = select_primary_peak(df)

    # Step 3: Filter by solvent
    print("\n[3/6] Filtering by solvent...")
    df = filter_solvents(df, args.solvent)

    # Step 4: Canonicalize SMILES
    print("\n[4/6] Canonicalizing SMILES...")
    df = canonicalize_dataset(df)

    # Step 5: Derive POS/NEG class labels from UV data (Mamede ICH S10 criteria)
    print("\n[5/6] Assigning class labels from UV data...")
    df = assign_class_labels(df, raw_df)

    # Step 6: Deduplication
    if not args.no_dedup:
        print("\n[6/6] Deduplicating against training data...")
        df = deduplicate_training(df)
    else:
        print("\n[6/6] Skipping deduplication (--no-dedup)")

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving datasets")
    print("=" * 60)
    save_regression_dataset(df)
    save_classification_dataset(raw_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
