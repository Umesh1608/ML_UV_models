#!/usr/bin/env python3
"""
Convert Reaxys web exports into the unified raw CSV format expected by postprocess_reaxys.py.

Three modes:
  --generate-batches   Generate .txt batch query files for Reaxys web import
  --preview            Preview column detection on exported files (no output written)
  (default)            Convert all exports → data/raw/reaxys_uv_raw.csv

Usage:
    # Step 1: Generate batch files for Reaxys web import
    python3 convert_reaxys_web_export.py --generate-batches

    # Step 2: After exporting from Reaxys web, preview column detection
    python3 convert_reaxys_web_export.py --preview

    # Step 3: Convert all exports to unified format
    python3 convert_reaxys_web_export.py

    # Optional: manual column mapping if auto-detect fails
    python3 convert_reaxys_web_export.py --column-map xrn="Registry Number" wavelength_nm="Abs Max"

    # Optional: specify input files explicitly
    python3 convert_reaxys_web_export.py --input data/raw/reaxys_exports/batch_001.csv
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
MAMEDE_PARSED = DATA_DIR / "mamede_parsed.csv"
BATCH_DIR = DATA_DIR / "reaxys_batches"
EXPORT_DIR = RAW_DIR / "reaxys_exports"
OUTPUT_FILE = RAW_DIR / "reaxys_uv_raw.csv"

BATCH_SIZE = 1000  # default batch size
BATCH_DIR_LARGE = DATA_DIR / "reaxys_batches_25"  # larger batches for 25/day limit

# ---------------------------------------------------------------------------
# Column auto-detection mappings
# ---------------------------------------------------------------------------
# Each key is our target column; values are possible names in Reaxys exports
# (matched case-insensitively)
COLUMN_ALIASES = {
    "xrn": [
        "reaxys registry number",
        "ide.xrn",
        "registry number",
        "rn",
        "reaxys id",
        "xrn",
        "reaxys_registry_number",
        "ide_xrn",
    ],
    "wavelength_nm": [
        "absorption maxima (uv/vis) [nm]",
        "absorption maximum",
        "uv/vis absorption maximum",
        "uv-vis absorption maximum",
        "uv vis absorption maximum",
        "wav",
        "λmax",
        "lambda_max",
        "lambda max",
        "wavelength",
        "wavelength_nm",
        "absorption maximum (nm)",
        "abs max",
        "abs_max",
    ],
    "mec": [
        "ext./abs. coefficient [l·mol-1cm-1]",
        "ext./abs. coefficient [l·mol 1cm 1]",
        "molar extinction coefficient",
        "extinction coefficient",
        "ε",
        "ext",
        "mec",
        "molar_extinction_coefficient",
        "log ε",
        "log_epsilon",
    ],
    "solvent": [
        "solvent (uv/vis spectroscopy)",
        "solvent (uv/vis)",
        "solvent (uv-vis)",
        "solvent",
        "sol",
    ],
    "smiles": [
        "smiles",
        "molecular structure",
        "structure",
        "canonical smiles",
        "canonical_smiles",
    ],
}


def _normalize(s: str) -> str:
    """Normalize column name for matching."""
    return s.strip().lower().replace("_", " ").replace("-", " ")


def auto_detect_columns(columns: list[str]) -> dict[str, str]:
    """Map export columns to our schema. Returns {our_name: export_col_name}."""
    mapping = {}
    norm_to_orig = {_normalize(c): c for c in columns}

    for our_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            norm_alias = _normalize(alias)
            if norm_alias in norm_to_orig:
                mapping[our_col] = norm_to_orig[norm_alias]
                break
    return mapping


# ---------------------------------------------------------------------------
# Mode 1: Generate batch query files
# ---------------------------------------------------------------------------
def generate_batches(batch_size: int = BATCH_SIZE, batch_dir: Path = BATCH_DIR):
    """Generate .txt batch files for Reaxys web import."""
    if not MAMEDE_PARSED.exists():
        print(f"ERROR: {MAMEDE_PARSED} not found")
        sys.exit(1)

    mamede = pd.read_csv(MAMEDE_PARSED)
    xrns = mamede["xrn"].dropna().astype(int).unique()
    xrns.sort()
    n_total = len(xrns)
    n_batches = math.ceil(n_total / batch_size)

    print(f"Generating batch query files for {n_total} XRNs...")
    print(f"  Batch size: {batch_size} XRNs per file")
    print(f"  Total batches: {n_batches}")
    print(f"  Output directory: {batch_dir}/")

    batch_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_total)
        batch_xrns = xrns[start:end]
        batch_file = batch_dir / f"batch_{i + 1:03d}.txt"

        with open(batch_file, "w") as f:
            for xrn in batch_xrns:
                f.write(f"IDE.XRN={xrn}\n")

        print(f"  Written {batch_file} ({len(batch_xrns)} queries)")

    print(f"\nDone! {n_batches} batch files in {batch_dir}/")
    print()
    print("=" * 60)
    print("REAXYS WEB INSTRUCTIONS")
    print("=" * 60)
    print("""
For each batch file:

1. Go to Reaxys (reaxys.com) → Query Builder → click "Import"
2. Upload the batch .txt file (e.g. batch_001.txt)
3. Click "Search" → select "Substances" tab
4. In the results table, ensure these columns are visible:
   - Reaxys Registry Number (or Reaxys ID)
   - SMILES (or Molecular Structure)
   - UV/VIS Absorption Maximum
   - Molar Extinction Coefficient
   - Solvent
5. Click "Export" → choose CSV format
6. Save the exported file to:
   data/raw/reaxys_exports/batch_001.csv
7. Repeat for remaining batches (or as many as practical)

Then run:
    python3 convert_reaxys_web_export.py --preview   # check column detection
    python3 convert_reaxys_web_export.py              # convert to unified format
""")


# ---------------------------------------------------------------------------
# Mode 2: Preview column detection
# ---------------------------------------------------------------------------
def preview_exports(input_files: list[Path] | None = None):
    """Preview column detection on exported files without writing output."""
    files = input_files or _find_export_files()
    if not files:
        print(f"No export files found in {EXPORT_DIR}/")
        print("  Expected CSV or Excel files (.csv, .xlsx, .xls)")
        sys.exit(1)

    print(f"Found {len(files)} export file(s)\n")

    for fpath in files:
        print(f"--- {fpath.name} ---")
        df = _read_export_file(fpath)
        if df is None:
            print("  ERROR: could not read file\n")
            continue

        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        mapping = auto_detect_columns(list(df.columns))
        print(f"  Auto-mapped:")
        for our_col, export_col in mapping.items():
            sample = df[export_col].dropna().head(3).tolist()
            print(f"    {our_col} ← \"{export_col}\"  (samples: {sample})")

        unmapped = set(COLUMN_ALIASES.keys()) - set(mapping.keys())
        if unmapped:
            print(f"  NOT mapped: {unmapped}")
            if "smiles" in unmapped:
                print("    → SMILES will be filled from mamede_parsed.csv (if xrn is mapped)")
        print()


# ---------------------------------------------------------------------------
# Mode 3: Convert exports
# ---------------------------------------------------------------------------
def convert_exports(
    input_files: list[Path] | None = None,
    column_map: dict[str, str] | None = None,
):
    """Convert all Reaxys web exports to unified raw CSV."""
    files = input_files or _find_export_files()
    if not files:
        print(f"No export files found in {EXPORT_DIR}/")
        print("  Expected CSV or Excel files (.csv, .xlsx, .xls)")
        sys.exit(1)

    # Load mamede for SMILES fallback
    mamede_smiles = {}
    if MAMEDE_PARSED.exists():
        mamede = pd.read_csv(MAMEDE_PARSED)
        mamede_smiles = dict(zip(mamede["xrn"].astype(str), mamede["smiles"]))
        print(f"Loaded {len(mamede_smiles)} SMILES from {MAMEDE_PARSED}")

    all_dfs = []
    for fpath in files:
        print(f"\nProcessing {fpath.name}...")
        df = _read_export_file(fpath)
        if df is None:
            print(f"  WARNING: could not read {fpath}, skipping")
            continue

        # Detect or apply column mapping
        mapping = auto_detect_columns(list(df.columns))
        if column_map:
            mapping.update(column_map)

        # Check minimum required columns
        if "xrn" not in mapping and "smiles" not in mapping:
            print(f"  WARNING: cannot map XRN or SMILES columns, skipping")
            print(f"    Available columns: {list(df.columns)}")
            print(f"    Use --column-map to specify manually")
            continue

        if "wavelength_nm" not in mapping:
            print(f"  WARNING: cannot map wavelength column, skipping")
            print(f"    Available columns: {list(df.columns)}")
            continue

        # Rename columns to our schema
        rename_map = {v: k for k, v in mapping.items()}
        df = df.rename(columns=rename_map)

        # Fill missing SMILES from mamede_parsed.csv
        if "smiles" not in df.columns or df["smiles"].isna().all():
            if "xrn" in df.columns and mamede_smiles:
                df["smiles"] = df["xrn"].astype(str).map(mamede_smiles)
                filled = df["smiles"].notna().sum()
                print(f"  Filled {filled}/{len(df)} SMILES from mamede_parsed.csv")
            else:
                print(f"  WARNING: no SMILES column and no XRN to fill from")
        else:
            # Fill only missing rows
            if "xrn" in df.columns and mamede_smiles:
                missing_mask = df["smiles"].isna()
                if missing_mask.any():
                    df.loc[missing_mask, "smiles"] = (
                        df.loc[missing_mask, "xrn"].astype(str).map(mamede_smiles)
                    )
                    filled = missing_mask.sum() - df["smiles"].isna().sum()
                    print(f"  Filled {filled} missing SMILES from mamede_parsed.csv")

        # Ensure required columns exist
        for col in ["xrn", "smiles", "wavelength_nm", "mec", "solvent"]:
            if col not in df.columns:
                df[col] = pd.NA

        # Select and clean
        df = df[["xrn", "smiles", "wavelength_nm", "mec", "solvent"]].copy()

        # Expand semicolon-separated multi-value fields into separate rows.
        # Reaxys exports wavelength and MEC as e.g. "205; 232; 273" and
        # "1774; 6; 14" — paired positionally.
        before_expand = len(df)
        df = _expand_multivalue_rows(df)
        if len(df) > before_expand:
            print(f"  Expanded multi-value fields: {before_expand} → {len(df)} rows")

        # Parse numeric fields (handle ranges like "350-400" → take first value)
        df["wavelength_nm"] = df["wavelength_nm"].apply(_parse_numeric)
        df["mec"] = df["mec"].apply(_parse_numeric)

        # Drop rows with no wavelength
        before = len(df)
        df = df.dropna(subset=["wavelength_nm"])
        if len(df) < before:
            print(f"  Dropped {before - len(df)} rows with missing wavelength")

        print(f"  Kept {len(df)} records from {fpath.name}")
        all_dfs.append(df)

    if not all_dfs:
        print("\nERROR: No valid data extracted from any export file")
        sys.exit(1)

    # Concatenate all
    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove exact duplicates
    before = len(combined)
    combined = combined.drop_duplicates()
    if len(combined) < before:
        print(f"\nRemoved {before - len(combined)} exact duplicates")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    # Report
    print(f"\n{'=' * 60}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'=' * 60}")
    print(f"  Total records: {len(combined)}")
    print(f"  Unique XRNs: {combined['xrn'].dropna().nunique()}")
    if mamede_smiles:
        coverage = combined["xrn"].dropna().astype(str).isin(mamede_smiles).sum()
        pct = coverage / len(mamede_smiles) * 100
        print(f"  Coverage: {combined['xrn'].dropna().nunique()} / {len(mamede_smiles)} "
              f"Mamede molecules ({pct:.1f}%)")
    print(f"  Wavelength range: {combined['wavelength_nm'].min():.0f} - "
          f"{combined['wavelength_nm'].max():.0f} nm")
    print(f"  Records with MEC: {combined['mec'].notna().sum()}")
    print(f"  Records with solvent: {combined['solvent'].notna().sum()}")
    print(f"  Records with SMILES: {combined['smiles'].notna().sum()}")

    # Solvent distribution
    if combined["solvent"].notna().any():
        print(f"\n  Top solvents:")
        top = combined["solvent"].value_counts().head(10)
        for solvent, count in top.items():
            print(f"    {solvent}: {count}")

    print(f"\nNext step:")
    print(f"  python3 postprocess_reaxys.py --solvent all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expand_multivalue_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Expand semicolon-separated wavelength/MEC values into separate rows.

    Reaxys exports multi-peak data as e.g.:
        wavelength_nm = "205; 232; 273"
        mec           = "1774; 6; 14"
    These are paired positionally. This function splits them into individual rows.
    """
    expanded_rows = []
    for _, row in df.iterrows():
        wav_str = str(row["wavelength_nm"]) if pd.notna(row["wavelength_nm"]) else ""
        mec_str = str(row["mec"]) if pd.notna(row["mec"]) else ""

        # Split on semicolons
        wavs = [w.strip() for w in wav_str.split(";")] if wav_str else [""]
        mecs = [m.strip() for m in mec_str.split(";")] if mec_str else [""]

        # If only one wavelength (no semicolons), keep as-is
        if len(wavs) <= 1 and len(mecs) <= 1:
            expanded_rows.append(row)
            continue

        # Pair wavelengths with MECs positionally; pad shorter list with empty
        n = max(len(wavs), len(mecs))
        wavs += [""] * (n - len(wavs))
        mecs += [""] * (n - len(mecs))

        for w, m in zip(wavs, mecs):
            new_row = row.copy()
            new_row["wavelength_nm"] = w if w else pd.NA
            new_row["mec"] = m if m else pd.NA
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows, columns=df.columns).reset_index(drop=True)


def _find_export_files() -> list[Path]:
    """Find all CSV/Excel files in the export directory."""
    if not EXPORT_DIR.exists():
        return []
    exts = {".csv", ".xlsx", ".xls", ".tsv"}
    files = [f for f in sorted(EXPORT_DIR.iterdir()) if f.suffix.lower() in exts]
    return files


def _read_export_file(fpath: Path) -> pd.DataFrame | None:
    """Read a CSV or Excel export file."""
    try:
        suffix = fpath.suffix.lower()
        if suffix == ".csv":
            # Try common encodings and separators
            for sep in [",", "\t", ";"]:
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        df = pd.read_csv(fpath, sep=sep, encoding=encoding)
                        if len(df.columns) > 1:
                            return df
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
            # Last resort: single-column CSV might just need default
            return pd.read_csv(fpath, encoding="latin-1")
        elif suffix == ".tsv":
            return pd.read_csv(fpath, sep="\t", encoding="utf-8")
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(fpath)
    except Exception as e:
        print(f"  Error reading {fpath}: {e}")
    return None


def _parse_numeric(val) -> float | None:
    """Parse a numeric value, handling ranges and units.

    Examples:
        "350" → 350.0
        "350-400" → 350.0 (take first value)
        "350 nm" → 350.0
        ">350" → 350.0
        "~350" → 350.0
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None

    # Remove common units and prefixes
    for remove in ["nm", "l/(mol*cm)", "l·mol⁻¹·cm⁻¹", ">", "<", "~", "≈", "ca.", "ca "]:
        s = s.replace(remove, "")
    s = s.strip()

    # Handle ranges: take first value
    for delimiter in ["-", "–", "—", "to", ","]:
        if delimiter in s:
            parts = s.split(delimiter)
            s = parts[0].strip()
            break

    # Handle parenthetical notes
    if "(" in s:
        s = s[:s.index("(")].strip()

    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_column_map(args: list[str]) -> dict[str, str]:
    """Parse --column-map arguments like xrn="Registry Number"."""
    mapping = {}
    for arg in args:
        if "=" not in arg:
            print(f"  WARNING: ignoring invalid column-map entry: {arg}")
            continue
        our_col, export_col = arg.split("=", 1)
        our_col = our_col.strip().strip('"').strip("'")
        export_col = export_col.strip().strip('"').strip("'")
        if our_col not in COLUMN_ALIASES:
            print(f"  WARNING: unknown column '{our_col}', expected one of: "
                  f"{list(COLUMN_ALIASES.keys())}")
            continue
        mapping[our_col] = export_col
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Convert Reaxys web exports to unified raw CSV format"
    )
    parser.add_argument(
        "--generate-batches", action="store_true",
        help="Generate .txt batch query files for Reaxys web import",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"XRNs per batch file (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--batch-dir", type=str, default=None,
        help="Output directory for batch files (default: auto based on size)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview column detection on exported files (no output written)",
    )
    parser.add_argument(
        "--input", nargs="+", type=str, default=None,
        help="Explicit input file(s) instead of scanning export directory",
    )
    parser.add_argument(
        "--column-map", nargs="+", type=str, default=None,
        metavar='COL="Export Name"',
        help='Manual column mapping, e.g. xrn="Registry Number" wavelength_nm="Abs Max"',
    )
    args = parser.parse_args()

    if args.generate_batches:
        if args.batch_dir:
            batch_dir = Path(args.batch_dir)
        elif args.batch_size != BATCH_SIZE:
            batch_dir = DATA_DIR / f"reaxys_batches_{math.ceil(74784 / args.batch_size)}"
        else:
            batch_dir = BATCH_DIR
        generate_batches(batch_size=args.batch_size, batch_dir=batch_dir)
        return

    input_files = [Path(p) for p in args.input] if args.input else None
    column_map_parsed = parse_column_map(args.column_map) if args.column_map else None

    if args.preview:
        preview_exports(input_files)
        return

    convert_exports(input_files, column_map_parsed)


if __name__ == "__main__":
    main()
