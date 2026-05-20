#!/usr/bin/env python3
"""
R1-7 — Quantify overlap between the Joung+Beard primary dataset and the
Deep4Chem "external" dataset used in Cross-Dataset Benchmarking.

The reviewer's concern is that Deep4Chem is essentially the source dataset
for the Joung half of Joung+Beard, so calling it an *external* validation
overstates the OOD-ness of the test. To answer with numbers rather than
hand-waving, we canonicalize both datasets through RDKit and report
intersections at three levels of granularity:

  (1) unique solutes (canonical solute SMILES)
  (2) unique (solute, solvent) pairs
  (3) unique solute *scaffolds* (Bemis--Murcko, generic)

A 2-set Venn diagram of unique solute SMILES is saved to
results/r1_7_overlap_venn.png.

Outputs:
  results/r1_7_deep4chem_overlap.json     numeric summary
  results/r1_7_overlap_venn.png           Venn diagram (solute level)
  results/r1_7_overlap_table.csv          per-level overlap counts
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SCRIPT_DIR = Path(__file__).resolve().parent
PRIMARY = SCRIPT_DIR / "previous_code" / "UV_canonical_full_dataset.csv"
DEEP4CHEM = SCRIPT_DIR / "data" / "deep4chem_processed.csv"
OUT_DIR = SCRIPT_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)


def canonical(smi):
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def generic_scaffold(smi):
    """Return the generic Bemis-Murcko scaffold SMILES, or '' on failure."""
    if not isinstance(smi, str) or not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        generic = MurckoScaffold.MakeScaffoldGeneric(Chem.MolFromSmiles(scaf))
        return Chem.MolToSmiles(generic)
    except Exception:
        return ""


def main():
    print(f"[LOAD] {PRIMARY}")
    a = pd.read_csv(PRIMARY)[["canon", "solvents", "lambda_max"]].dropna()
    print(f"  Joung+Beard primary: {len(a)} rows")

    print(f"[LOAD] {DEEP4CHEM}")
    b = pd.read_csv(DEEP4CHEM).dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    print(f"  Deep4Chem processed: {len(b)} rows (before canonicalisation)")

    # Canonicalise Deep4Chem solutes and solvents via RDKit
    print("[CANON] canonicalising Deep4Chem solute SMILES...")
    b["canon"] = [canonical(s) for s in b["smiles"]]
    print("[CANON] canonicalising Deep4Chem solvent SMILES...")
    b["solvents_canon"] = [canonical(s) for s in b["solvent_smiles"]]
    n_drop = b["canon"].isna().sum() + b["solvents_canon"].isna().sum()
    b = b.dropna(subset=["canon", "solvents_canon"])
    print(f"  Deep4Chem after canonicalisation drops: {len(b)} rows")

    # Also canonicalise Joung+Beard solvent SMILES so apples-to-apples comparison
    # (the file's `solvents` column is already canonical for most rows but we
    #  pass everything through RDKit for safety).
    print("[CANON] re-canonicalising Joung+Beard solvent column...")
    a["solvents_canon"] = [canonical(s) or s for s in a["solvents"]]
    a = a.rename(columns={"canon": "canon"})  # keep name explicit

    # Solute level — canonical SMILES
    set_a_solute = set(a["canon"])
    set_b_solute = set(b["canon"])
    inter_solute = set_a_solute & set_b_solute
    print(f"\n[OVERLAP] solute-only (canonical SMILES)")
    print(f"  Joung+Beard unique solutes:  {len(set_a_solute):>6d}")
    print(f"  Deep4Chem    unique solutes: {len(set_b_solute):>6d}")
    print(f"  intersection:                {len(inter_solute):>6d}")
    print(f"  D4C fraction in J+B:         {len(inter_solute)/max(len(set_b_solute),1):.1%}")
    print(f"  J+B fraction in D4C:         {len(inter_solute)/max(len(set_a_solute),1):.1%}")

    # Solute+solvent pair level
    pairs_a = set(zip(a["canon"], a["solvents_canon"]))
    pairs_b = set(zip(b["canon"], b["solvents_canon"]))
    inter_pair = pairs_a & pairs_b
    print(f"\n[OVERLAP] (solute, solvent) pair (both canonicalised)")
    print(f"  Joung+Beard unique pairs:  {len(pairs_a):>6d}")
    print(f"  Deep4Chem    unique pairs: {len(pairs_b):>6d}")
    print(f"  intersection:              {len(inter_pair):>6d}")
    print(f"  D4C fraction in J+B:       {len(inter_pair)/max(len(pairs_b),1):.1%}")
    print(f"  J+B fraction in D4C:       {len(inter_pair)/max(len(pairs_a),1):.1%}")

    # Scaffold level (generic Bemis-Murcko) — measures structural overlap rather
    # than literal compound overlap
    print(f"\n[CANON] computing generic Bemis--Murcko scaffolds...")
    # Use the unique-solute set to save compute (~6-8K compounds each)
    scaf_map_a = {s: generic_scaffold(s) for s in set_a_solute}
    scaf_map_b = {s: generic_scaffold(s) for s in set_b_solute}
    scaffolds_a = set(v for v in scaf_map_a.values() if v)
    scaffolds_b = set(v for v in scaf_map_b.values() if v)
    inter_scaf = scaffolds_a & scaffolds_b
    print(f"[OVERLAP] generic Bemis--Murcko scaffold")
    print(f"  Joung+Beard unique scaffolds:  {len(scaffolds_a):>6d}")
    print(f"  Deep4Chem    unique scaffolds: {len(scaffolds_b):>6d}")
    print(f"  intersection:                  {len(inter_scaf):>6d}")
    print(f"  D4C fraction in J+B:           {len(inter_scaf)/max(len(scaffolds_b),1):.1%}")
    print(f"  J+B fraction in D4C:           {len(inter_scaf)/max(len(scaffolds_a),1):.1%}")

    # Save a per-level overlap table
    rows = [
        {"granularity": "canonical solute SMILES",
         "joung_beard": len(set_a_solute), "deep4chem": len(set_b_solute),
         "intersection": len(inter_solute),
         "frac_d4c_in_jb": round(len(inter_solute)/max(len(set_b_solute),1), 4),
         "frac_jb_in_d4c": round(len(inter_solute)/max(len(set_a_solute),1), 4)},
        {"granularity": "(solute, solvent) pair (both canonical)",
         "joung_beard": len(pairs_a), "deep4chem": len(pairs_b),
         "intersection": len(inter_pair),
         "frac_d4c_in_jb": round(len(inter_pair)/max(len(pairs_b),1), 4),
         "frac_jb_in_d4c": round(len(inter_pair)/max(len(pairs_a),1), 4)},
        {"granularity": "generic Bemis-Murcko scaffold",
         "joung_beard": len(scaffolds_a), "deep4chem": len(scaffolds_b),
         "intersection": len(inter_scaf),
         "frac_d4c_in_jb": round(len(inter_scaf)/max(len(scaffolds_b),1), 4),
         "frac_jb_in_d4c": round(len(inter_scaf)/max(len(scaffolds_a),1), 4)},
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "r1_7_overlap_table.csv", index=False)

    summary = {
        "joung_beard_rows": int(len(a)),
        "deep4chem_rows": int(len(b)),
        "joung_beard_unique_solutes": int(len(set_a_solute)),
        "deep4chem_unique_solutes": int(len(set_b_solute)),
        "solute_intersection": int(len(inter_solute)),
        "joung_beard_unique_pairs": int(len(pairs_a)),
        "deep4chem_unique_pairs": int(len(pairs_b)),
        "pair_intersection": int(len(inter_pair)),
        "joung_beard_unique_scaffolds": int(len(scaffolds_a)),
        "deep4chem_unique_scaffolds": int(len(scaffolds_b)),
        "scaffold_intersection": int(len(inter_scaf)),
    }
    with open(OUT_DIR / "r1_7_deep4chem_overlap.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Venn diagrams — solute level and pair level side-by-side
    print(f"\n[FIG] rendering Venn diagram...")
    try:
        from matplotlib_venn import venn2
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib_venn not installed; falling back to plain bar chart")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(["Solutes", "Pairs", "Scaffolds"],
                [len(inter_solute), len(inter_pair), len(inter_scaf)],
                color="#7090c0", label="overlap")
        ax.barh(["Solutes", "Pairs", "Scaffolds"],
                [len(set_b_solute) - len(inter_solute),
                 len(pairs_b) - len(inter_pair),
                 len(scaffolds_b) - len(inter_scaf)],
                left=[len(inter_solute), len(inter_pair), len(inter_scaf)],
                color="#cccccc", label="Deep4Chem only")
        ax.set_xlabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "r1_7_overlap_venn.png", dpi=200)
        print(f"  saved {OUT_DIR/'r1_7_overlap_venn.png'}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    venn2([set_a_solute, set_b_solute],
          set_labels=("Joung+Beard\nprimary", "Deep4Chem\nexternal"),
          ax=axes[0])
    axes[0].set_title("Unique canonical solute SMILES")

    venn2([pairs_a, pairs_b],
          set_labels=("Joung+Beard\nprimary", "Deep4Chem\nexternal"),
          ax=axes[1])
    axes[1].set_title("Unique (solute, solvent) pairs")

    fig.suptitle("Overlap between primary dataset and Deep4Chem external benchmark",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "r1_7_overlap_venn.png", dpi=200, bbox_inches="tight")
    print(f"  saved {OUT_DIR/'r1_7_overlap_venn.png'}")


if __name__ == "__main__":
    main()
