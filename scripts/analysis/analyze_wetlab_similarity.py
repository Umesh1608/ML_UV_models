#!/usr/bin/env python3
"""
R1-13 Step 0 -- Wetlab OOD audit (scaffold + Tanimoto-similarity-to-train).

For each of the 16 commercial UV-absorbers in data/wetlab_experimental.csv,
compute:
  1. Nearest-neighbour Tanimoto similarity (Morgan FP r=2, 2048 bits) to the
     full v3 cleaned Joung+Beard training set
     (previous_code/UV_canonical_v3_dedup.csv, 18,415 records).
  2. Bemis-Murcko generic scaffold via RDKit; check whether the exact scaffold
     SMILES appears in any training molecule.

Aggregate into Tanimoto bins (>0.85, >0.7, >0.5, <0.5) and report the count
of wetlab molecules whose scaffold is present in training.

Outputs:
  results/r1_13_wetlab_per_molecule.csv  -- per-molecule audit table
  results/r1_13_aggregate.json           -- bin counts + summary stats
  results/r1_13_summary.md               -- one-page human-readable summary
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results"
RESULTS.mkdir(exist_ok=True)

WETLAB_CSV = SCRIPT_DIR / "data" / "wetlab_experimental.csv"
TRAIN_CSV = SCRIPT_DIR / "previous_code" / "UV_canonical_v3_dedup.csv"


def canonicalise(smi):
    if smi is None or (isinstance(smi, float) and np.isnan(smi)):
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)


def morgan_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def murcko_generic(smi):
    """Return generic (atom-type-stripped) Bemis-Murcko scaffold SMILES, or
    None if the scaffold is empty (acyclic molecule)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    gen = MurckoScaffold.MakeScaffoldGeneric(scaf)
    if gen is None or gen.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(gen, canonical=True)


def murcko_specific(smi):
    """Return atom-type-aware Bemis-Murcko scaffold SMILES."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(m)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaf, canonical=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wetlab", default=str(WETLAB_CSV))
    ap.add_argument("--train", default=str(TRAIN_CSV))
    args = ap.parse_args()

    # --- Load wetlab molecules (unique, by name+SMILES) -------------------
    wl = pd.read_csv(args.wetlab)
    # Deduplicate to one row per molecule (same SMILES across solvents),
    # keep per-solvent lambda_max alongside.
    wl["canon"] = wl["smiles"].map(canonicalise)
    wl = wl.dropna(subset=["canon"]).reset_index(drop=True)

    by_mol = wl.groupby(["molecule", "canon"], sort=False).agg(
        lambda_max_EtOH=("lambda_max_exp",
                         lambda s: float(s[wl.loc[s.index, "solvent_name"]
                                          == "EtOH"].iloc[0])
                                   if (wl.loc[s.index, "solvent_name"]
                                       == "EtOH").any() else np.nan),
        lambda_max_MeOH=("lambda_max_exp",
                         lambda s: float(s[wl.loc[s.index, "solvent_name"]
                                          == "MeOH"].iloc[0])
                                   if (wl.loc[s.index, "solvent_name"]
                                       == "MeOH").any() else np.nan),
    ).reset_index()
    n_wl = len(by_mol)
    print(f"  loaded {n_wl} unique wetlab molecules from {args.wetlab}")

    # --- Load training set ------------------------------------------------
    tr = pd.read_csv(args.train)
    tr["canon"] = tr["canon"].fillna(tr["smiles"])
    tr["canon"] = tr["canon"].map(canonicalise)
    tr = tr.dropna(subset=["canon"]).reset_index(drop=True)
    print(f"  loaded {len(tr)} training records from {args.train}")

    # --- Build training fingerprints + scaffolds (cache per unique solute)-
    unique_train = tr.drop_duplicates(subset=["canon"]).reset_index(drop=True)
    print(f"  {len(unique_train)} unique training solutes (canonical SMILES)")

    print("  computing training Morgan FPs + Murcko scaffolds...")
    train_fps = []
    train_canons = []
    train_lmaxes = []
    train_gen_scaffolds = []
    train_spec_scaffolds = []
    # Lookup: canonical SMILES -> mean lambda_max in train
    lmax_map = tr.groupby("canon")["lambda_max"].mean().to_dict()

    for i, row in unique_train.iterrows():
        smi = row["canon"]
        fp = morgan_fp(smi)
        if fp is None:
            continue
        train_fps.append(fp)
        train_canons.append(smi)
        train_lmaxes.append(lmax_map.get(smi, np.nan))
        train_gen_scaffolds.append(murcko_generic(smi))
        train_spec_scaffolds.append(murcko_specific(smi))
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(unique_train)}")

    train_gen_set = set(s for s in train_gen_scaffolds if s)
    train_spec_set = set(s for s in train_spec_scaffolds if s)
    print(f"  {len(train_gen_set)} unique generic Murcko scaffolds in train")
    print(f"  {len(train_spec_set)} unique specific Murcko scaffolds in train")

    # --- Per-wetlab-molecule audit ----------------------------------------
    rows = []
    for _, w in by_mol.iterrows():
        smi = w["canon"]
        fp = morgan_fp(smi)
        if fp is None:
            print(f"  WARNING: {w['molecule']} failed FP -> skip")
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        j = int(np.argmax(sims))
        nearest_tanimoto = float(sims[j])
        nearest_smi = train_canons[j]
        nearest_lmax = train_lmaxes[j]

        gen_scaf = murcko_generic(smi)
        spec_scaf = murcko_specific(smi)
        gen_match = gen_scaf in train_gen_set if gen_scaf else False
        spec_match = spec_scaf in train_spec_set if spec_scaf else False

        # Find an example training SMILES whose generic scaffold matches
        example_train_for_scaf = None
        if gen_match:
            for ts, gs in zip(train_canons, train_gen_scaffolds):
                if gs == gen_scaf:
                    example_train_for_scaf = ts
                    break

        # Also: is the exact canonical SMILES present in train (true duplicate)?
        exact_in_train = smi in set(train_canons)

        rows.append({
            "molecule": w["molecule"],
            "smiles": smi,
            "lambda_max_EtOH": w["lambda_max_EtOH"],
            "lambda_max_MeOH": w["lambda_max_MeOH"],
            "exact_smiles_in_train": exact_in_train,
            "generic_scaffold": gen_scaf if gen_scaf else "<acyclic>",
            "generic_scaffold_in_train": gen_match,
            "specific_scaffold": spec_scaf if spec_scaf else "<acyclic>",
            "specific_scaffold_in_train": spec_match,
            "example_train_smiles_for_generic_scaffold": example_train_for_scaf,
            "nearest_tanimoto": round(nearest_tanimoto, 4),
            "nearest_train_smiles": nearest_smi,
            "nearest_train_lambda_max": (round(nearest_lmax, 2)
                                         if not np.isnan(nearest_lmax)
                                         else None),
        })

    per_mol = pd.DataFrame(rows)
    out_csv = RESULTS / "r1_13_wetlab_per_molecule.csv"
    per_mol.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv} ({len(per_mol)} rows)")

    # --- Aggregate --------------------------------------------------------
    t = per_mol["nearest_tanimoto"].values
    bins = {
        "near_duplicate_gt_0.85":   int((t > 0.85).sum()),
        "high_similar_gt_0.7":      int(((t > 0.7) & (t <= 0.85)).sum()),
        "some_similar_gt_0.5":      int(((t > 0.5) & (t <= 0.7)).sum()),
        "dissimilar_le_0.5":        int((t <= 0.5).sum()),
    }
    gen_matches = int(per_mol["generic_scaffold_in_train"].sum())
    spec_matches = int(per_mol["specific_scaffold_in_train"].sum())
    exact_dups = int(per_mol["exact_smiles_in_train"].sum())

    agg = {
        "n_wetlab_molecules": int(len(per_mol)),
        "n_train_records": int(len(tr)),
        "n_train_unique_solutes": int(len(unique_train)),
        "n_train_unique_generic_scaffolds": int(len(train_gen_set)),
        "n_train_unique_specific_scaffolds": int(len(train_spec_set)),
        "tanimoto_bins": bins,
        "generic_scaffold_match_count": gen_matches,
        "specific_scaffold_match_count": spec_matches,
        "exact_smiles_duplicate_count": exact_dups,
        "tanimoto_mean": round(float(np.mean(t)), 4),
        "tanimoto_median": round(float(np.median(t)), 4),
        "tanimoto_min": round(float(np.min(t)), 4),
        "tanimoto_max": round(float(np.max(t)), 4),
    }
    out_json = RESULTS / "r1_13_aggregate.json"
    out_json.write_text(json.dumps(agg, indent=2))
    print(f"  wrote {out_json}")

    # --- Summary markdown -------------------------------------------------
    md_lines = []
    md_lines.append("# R1-13 Step 0: Wetlab OOD Audit")
    md_lines.append("")
    md_lines.append(f"- **Wetlab molecules:** {agg['n_wetlab_molecules']}")
    md_lines.append(f"- **Training records (v3):** {agg['n_train_records']:,}")
    md_lines.append(f"- **Training unique solutes:** {agg['n_train_unique_solutes']:,}")
    md_lines.append(f"- **Training unique generic Murcko scaffolds:** {agg['n_train_unique_generic_scaffolds']:,}")
    md_lines.append(f"- **Training unique specific Murcko scaffolds:** {agg['n_train_unique_specific_scaffolds']:,}")
    md_lines.append("")
    md_lines.append("## Nearest-Train Tanimoto Similarity Bins")
    md_lines.append(f"- Near-duplicate (>0.85): **{bins['near_duplicate_gt_0.85']}**")
    md_lines.append(f"- High similar (0.7--0.85): **{bins['high_similar_gt_0.7']}**")
    md_lines.append(f"- Some similar (0.5--0.7): **{bins['some_similar_gt_0.5']}**")
    md_lines.append(f"- Dissimilar (<=0.5): **{bins['dissimilar_le_0.5']}**")
    md_lines.append(f"- Tanimoto: mean={agg['tanimoto_mean']:.3f}, "
                    f"median={agg['tanimoto_median']:.3f}, "
                    f"min={agg['tanimoto_min']:.3f}, "
                    f"max={agg['tanimoto_max']:.3f}")
    md_lines.append("")
    md_lines.append("## Bemis-Murcko Scaffold Match")
    md_lines.append(f"- Exact SMILES duplicate of a train molecule: **{exact_dups}** / {agg['n_wetlab_molecules']}")
    md_lines.append(f"- Generic scaffold present in train: **{gen_matches}** / {agg['n_wetlab_molecules']}")
    md_lines.append(f"- Specific scaffold present in train: **{spec_matches}** / {agg['n_wetlab_molecules']}")
    md_lines.append("")
    md_lines.append("## Per-Molecule Table")
    md_lines.append("")
    md_lines.append("| Molecule | lambda_EtOH | lambda_MeOH | Exact dup | Gen-scaf in train | Spec-scaf in train | Nearest Tan | Nearest train SMILES | Nearest train lambda_max |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in per_mol.iterrows():
        md_lines.append(
            f"| {r['molecule']} | {r['lambda_max_EtOH']:.0f} | "
            f"{r['lambda_max_MeOH']:.0f} | "
            f"{'Y' if r['exact_smiles_in_train'] else 'N'} | "
            f"{'Y' if r['generic_scaffold_in_train'] else 'N'} | "
            f"{'Y' if r['specific_scaffold_in_train'] else 'N'} | "
            f"{r['nearest_tanimoto']:.3f} | "
            f"`{r['nearest_train_smiles']}` | "
            f"{r['nearest_train_lambda_max']} |"
        )
    md_lines.append("")
    out_md = RESULTS / "r1_13_summary.md"
    out_md.write_text("\n".join(md_lines))
    print(f"  wrote {out_md}")

    # --- Print key numbers ------------------------------------------------
    print("\n  === KEY NUMBERS ===")
    print(f"  Tanimoto bins:           {bins}")
    print(f"  Generic scaffold match:  {gen_matches} / {len(per_mol)}")
    print(f"  Specific scaffold match: {spec_matches} / {len(per_mol)}")
    print(f"  Exact SMILES duplicates: {exact_dups} / {len(per_mol)}")
    print(f"  Tanimoto mean = {agg['tanimoto_mean']:.3f}, "
          f"median = {agg['tanimoto_median']:.3f}, "
          f"min = {agg['tanimoto_min']:.3f}, "
          f"max = {agg['tanimoto_max']:.3f}")


if __name__ == "__main__":
    main()
