#!/usr/bin/env python3
"""Generate a 4x4 grid image of 2D molecule structures for the 16 wetlab validation compounds."""

import csv
from collections import OrderedDict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image


DATA_PATH = Path(__file__).parent / "data" / "wetlab_experimental.csv"
OUTPUT_PATH = Path(__file__).parent / "results" / "wetlab_molecules.png"

# Desired order for the grid (alphabetical)
MOLECULE_ORDER = [
    "Amiloxate",
    "Avobenzone",
    "Caffeic Acid",
    "Cinnamic Acid",
    "Coumaric Acid",
    "Dioxybenzone",
    "Ferulic Acid",
    "Homosalate",
    "Octinoxate",
    "Octisalate",
    "Octocrylene",
    "Oxybenzone",
    "PABA",
    "Padimate O",
    "Sinapic Acid",
    "Sulisobenzone",
]


def load_molecules(csv_path: Path) -> OrderedDict:
    """Read CSV and return {name: SMILES} keeping one entry per molecule."""
    seen = OrderedDict()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["molecule"].strip()
            if name not in seen:
                seen[name] = row["smiles"].strip()
    return seen


def main():
    smiles_map = load_molecules(DATA_PATH)

    # Build molecule list in desired order
    mols = []
    legends = []
    for name in MOLECULE_ORDER:
        smi = smiles_map.get(name)
        if smi is None:
            raise ValueError(f"Molecule '{name}' not found in {DATA_PATH}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES for '{name}': {smi}")
        mols.append(mol)
        legends.append(name)

    print(f"Loaded {len(mols)} molecules")

    # Generate grid image
    cell_size = (300, 300)
    n_cols = 4
    n_rows = 4

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=cell_size,
        legends=legends,
        useSVG=False,
    )

    # MolsToGridImage returns a PIL Image; save at 300 DPI
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(OUTPUT_PATH), dpi=(300, 300))
    print(f"Saved grid image to {OUTPUT_PATH}  ({img.size[0]}x{img.size[1]} px)")


if __name__ == "__main__":
    main()
