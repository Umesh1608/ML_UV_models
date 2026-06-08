#!/usr/bin/env python3
"""
BiGRU Gradient Saliency Analysis for UV λ_max Prediction.

Computes InputxGradient saliency per SMILES character, maps to atoms via
RDKit's _smilesAtomOutputOrder, and generates publication-quality 2D
molecular heatmaps using SimilarityMaps.

Outputs:
  results/bigru_saliency_atoms.pdf     — Grid of molecules with atom-level heatmaps (MAIN TEXT)
  results/bigru_saliency_chartype.pdf  — Importance by character type (SUPPLEMENTARY)

Usage:
  python3 analyze_bigru_saliency.py [--tuned]
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from rdkit.Chem import AllChem

# ─── GPU / CUDA setup ────────────────────────────────────────────────────────
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/home/umesh/.local/cuda_compat"
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "wetlab_experimental.csv")
N_FOLDS = 5


def load_config_and_models(use_tuned=False):
    """Load BiGRU config and fold models."""
    import tensorflow as tf

    if use_tuned:
        config_path = os.path.join(RESULTS_DIR, "bigru_tuned_config.json")
        model_prefix = "bigru_tuned"
    else:
        config_path = os.path.join(RESULTS_DIR, "bigru_solvent_v2_config.json")
        model_prefix = "bigru_solvent_v2"

    with open(config_path) as f:
        config = json.load(f)

    models = []
    for fold in range(N_FOLDS):
        model_path = os.path.join(RESULTS_DIR, f"{model_prefix}_fold{fold}_model.keras")
        if not os.path.exists(model_path):
            print(f"  WARNING: Missing fold {fold} model: {model_path}")
            continue
        model = tf.keras.models.load_model(model_path)
        models.append(model)

    print(f"  Loaded {len(models)} fold models ({model_prefix})")
    return config, models


def compute_saliency(model, X_input):
    """Compute InputxGradient saliency for each character position.

    Returns:
        saliency: np.ndarray of shape (n_samples, seq_len), non-negative importance scores
    """
    import tensorflow as tf

    X_tensor = tf.constant(X_input, dtype=tf.int32)

    with tf.GradientTape() as tape:
        # Get embedding output and watch it
        embedding_layer = model.layers[0]
        embedding_output = embedding_layer(X_tensor)
        tape.watch(embedding_output)

        # Forward pass through remaining layers
        x = embedding_output
        for layer in model.layers[1:]:
            x = layer(x)
        output = x

    grads = tape.gradient(output, embedding_output)
    # InputxGradient: element-wise product, then sum over embedding dim
    saliency = tf.abs(tf.reduce_sum(grads * embedding_output, axis=-1))
    return saliency.numpy()  # (n_samples, seq_len)


def ensemble_saliency(X_input, models):
    """Average saliency across fold models."""
    all_saliency = []
    all_preds = []
    for i, model in enumerate(models):
        sal = compute_saliency(model, X_input)
        pred = model.predict(X_input, verbose=0).flatten()
        all_saliency.append(sal)
        all_preds.append(pred)
        print(f"    Fold {i}: saliency computed")

    avg_saliency = np.mean(all_saliency, axis=0)
    avg_preds = np.mean(all_preds, axis=0)
    return avg_saliency, avg_preds


def smiles_char_to_atom_map(smiles):
    """Map SMILES character positions to atom indices using RDKit.

    Returns:
        char_to_atom: dict mapping character position (0-indexed in SMILES string)
                      to atom index. Only atom characters are mapped; syntax chars
                      (brackets, digits, bonds, etc.) are excluded.
        mol: RDKit Mol object
        canon_smiles: canonical SMILES used
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    # Generate canonical SMILES and get atom output order
    canon = Chem.MolToSmiles(mol)
    # Re-parse to get the atom order property
    mol2 = Chem.MolFromSmiles(canon)
    if mol2 is None:
        return None, None, None
    # MolToSmiles sets _smilesAtomOutputOrder on the mol
    canon2 = Chem.MolToSmiles(mol2)
    atom_order = list(mol2.GetPropsAsDict().get('_smilesAtomOutputOrder', []))

    if not atom_order:
        # Fallback: parse SMILES manually
        atom_order = _parse_atom_order(canon2, mol2.GetNumAtoms())

    # Now map character positions to atoms
    # Atom characters in SMILES: uppercase letters (C,N,O,S,P,B,F,I) and
    # lowercase aromatic (c,n,o,s,p,b). Also handle bracket atoms [...]
    char_to_atom = {}
    atom_idx = 0  # index into atom_order
    i = 0
    smi = canon2

    while i < len(smi) and atom_idx < len(atom_order):
        ch = smi[i]

        if ch == '[':
            # Bracket atom: all chars from [ to ] map to this atom
            j = smi.index(']', i)
            for k in range(i, j + 1):
                char_to_atom[k] = atom_order[atom_idx]
            atom_idx += 1
            i = j + 1
        elif ch in 'BCNOPSFIbcnops':
            # Possible atom character
            # Check for two-letter elements: Br, Cl, Si, Se, etc.
            if i + 1 < len(smi) and ch + smi[i + 1] in ('Br', 'Cl', 'Si', 'Se', 'Te', 'se', 'te'):
                char_to_atom[i] = atom_order[atom_idx]
                char_to_atom[i + 1] = atom_order[atom_idx]
                atom_idx += 1
                i += 2
            else:
                char_to_atom[i] = atom_order[atom_idx]
                atom_idx += 1
                i += 1
        else:
            # Syntax character: bond symbols, digits, parentheses, etc.
            i += 1

    return char_to_atom, mol2, canon2


def _parse_atom_order(smiles, n_atoms):
    """Fallback: generate sequential atom order."""
    return list(range(n_atoms))


def aggregate_to_atoms(char_saliency, smiles_in_input, char_to_atom_map, n_atoms):
    """Aggregate character-level saliency to atom-level.

    Args:
        char_saliency: 1D array of saliency values for the input sequence
        smiles_in_input: the SMILES string as it appears in the input (with ! prefix)
        char_to_atom_map: dict from smiles_char_to_atom_map (positions in canonical SMILES)
        n_atoms: number of atoms in the molecule

    Returns:
        atom_weights: array of shape (n_atoms,) with mean saliency per atom
    """
    atom_saliency = {i: [] for i in range(n_atoms)}

    for char_pos, atom_idx in char_to_atom_map.items():
        if atom_idx < n_atoms:
            # char_pos is in the canonical SMILES; in input, offset by 1 (! prefix)
            input_pos = char_pos + 1  # +1 for the '!' start token
            if input_pos < len(char_saliency):
                atom_saliency[atom_idx].append(char_saliency[input_pos])

    atom_weights = np.zeros(n_atoms)
    for atom_idx in range(n_atoms):
        vals = atom_saliency[atom_idx]
        if vals:
            atom_weights[atom_idx] = np.mean(vals)

    # Normalize to [0, 1]
    if atom_weights.max() > 0:
        atom_weights = atom_weights / atom_weights.max()

    return atom_weights


def _draw_similarity_map(mol, weights, size=(400, 400)):
    """Draw a SimilarityMap and return as a PIL Image.

    Uses the new RDKit API with draw2d (MolDraw2DCairo or MolDraw2DSVG).
    """
    from rdkit.Chem.Draw import SimilarityMaps
    from io import BytesIO

    d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, weights, draw2d=d2d, colorMap='RdBu_r', alpha=0.5
    )
    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()
    return png_data


def plot_saliency_grid(molecules_data, out_path):
    """Grid of 6 molecules as 2D structures with atom-level heatmaps.

    Layout: 2 rows x 3 columns with clear group headers.
    Top row = well-predicted (low |error|), bottom row = poorly-predicted (high |error|).
    Green borders for good predictions, red borders for poor predictions.
    Error magnitude shown prominently.

    molecules_data: list of dicts with keys:
        mol, atom_weights, name, actual, predicted, error, solvent
    """
    from io import BytesIO
    from PIL import Image

    # Sort by absolute error and split into two groups
    sorted_data = sorted(molecules_data, key=lambda d: abs(d['error']))
    n = len(sorted_data)
    mid = n // 2
    good_group = sorted_data[:mid]     # low error
    poor_group = sorted_data[mid:]     # high error
    # Sort poor group by increasing error for visual progression
    poor_group = sorted(poor_group, key=lambda d: abs(d['error']))

    ncols = max(len(good_group), len(poor_group))
    nrows = 2

    fig = plt.figure(figsize=(5.5 * ncols, 6.5 * nrows + 1.2))
    gs = GridSpec(nrows + 2, ncols, figure=fig,
                  height_ratios=[0.08, 1, 0.08, 1],
                  hspace=0.15, wspace=0.15)

    # Group headers
    ax_header_good = fig.add_subplot(gs[0, :])
    ax_header_good.set_xlim(0, 1)
    ax_header_good.set_ylim(0, 1)
    ax_header_good.text(0.5, 0.5,
                        r'$\bf{Well\ Predicted}$' + f'  (|error| < {abs(good_group[-1]["error"]):.0f} nm)',
                        ha='center', va='center', fontsize=15,
                        color='#1a7a2e',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4edda',
                                  edgecolor='#1a7a2e', linewidth=1.5))
    ax_header_good.axis('off')

    ax_header_poor = fig.add_subplot(gs[2, :])
    ax_header_poor.set_xlim(0, 1)
    ax_header_poor.set_ylim(0, 1)
    ax_header_poor.text(0.5, 0.5,
                        r'$\bf{Poorly\ Predicted}$' + f'  (|error| > {abs(poor_group[0]["error"]):.0f} nm)',
                        ha='center', va='center', fontsize=15,
                        color='#a71d2a',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8d7da',
                                  edgecolor='#a71d2a', linewidth=1.5))
    ax_header_poor.axis('off')

    def _render_molecule(ax, data, border_color):
        mol = data['mol']
        weights = [float(data['atom_weights'][i]) for i in range(mol.GetNumAtoms())]

        # Render with RDKit
        png_data = _draw_similarity_map(mol, weights, size=(500, 500))
        img = Image.open(BytesIO(png_data))
        ax.imshow(img)
        ax.axis('off')

        abs_err = abs(data['error'])
        sign = "+" if data['error'] >= 0 else "\u2212"

        # Title: molecule name + solvent
        ax.set_title(f"{data['name']} ({data['solvent']})",
                      fontsize=13, fontweight='bold', pad=10)

        # Subtitle: actual, predicted, error — error in color
        err_color = '#1a7a2e' if abs_err < 20 else '#d4880f' if abs_err < 40 else '#a71d2a'
        ax.text(0.5, -0.02,
                f"Exp: {data['actual']:.0f} nm  |  Pred: {data['predicted']:.0f} nm",
                ha='center', va='top', transform=ax.transAxes, fontsize=11)
        ax.text(0.5, -0.07,
                f"Error: {sign}{abs_err:.0f} nm",
                ha='center', va='top', transform=ax.transAxes, fontsize=12,
                fontweight='bold', color=err_color)

        # Colored border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    # Plot good predictions (top row)
    for col, data in enumerate(good_group):
        ax = fig.add_subplot(gs[1, col])
        _render_molecule(ax, data, border_color='#28a745')

    # Hide unused good slots
    for col in range(len(good_group), ncols):
        ax = fig.add_subplot(gs[1, col])
        ax.axis('off')

    # Plot poor predictions (bottom row)
    for col, data in enumerate(poor_group):
        ax = fig.add_subplot(gs[3, col])
        _render_molecule(ax, data, border_color='#dc3545')

    # Hide unused poor slots
    for col in range(len(poor_group), ncols):
        ax = fig.add_subplot(gs[3, col])
        ax.axis('off')

    for ext in ('pdf', 'png'):
        plt.savefig(out_path.replace('.pdf', f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saliency grid saved to {out_path}")


def plot_chartype_importance(all_saliency, all_combined_smiles, char_to_int, out_path):
    """Aggregated importance by character type: aromatic, aliphatic, heteroatom, bond, syntax.

    Produces a bar chart for the supplementary material.
    """
    # Define character categories
    categories = {
        'Aromatic atoms': set('cnops'),
        'Aliphatic atoms': set('CNOPS'),
        'Halogens/metals': set('FIBb'),
        'Bond symbols': set('=#/\\'),
        'Ring/branch': set('()0123456789'),
        'Brackets': set('[]@+-.'),
        'Heteroatoms (N,O,S)': None,  # computed separately
    }

    # Simpler categorization
    cat_names = ['Aromatic C/hetero', 'Aliphatic C', 'Heteroatoms (N,O,S)',
                 'Bond symbols (=,#)', 'Ring digits', 'Brackets/syntax']
    cat_sets = [
        set('cp'),          # aromatic carbon + aromatic p
        set('C'),           # aliphatic carbon
        set('NnOoSs'),      # heteroatoms
        set('=#/\\'),       # bonds
        set('0123456789%'), # ring
        set('[]()@+-.!E'),  # syntax
    ]

    int_to_char = {v: k for k, v in char_to_int.items()}
    cat_saliencies = {name: [] for name in cat_names}

    for sample_idx in range(len(all_combined_smiles)):
        smi = all_combined_smiles[sample_idx]
        sal = all_saliency[sample_idx]

        for pos in range(min(len(smi) + 1, len(sal))):
            if pos == 0:
                # Start token '!'
                cat_saliencies['Brackets/syntax'].append(sal[pos])
                continue

            char_idx = pos - 1
            if char_idx >= len(smi):
                break
            ch = smi[char_idx]
            val = sal[pos]

            assigned = False
            for cat_name, cat_set in zip(cat_names, cat_sets):
                if ch in cat_set:
                    cat_saliencies[cat_name].append(val)
                    assigned = True
                    break
            if not assigned:
                cat_saliencies['Brackets/syntax'].append(val)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(cat_saliencies[n]) if cat_saliencies[n] else 0 for n in cat_names]
    stds = [np.std(cat_saliencies[n]) / np.sqrt(max(len(cat_saliencies[n]), 1))
            for n in cat_names]

    colors = ['#e74c3c', '#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#95a5a6']
    y_pos = np.arange(len(cat_names))

    ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor='white',
            height=0.6, capsize=3, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Saliency (InputxGradient)', fontsize=12)
    ax.set_title('BiGRU Character-Level Saliency by Type\n(averaged over wetlab molecules)',
                 fontsize=13)

    # Add count annotations
    for i, name in enumerate(cat_names):
        n = len(cat_saliencies[name])
        ax.text(means[i] + stds[i] + 0.001, i, f'n={n}',
                va='center', fontsize=8, color='#555555')

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(out_path.replace('.pdf', f'.{ext}'), dpi=300)
    plt.close()
    print(f"  Character-type importance saved to {out_path}")


def select_molecules(df, preds, errors):
    """Select 6 molecules for the main figure.

    Selection:
    - 2 with lowest absolute error (BiGRU predicts well)
    - 2 with highest absolute error (BiGRU predicts poorly)
    - 2 with intermediate error (representative)
    """
    abs_errors = np.abs(errors)
    sorted_idx = np.argsort(abs_errors)

    selected = set()
    selected_indices = []

    # 2 best predictions
    for idx in sorted_idx:
        mol_name = df.iloc[idx]['molecule']
        if mol_name not in selected:
            selected.add(mol_name)
            selected_indices.append(idx)
            if len(selected_indices) >= 2:
                break

    # 2 worst predictions
    for idx in sorted_idx[::-1]:
        mol_name = df.iloc[idx]['molecule']
        if mol_name not in selected:
            selected.add(mol_name)
            selected_indices.append(idx)
            if len(selected_indices) >= 4:
                break

    # 2 intermediate
    mid = len(sorted_idx) // 2
    for idx in sorted_idx[mid-5:mid+5]:
        mol_name = df.iloc[idx]['molecule']
        if mol_name not in selected:
            selected.add(mol_name)
            selected_indices.append(idx)
            if len(selected_indices) >= 6:
                break

    return selected_indices


def main():
    parser = argparse.ArgumentParser(description="BiGRU saliency analysis")
    parser.add_argument("--tuned", action="store_true",
                        help="Use tuned BiGRU models instead of v2 default")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  BiGRU GRADIENT SALIENCY ANALYSIS")
    print(f"{'=' * 70}")

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} wetlab measurements ({df['molecule'].nunique()} molecules)")

    # Load models
    config, models = load_config_and_models(use_tuned=args.tuned)
    char_to_int = config["char_to_int"]
    embed_len = config["embed_len"]

    # Vectorize SMILES
    from paper1_new_cl.models import vectorize_smiles
    combined = (df["smiles"] + "!" + df["solvent_smiles"]).values
    X = vectorize_smiles(combined, char_to_int, embed_len)

    # Compute ensemble saliency
    print(f"\n{'─' * 70}")
    print("  Computing ensemble saliency...")
    avg_saliency, avg_preds = ensemble_saliency(X, models)
    y_true = df["lambda_max_exp"].values
    errors = avg_preds - y_true

    print(f"  BiGRU ensemble MAE: {np.mean(np.abs(errors)):.2f} nm")

    # ── Map saliency to atoms ──
    print(f"\n{'─' * 70}")
    print("  Mapping character saliency to atoms...")

    all_mol_data = []
    for idx in range(len(df)):
        smiles = df.iloc[idx]['smiles']
        char_to_atom, mol, canon = smiles_char_to_atom_map(smiles)
        if mol is None or char_to_atom is None:
            print(f"  WARNING: Could not parse {smiles}")
            continue

        atom_weights = aggregate_to_atoms(
            avg_saliency[idx], combined[idx], char_to_atom, mol.GetNumAtoms()
        )

        all_mol_data.append({
            'idx': idx,
            'mol': mol,
            'atom_weights': atom_weights,
            'name': df.iloc[idx]['molecule'],
            'solvent': df.iloc[idx]['solvent_name'],
            'actual': y_true[idx],
            'predicted': avg_preds[idx],
            'error': errors[idx],
            'smiles': smiles,
            'canon': canon,
        })

    print(f"  Successfully mapped {len(all_mol_data)}/{len(df)} molecules")

    # ── Select molecules for main figure ──
    sel_indices = select_molecules(df, avg_preds, errors)
    selected_data = [d for d in all_mol_data if d['idx'] in sel_indices]

    # Sort: best predictions first, then worst
    selected_data.sort(key=lambda d: abs(d['error']))

    print(f"\n  Selected {len(selected_data)} molecules for main figure:")
    for d in selected_data:
        sign = "+" if d['error'] >= 0 else ""
        print(f"    {d['name']:20s} ({d['solvent']}) — "
              f"actual={d['actual']:.0f}, pred={d['predicted']:.1f}, "
              f"err={sign}{d['error']:.1f}")

    # ── Generate main figure ──
    print(f"\n{'─' * 70}")
    print("  Generating saliency grid figure...")
    grid_path = os.path.join(RESULTS_DIR, "bigru_saliency_atoms.pdf")
    plot_saliency_grid(selected_data, grid_path)

    # ── Generate supplementary character-type figure ──
    print("  Generating character-type importance figure...")
    chartype_path = os.path.join(RESULTS_DIR, "bigru_saliency_chartype.pdf")
    plot_chartype_importance(avg_saliency, combined.tolist(), char_to_int, chartype_path)

    # ── Print atom-level summary for cross-validation with RF ──
    print(f"\n{'─' * 70}")
    print("  ATOM-LEVEL SALIENCY SUMMARY (top atoms per molecule)")
    print(f"{'─' * 70}")
    for d in selected_data:
        mol = d['mol']
        w = d['atom_weights']
        top_atoms = np.argsort(w)[::-1][:5]
        print(f"\n  {d['name']} ({d['solvent']}):")
        for rank, ai in enumerate(top_atoms, 1):
            atom = mol.GetAtomWithIdx(int(ai))
            symbol = atom.GetSymbol()
            is_aromatic = atom.GetIsAromatic()
            ar_str = " (aromatic)" if is_aromatic else ""
            print(f"    #{rank}: Atom {ai} ({symbol}{ar_str}) — saliency={w[ai]:.3f}")

    # Clean up GPU memory
    import tensorflow as tf
    for model in models:
        del model
    tf.keras.backend.clear_session()

    print(f"\n{'=' * 70}")
    print(f"  SALIENCY ANALYSIS COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
