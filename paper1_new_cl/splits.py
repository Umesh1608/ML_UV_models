"""
Scaffold split utility for UV absorption datasets.

Implements Bemis-Murcko scaffold-based splitting for direct comparison
with published results that use scaffold splits.
"""

from collections import Counter, defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold, train_test_split


def get_scaffold(smiles):
    """Get generic Bemis-Murcko scaffold for a SMILES string.

    Returns the scaffold SMILES, or the original SMILES on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        # Convert to generic scaffold (all atoms → C, all bonds → single)
        generic = MurckoScaffold.MakeScaffoldGeneric(
            Chem.MolFromSmiles(scaffold)
        )
        return Chem.MolToSmiles(generic)
    except Exception:
        return smiles


def scaffold_split(smiles_list, train_frac=0.8, val_frac=0.1, test_frac=0.1,
                   seed=7):
    """Single Bemis-Murcko scaffold split.

    Groups molecules by their generic scaffold, sorts groups largest-first,
    then greedily assigns groups to train/val/test to match target fractions.

    Args:
        smiles_list: Array-like of SMILES strings.
        train_frac: Target fraction for training set.
        val_frac: Target fraction for validation set.
        test_frac: Target fraction for test set.
        seed: Random seed for shuffling scaffold groups of equal size.

    Returns:
        (train_idx, val_idx, test_idx) — numpy arrays of integer indices.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    n = len(smiles_list)

    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(str(smi))
        scaffold_to_indices[scaffold].append(i)

    # Sort scaffold groups: largest first, then shuffle within same-size groups
    rng = np.random.RandomState(seed)
    scaffold_groups = list(scaffold_to_indices.values())
    # Add random tiebreaker for deterministic ordering of same-size groups
    scaffold_groups.sort(key=lambda g: (-len(g), rng.random()))

    train_idx, val_idx, test_idx = [], [], []
    train_target = int(n * train_frac)
    val_target = int(n * (train_frac + val_frac))

    for group in scaffold_groups:
        if len(train_idx) < train_target:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) < val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    print(f"  Scaffold split: {len(scaffold_to_indices)} unique scaffolds")
    print(f"  Train: {len(train_idx)} ({len(train_idx)/n:.1%}), "
          f"Val: {len(val_idx)} ({len(val_idx)/n:.1%}), "
          f"Test: {len(test_idx)} ({len(test_idx)/n:.1%})")

    return train_idx, val_idx, test_idx


# ─── Solvent-stratified splitting ────────────────────────────────────────────


def create_solvent_bins(solvents, top_n=20):
    """Assign each sample to a solvent stratum bin.

    The top ``top_n`` solvents (by frequency) each get their own bin label;
    all remaining solvents are lumped into an ``"other"`` bin.

    Args:
        solvents: Array-like of solvent SMILES strings.
        top_n: Number of top solvents to keep as individual bins.

    Returns:
        numpy array of string bin labels, same length as *solvents*.
    """
    counts = Counter(solvents)
    top_solvents = {s for s, _ in counts.most_common(top_n)}
    bins = np.array([s if s in top_solvents else "other" for s in solvents])
    return bins


def create_stratified_folds(n_samples, solvents, n_folds=5, seed=7):
    """Create solvent-stratified train/val/test folds.

    Outer split via ``StratifiedKFold`` (80 % train+val / 20 % test), then an
    inner ``train_test_split(stratify=...)`` to carve out a validation set.
    Final approximate ratios per fold: 64 % train / 16 % val / 20 % test.

    Args:
        n_samples: Total number of samples.
        solvents: Array-like of solvent strings (length *n_samples*).
        n_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        List of ``(train_idx, val_idx, test_idx)`` tuples (numpy arrays).
    """
    bins = create_solvent_bins(solvents)
    indices = np.arange(n_samples)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []

    for trainval_idx, test_idx in skf.split(indices, bins):
        # Inner split: 80 % train, 20 % val (of the trainval portion)
        trainval_bins = bins[trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=0.2, random_state=seed,
            stratify=trainval_bins,
        )
        folds.append((train_idx, val_idx, test_idx))

    return folds
