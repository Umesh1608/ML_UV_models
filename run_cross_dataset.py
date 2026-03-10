#!/usr/bin/env python3
"""
Cross-dataset benchmarking: train BiGRU+Solvent and RF on external UV absorption datasets.

Trains BiGRU+Solvent and Random Forest models on Deep4Chem, Jung 2024, and nablaColors
datasets for direct comparison with published results.

Split protocols match each paper's published method:
  - Deep4Chem (Joung 2020): Single random 80/10/10 (train/val/test)
  - Jung 2024: Single random 72/18/10 (90/10 outer, 20% of train as val)
               + scaffold 80/10/10
  - nablaColors: Author-provided precomputed split

Usage:
  python3 run_cross_dataset.py --dataset deep4chem                         # BiGRU random 80/10/10
  python3 run_cross_dataset.py --dataset deep4chem --model rf              # RF random 80/10/10
  python3 run_cross_dataset.py --dataset jung2024                          # random 72/18/10
  python3 run_cross_dataset.py --dataset jung2024 --split scaffold         # scaffold 80/10/10
  python3 run_cross_dataset.py --dataset nablacolors --split precomputed   # author-defined split
  python3 run_cross_dataset.py --summary                                   # comparison table
"""

import argparse
import gc
import json
import os
import sys
import time

# ─── GPU / CUDA setup ────────────────────────────────────────────────────────
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/home/umesh/.local/cuda_compat"
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from paper1_new_cl.models import (
    build_char_maps,
    compute_metrics,
    compute_morgan_fps,
    train_dl_model,
    vectorize_smiles,
)
from paper1_new_cl.splits import scaffold_split

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_BASE = os.path.join(SCRIPT_DIR, "results")

SEED = 7
EPOCHS = 250
BATCH_SIZE = 32
PATIENCE = 25

DATASET_INFO = {
    "deep4chem": {
        "file": os.path.join(DATA_DIR, "deep4chem_processed.csv"),
        "display": "Deep4Chem (Joung 2020)",
        "published": {"method": "GCNN", "RMSE": 26.6, "source": "Joung 2021"},
        # Joung 2021 JACS Au: GCNN on full 30,094 records; split: random 80/10/10
        "random_split": {"train": 0.8, "val": 0.1, "test": 0.1},
    },
    "jung2024": {
        "file": os.path.join(DATA_DIR, "jung2024_processed.csv"),
        "display": "Jung et al. 2024",
        "published": {"method": "GBFS", "RMSE": 32.2, "source": "Jung 2024"},
        # Jung 2024: 90/10 train/test, then 20% of train as val → 72/18/10
        "random_split": {"train": 0.72, "val": 0.18, "test": 0.10},
    },
    "nablacolors": {
        "file": os.path.join(DATA_DIR, "nablacolors_processed.csv"),
        "display": "nablaColors (Potapov 2026)",
        "splits_file": os.path.join(DATA_DIR, "nablacolors_splits.npz"),
        "published": {"method": "UniProp (3D GNN, 27.7M params)", "MAE": 15.97, "source": "Potapov 2026"},
    },
}


# ─── Data loading ──────────────────────────────────────────────────────────

def load_dataset(dataset_key):
    """Load a preprocessed dataset CSV. Returns dict with combined SMILES, y, charset, etc."""
    info = DATASET_INFO[dataset_key]
    path = info["file"]

    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        print(f"  Run: python3 download_datasets.py --dataset {dataset_key}")
        sys.exit(1)

    print(f"\n[DATA] Loading {info['display']} from {path}...")
    df = pd.read_csv(path)
    df = df.dropna(subset=["smiles", "solvent_smiles", "lambda_max"])

    # Build combined SMILES (solute!solvent)
    combined = (df["smiles"] + "!" + df["solvent_smiles"]).values
    smiles = df["smiles"].values
    solvents = df["solvent_smiles"].values
    y = df["lambda_max"].values.astype(np.float64)

    # Compute charset from ALL data
    charset = sorted(set("".join(combined) + "!E"))
    embed_len = max(len(s) for s in combined) + 5

    print(f"  Samples: {len(df)}")
    print(f"  Unique solutes: {df['smiles'].nunique()}")
    print(f"  Unique solvents: {df['solvent_smiles'].nunique()}")
    print(f"  Charset: {len(charset)} tokens, max seq len: {embed_len}")
    print(f"  λ_max: {y.min():.1f} – {y.max():.1f} nm (mean {y.mean():.1f})")

    return {
        "combined": combined,
        "smiles": smiles,
        "solvents": solvents,
        "y": y,
        "charset": charset,
        "embed_len": embed_len,
    }


# ─── Results I/O ──────────────────────────────────────────────────────────

def get_results_dir(dataset_key):
    """Results directory for a dataset."""
    d = os.path.join(RESULTS_BASE, dataset_key)
    os.makedirs(d, exist_ok=True)
    return d


def _save_results(results_dir, prefix, metrics, y_pred, y_test, test_indices):
    with open(os.path.join(results_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, f"{prefix}_predictions.npy"), y_pred)
    np.save(os.path.join(results_dir, f"{prefix}_y_test.npy"), y_test)
    np.save(os.path.join(results_dir, f"{prefix}_test_indices.npy"), test_indices)


def _load_metrics(results_dir, prefix):
    path = os.path.join(results_dir, f"{prefix}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_config(results_dir, data, split_type):
    """Save dataset config for reproducibility."""
    config = {
        "charset": data["charset"],
        "embed_len": data["embed_len"],
        "char_to_int": build_char_maps(data["charset"]),
        "model_type": "bigru",
        "with_solvent": True,
        "split_type": split_type,
        "seed": SEED,
    }
    with open(os.path.join(results_dir, "bigru_solvent_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return config


# ─── Fingerprint helpers ──────────────────────────────────────────────────

def compute_cross_dataset_fps(data):
    """Compute 2048-bit Morgan FPs for cross-dataset. Returns (fp_array, valid_mask)."""
    print("  Computing Morgan fingerprints (solute)...")
    fp_solute, mask1 = compute_morgan_fps(list(data["smiles"]))
    print("  Computing Morgan fingerprints (solvent)...")
    fp_solvent, mask2 = compute_morgan_fps(list(data["solvents"]))
    fp_all = np.hstack([fp_solute, fp_solvent])
    valid_mask = mask1 & mask2
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  WARNING: {n_invalid} samples with invalid SMILES (will be excluded)")
    return fp_all, valid_mask


def train_rf_split(data, train_idx, val_idx, test_idx, name, results_dir,
                   fp_all, fp_valid_mask):
    """Train RF on Morgan FPs, evaluate on test split. Val is unused (RF has no early stopping)."""
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    # Combine train+val for RF (no early stopping needed)
    train_val_idx = np.concatenate([train_idx, val_idx])

    # Filter by valid mask
    train_mask = fp_valid_mask[train_val_idx]
    test_mask = fp_valid_mask[test_idx]

    X_train = fp_all[train_val_idx][train_mask]
    y_train = data["y"][train_val_idx][train_mask]
    X_test = fp_all[test_idx][test_mask]
    y_test = data["y"][test_idx][test_mask]

    print(f"  RF training: {len(X_train)} samples ({(~train_mask).sum()} excluded)")
    print(f"  RF test:     {len(X_test)} samples ({(~test_mask).sum()} excluded)")

    import time as _time
    t0 = _time.time()
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED)
    model.fit(X_train, y_train)
    elapsed = _time.time() - t0

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print(f"  RF done in {elapsed:.0f}s — RMSE={metrics['RMSE']:.2f} "
          f"MAE={metrics['MAE']:.2f} R²={metrics['R2']:.4f}")

    # Save model
    model_path = os.path.join(results_dir, f"{name}_model.joblib")
    joblib.dump(model, model_path)
    print(f"  >> Model saved to {model_path}")

    return y_pred, metrics


# ─── Train one split ──────────────────────────────────────────────────────

def train_split(data, train_idx, val_idx, test_idx, name, results_dir):
    """Train BiGRU+Solvent with val for early stopping, evaluate on test."""
    import tensorflow as tf

    charset = data["charset"]
    embed_len = data["embed_len"]
    c2i = build_char_maps(charset)

    X_train_smiles = data["combined"][train_idx]
    X_val_smiles = data["combined"][val_idx]
    X_test_smiles = data["combined"][test_idx]
    y_train = data["y"][train_idx]
    y_val = data["y"][val_idx]
    y_test = data["y"][test_idx]

    print(f"  Vectorizing {len(X_train_smiles)} train + {len(X_val_smiles)} val + {len(X_test_smiles)} test...")
    X_train_vec = vectorize_smiles(X_train_smiles, c2i, embed_len)
    X_val_vec = vectorize_smiles(X_val_smiles, c2i, embed_len)
    X_test_vec = vectorize_smiles(X_test_smiles, c2i, embed_len)

    num_words = len(charset)
    input_length = embed_len - 1

    y_pred, metrics, model = train_dl_model(
        "bigru", X_train_vec, X_test_vec,
        y_train.reshape(-1, 1), y_test.reshape(-1, 1),
        num_words=num_words, input_length=input_length, name=name,
        results_dir=results_dir, seed=SEED, epochs=EPOCHS,
        batch_size=BATCH_SIZE, patience=PATIENCE,
        X_val=X_val_vec, y_val=y_val.reshape(-1, 1)
    )

    # Save model
    model_path = os.path.join(results_dir, f"{name}_model.keras")
    model.save(model_path)
    print(f"  >> Model saved to {model_path}")

    # Cleanup
    del model, X_train_vec, X_val_vec, X_test_vec
    tf.keras.backend.clear_session()
    gc.collect()

    return y_pred, metrics


# ─── Run single random split ─────────────────────────────────────────────

def run_single_split(dataset_key, data, model="bigru"):
    """Run a single random train/val/test split matching the published protocol."""
    info = DATASET_INFO[dataset_key]
    split_cfg = info["random_split"]
    results_dir = get_results_dir(dataset_key)

    prefix = "rf_solvent_random" if model == "rf" else "bigru_solvent_random"

    # Check if already done
    existing = _load_metrics(results_dir, prefix)
    if existing is not None:
        print(f"\n  [{model.upper()} Random] Already completed (RMSE={existing['RMSE']:.2f}). Skipping.")
        return existing

    # Try to load existing split indices (reuse from BiGRU run)
    split_path = os.path.join(results_dir, "random_split_indices.npz")
    if os.path.exists(split_path):
        print(f"  Loading existing split indices from {split_path}")
        splits = np.load(split_path)
        train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    else:
        # Split: first carve off test, then split remainder into train/val
        n = len(data["y"])
        indices = np.arange(n)
        test_frac = split_cfg["test"]
        val_frac = split_cfg["val"]

        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_frac, random_state=SEED
        )
        val_frac_of_trainval = val_frac / (1.0 - test_frac)
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_frac_of_trainval, random_state=SEED
        )

        # Save split indices
        np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)

    if model == "bigru":
        save_config(results_dir, data, "random")

    print(f"\n{'─'*70}")
    print(f"  {info['display']} — {model.upper()} Random split "
          f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
    print(f"  Protocol: {split_cfg['train']:.0%}/{split_cfg['val']:.0%}/{split_cfg['test']:.0%} "
          f"(matching {info['published']['source']})")
    print(f"{'─'*70}")

    if model == "rf":
        fp_all, fp_valid_mask = compute_cross_dataset_fps(data)
        y_pred, metrics = train_rf_split(data, train_idx, val_idx, test_idx,
                                         prefix, results_dir, fp_all, fp_valid_mask)
        # Save using valid test indices only
        test_mask = fp_valid_mask[test_idx]
        _save_results(results_dir, prefix, metrics, y_pred,
                      data["y"][test_idx][test_mask], test_idx[test_mask])
    else:
        y_pred, metrics = train_split(data, train_idx, val_idx, test_idx,
                                      prefix, results_dir)
        _save_results(results_dir, prefix, metrics, y_pred,
                      data["y"][test_idx], test_idx)

    return metrics


# ─── Run scaffold split ───────────────────────────────────────────────────

def run_scaffold(dataset_key, data, model="bigru"):
    """Run single 80/10/10 scaffold split with val for early stopping."""
    results_dir = get_results_dir(dataset_key)

    prefix = "rf_solvent_scaffold" if model == "rf" else "bigru_solvent_scaffold"

    # Check if already done
    existing = _load_metrics(results_dir, prefix)
    if existing is not None:
        print(f"\n  [{model.upper()} Scaffold] Already completed (RMSE={existing['RMSE']:.2f}). Skipping.")
        return existing

    # Try to load existing split indices
    split_path = os.path.join(results_dir, "scaffold_split_indices.npz")
    if os.path.exists(split_path):
        print(f"  Loading existing scaffold split indices from {split_path}")
        splits = np.load(split_path)
        train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    else:
        print(f"\n  Computing scaffold split...")
        train_idx, val_idx, test_idx = scaffold_split(
            data["smiles"], train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=SEED
        )
        # Save split indices
        np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)

    if model == "bigru":
        save_config(results_dir, data, "scaffold")

    print(f"\n{'─'*70}")
    print(f"  {DATASET_INFO[dataset_key]['display']} — {model.upper()} Scaffold split "
          f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
    print(f"{'─'*70}")

    if model == "rf":
        fp_all, fp_valid_mask = compute_cross_dataset_fps(data)
        y_pred, metrics = train_rf_split(data, train_idx, val_idx, test_idx,
                                         prefix, results_dir, fp_all, fp_valid_mask)
        test_mask = fp_valid_mask[test_idx]
        _save_results(results_dir, prefix, metrics, y_pred,
                      data["y"][test_idx][test_mask], test_idx[test_mask])
    else:
        y_pred, metrics = train_split(data, train_idx, val_idx, test_idx,
                                      prefix, results_dir)
        _save_results(results_dir, prefix, metrics, y_pred,
                      data["y"][test_idx], test_idx)

    return metrics


# ─── Run precomputed split ───────────────────────────────────────────────

def run_precomputed(dataset_key, data, model="bigru"):
    """Run using the dataset's own pre-defined train/val/test split."""
    results_dir = get_results_dir(dataset_key)

    prefix = "rf_solvent_precomputed" if model == "rf" else "bigru_solvent_precomputed"

    # Check if already done
    existing = _load_metrics(results_dir, prefix)
    if existing is not None:
        print(f"\n  [{model.upper()} Precomputed] Already completed (RMSE={existing['RMSE']:.2f}). Skipping.")
        return existing

    # Load pre-defined split indices
    info = DATASET_INFO[dataset_key]
    splits_file = info.get("splits_file")
    if splits_file is None or not os.path.exists(splits_file):
        print(f"  ERROR: No precomputed splits file found: {splits_file}")
        print(f"  Run: python3 download_datasets.py --dataset {dataset_key}")
        sys.exit(1)

    splits = np.load(splits_file)
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    print(f"  Loaded precomputed splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Save split indices for reference
    np.savez(os.path.join(results_dir, "precomputed_split_indices.npz"),
             train=train_idx, val=val_idx, test=test_idx)

    if model == "bigru":
        save_config(results_dir, data, "precomputed")

    print(f"\n{'─'*70}")
    print(f"  {info['display']} — {model.upper()} Precomputed split "
          f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
    print(f"{'─'*70}")

    if model == "rf":
        fp_all, fp_valid_mask = compute_cross_dataset_fps(data)
        y_pred, metrics = train_rf_split(data, train_idx, val_idx, test_idx,
                                         prefix, results_dir, fp_all, fp_valid_mask)
        test_mask = fp_valid_mask[test_idx]
        _save_results(results_dir, prefix, metrics, y_pred,
                      data["y"][test_idx][test_mask], test_idx[test_mask])
    else:
        y_pred, metrics = train_split(data, train_idx, val_idx, test_idx,
                                      prefix, results_dir)
        _save_results(results_dir, prefix, metrics, y_pred,
                      data["y"][test_idx], test_idx)

    return metrics


# ─── Summary ───────────────────────────────────────────────────────────────

def generate_summary():
    """Generate cross-dataset comparison table from saved results."""
    print("\n" + "=" * 70)
    print("  CROSS-DATASET COMPARISON SUMMARY")
    print("=" * 70)

    rows = []

    # Joung+Beard (from main results dir)
    jb_agg_path = os.path.join(RESULTS_BASE, "bigru_solvent_cv_aggregate.json")
    if os.path.exists(jb_agg_path):
        with open(jb_agg_path) as f:
            jb_agg = json.load(f)
        rows.append({
            "Dataset": "Joung+Beard (18.7K)",
            "Model": "BiGRU",
            "Split": "Random 5-CV",
            "RMSE": f"{jb_agg['RMSE_mean']:.2f} +/- {jb_agg['RMSE_std']:.2f}",
            "MAE": f"{jb_agg['MAE_mean']:.2f} +/- {jb_agg['MAE_std']:.2f}",
            "R2": f"{jb_agg['R2_mean']:.4f} +/- {jb_agg['R2_std']:.4f}",
            "Published": "This work",
        })

    for ds_key, info in DATASET_INFO.items():
        rdir = get_results_dir(ds_key)
        pub = info.get("published", {})
        pub_str = "—"
        if pub:
            parts = [f"{pub.get('method', '?')}:"]
            if "RMSE" in pub:
                parts.append(f"RMSE {pub['RMSE']}")
            if "MAE" in pub:
                parts.append(f"MAE {pub['MAE']}")
            pub_str = " ".join(parts)

        # Single random split — BiGRU and RF
        for model_prefix, model_label in [("bigru_solvent_random", "BiGRU"),
                                           ("rf_solvent_random", "RF")]:
            rand = _load_metrics(rdir, model_prefix)
            if rand is not None:
                split_cfg = info.get("random_split", {})
                split_label = (
                    f"Random {split_cfg.get('train', '?'):.0%}/"
                    f"{split_cfg.get('val', '?'):.0%}/"
                    f"{split_cfg.get('test', '?'):.0%}"
                ) if split_cfg else "Random"
                rows.append({
                    "Dataset": f"{info['display']}",
                    "Model": model_label,
                    "Split": split_label,
                    "RMSE": f"{rand['RMSE']:.2f}",
                    "MAE": f"{rand['MAE']:.2f}",
                    "R2": f"{rand['R2']:.4f}",
                    "Published": pub_str,
                })

        # Scaffold — BiGRU and RF
        for model_prefix, model_label in [("bigru_solvent_scaffold", "BiGRU"),
                                           ("rf_solvent_scaffold", "RF")]:
            scf = _load_metrics(rdir, model_prefix)
            if scf is not None:
                rows.append({
                    "Dataset": f"{info['display']}",
                    "Model": model_label,
                    "Split": "Scaffold 80/10/10",
                    "RMSE": f"{scf['RMSE']:.2f}",
                    "MAE": f"{scf['MAE']:.2f}",
                    "R2": f"{scf['R2']:.4f}",
                    "Published": pub_str,
                })

        # Precomputed (nablaColors) — BiGRU and RF
        for model_prefix, model_label in [("bigru_solvent_precomputed", "BiGRU"),
                                           ("rf_solvent_precomputed", "RF")]:
            pre = _load_metrics(rdir, model_prefix)
            if pre is not None:
                rows.append({
                    "Dataset": f"{info['display']}",
                    "Model": model_label,
                    "Split": "Precomputed (author)",
                    "RMSE": f"{pre['RMSE']:.2f}",
                    "MAE": f"{pre['MAE']:.2f}",
                    "R2": f"{pre['R2']:.4f}",
                    "Published": pub_str,
                })

    if not rows:
        print("  No results found! Run experiments first.")
        return

    df = pd.DataFrame(rows)
    print(f"\n{df.to_string(index=False)}")

    out_path = os.path.join(RESULTS_BASE, "cross_dataset_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset UV λ_max benchmarking")
    parser.add_argument("--dataset", choices=list(DATASET_INFO.keys()),
                        help="Which dataset to train on")
    parser.add_argument("--split", choices=["random", "scaffold", "precomputed"],
                        default="random",
                        help="Split type (default: random — matches published protocol)")
    parser.add_argument("--model", choices=["bigru", "rf"], default="bigru",
                        help="Model type (default: bigru)")
    parser.add_argument("--summary", action="store_true",
                        help="Generate comparison table from saved results")
    args = parser.parse_args()

    if args.summary:
        generate_summary()
        return

    if args.dataset is None:
        parser.error("--dataset is required (unless --summary)")

    if args.split == "random" and "random_split" not in DATASET_INFO.get(args.dataset, {}):
        parser.error(f"--split random is not available for {args.dataset} "
                     f"(no published random split protocol defined)")

    if args.split == "precomputed" and "splits_file" not in DATASET_INFO.get(args.dataset, {}):
        parser.error(f"--split precomputed is not available for {args.dataset} "
                     f"(only nablacolors has pre-defined splits)")

    model = args.model
    print("\n" + "=" * 70)
    print(f"  CROSS-DATASET BENCHMARKING: {DATASET_INFO[args.dataset]['display']}")
    print(f"  Model: {model.upper()}, Split: {args.split}")
    print("=" * 70)

    # GPU setup (only needed for BiGRU)
    if model == "bigru":
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[GPU] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[GPU] Mixed precision (float16) enabled")
        else:
            print("[WARN] No GPU detected — training will be slow")

    # Load data
    data = load_dataset(args.dataset)

    model_label = model.upper()
    if args.split == "random":
        metrics = run_single_split(args.dataset, data, model=model)
    elif args.split == "scaffold":
        metrics = run_scaffold(args.dataset, data, model=model)
    elif args.split == "precomputed":
        metrics = run_precomputed(args.dataset, data, model=model)

    if metrics:
        pub = DATASET_INFO[args.dataset].get("published", {})
        split_label = args.split
        print(f"\n  {model_label} on {DATASET_INFO[args.dataset]['display']} ({split_label}):")
        print(f"    RMSE: {metrics['RMSE']:.2f}")
        print(f"    MAE:  {metrics['MAE']:.2f}")
        print(f"    R²:   {metrics['R2']:.4f}")
        if pub:
            print(f"\n  Published ({pub.get('method', '?')}, {pub.get('source', '?')}):")
            if "RMSE" in pub:
                print(f"    RMSE: {pub['RMSE']}")
            if "MAE" in pub:
                print(f"    MAE:  {pub['MAE']}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
