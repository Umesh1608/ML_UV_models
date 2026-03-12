#!/usr/bin/env python3
"""
Train Chemprop (D-MPNN) for UV λ_max prediction.

Model: MulticomponentMPNN (Chemprop v2) — separate D-MPNN for solute + solvent,
       BondMessagePassing(d_h=300, depth=3), MeanAggregation, RegressionFFN.
       ~636K params. Reference: Yang et al. 2019, JCIM.

Modes:
  1. Primary dataset (Joung+Beard 18,755) — 5-fold stratified CV (same folds as BiGRU/RF)
  2. Cross-dataset (Deep4Chem) — single random split matching published protocol

Usage:
  python3 run_chemprop.py                   # all 5 folds sequentially
  python3 run_chemprop.py --fold 0          # single fold (stop/resume safe)
  python3 run_chemprop.py --summary         # aggregate 5-fold results
  python3 run_chemprop.py --dataset deep4chem  # cross-dataset benchmark
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import torch

torch.set_float32_matmul_precision("medium")

from chemprop import data, featurizers, nn
from chemprop.models import multi
from chemprop.nn.metrics import RMSE as ChempropRMSE, MAE as ChempropMAE
from chemprop.nn.transforms import UnscaleTransform
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from paper1_new_cl.models import compute_metrics

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_PATH = os.path.join(SCRIPT_DIR, "previous_code", "UV_canonical_full_dataset.csv")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

SEED = 7
N_FOLDS = 5
MAX_EPOCHS = 100
PATIENCE = 15
WARMUP_EPOCHS = 2
INIT_LR = 1e-4
MAX_LR = 1e-3
FINAL_LR = 1e-4
BATCH_SIZE = 64
D_H = 300
DEPTH = 3

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Progress callback ───────────────────────────────────────────────────────

class ProgressCallback(pl.Callback):
    """Print training progress every N epochs with ETA."""

    def __init__(self, name="chemprop", report_every=5):
        super().__init__()
        self.name = name
        self.report_every = report_every
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        self.t0 = None
        self.current_epoch_t0 = None

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch_t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.current_epoch_t0 if self.current_epoch_t0 else 0
        self.epoch_times.append(epoch_time)

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Extract metrics from callback_metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")

        if train_loss is not None:
            self.train_losses.append(float(train_loss))
        if val_loss is not None:
            self.val_losses.append(float(val_loss))

        if (epoch + 1) % self.report_every == 0 or epoch == 0:
            elapsed = time.time() - self.t0 if self.t0 else 0
            avg_time = np.mean(self.epoch_times[-self.report_every:]) if self.epoch_times else 0
            remaining = MAX_EPOCHS - (epoch + 1)
            eta_min = (avg_time * remaining) / 60

            # Get early stopping info
            patience_counter = 0
            best_val = float("inf")
            for cb in trainer.callbacks:
                if isinstance(cb, EarlyStopping):
                    patience_counter = cb.wait_count
                    best_val = float(cb.best_score) if cb.best_score is not None else float("inf")
                    break

            tl = f"{float(train_loss):.2f}" if train_loss is not None else "N/A"
            vl = f"{float(val_loss):.2f}" if val_loss is not None else "N/A"
            bv = f"{best_val:.2f}" if best_val < float("inf") else "N/A"

            print(
                f"  [{self.name}] Epoch {epoch+1}/{MAX_EPOCHS} — "
                f"train_loss={tl}, val_loss={vl}, "
                f"best={bv}, patience={patience_counter}/{PATIENCE}, "
                f"{avg_time:.1f}s/ep, ETA: {eta_min:.0f}min",
                flush=True,
            )

    def get_history(self):
        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }


# ─── GPU utilities ────────────────────────────────────────────────────────────

def gpu_cooldown(seconds=10):
    """Empty CUDA cache, log GPU temp, and pause between folds."""
    torch.cuda.empty_cache()
    gc.collect()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        temp = result.stdout.strip()
        print(f"  >> GPU temp: {temp}°C — cooling down {seconds}s...")
    except Exception:
        print(f"  >> Cooling down {seconds}s...")
    time.sleep(seconds)


# ─── Data pipeline ────────────────────────────────────────────────────────────

def load_primary_data():
    """Load Joung+Beard primary dataset, pre-validating SMILES."""
    from rdkit import Chem

    print("\n[DATA] Loading primary dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "solvents", "lambda_max"]].dropna()

    # Pre-validate SMILES (Chemprop raises RuntimeError on invalid SMILES)
    valid_mask = np.array([
        Chem.MolFromSmiles(s) is not None and Chem.MolFromSmiles(v) is not None
        for s, v in zip(df["canon"], df["solvents"])
    ])
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  Skipping {n_invalid} entries with invalid SMILES")
        df = df[valid_mask].reset_index(drop=True)

    solutes = df["canon"].values
    solvents = df["solvents"].values
    y = df["lambda_max"].values.astype(np.float64)

    # Build index mapping: original dataset index → filtered index
    # The fold indices reference the original 18755-row dataset, so we need
    # to know which original indices are valid
    orig_indices = np.where(valid_mask)[0]

    print(f"  Samples: {len(df)} (of {len(valid_mask)} total)")
    print(f"  λ_max: {y.min():.1f} – {y.max():.1f} nm (mean {y.mean():.1f})")
    return solutes, solvents, y, orig_indices


def load_fold_indices():
    """Load precomputed stratified fold indices (same as BiGRU/RF)."""
    path = os.path.join(RESULTS_DIR, "cv_fold_indices_stratified.npz")
    if not os.path.exists(path):
        print(f"  ERROR: Fold indices not found at {path}")
        print("  Run: python3 run_baselines.py --model rf_v2 --fold 0  (to generate)")
        sys.exit(1)

    splits = np.load(path)
    folds = []
    for i in range(N_FOLDS):
        train_idx = splits[f"train_{i}"]
        val_idx = splits[f"val_{i}"]
        test_idx = splits[f"test_{i}"]
        folds.append((train_idx, val_idx, test_idx))

    print(f"  Loaded {len(folds)} folds from {path}")
    return folds


def make_chemprop_data(solutes, solvents, y, indices, orig_to_filtered=None):
    """Create Chemprop MoleculeDatapoint lists for given indices.

    Args:
        solutes, solvents, y: Arrays indexed by filtered dataset.
        indices: Original dataset indices (from fold files).
        orig_to_filtered: Mapping from original index → filtered index, or None if 1:1.

    Returns (solute_datapoints, solvent_datapoints, valid_original_indices).
    """
    solute_dps = []
    solvent_dps = []
    valid_orig = []

    for i in indices:
        if orig_to_filtered is not None:
            if i not in orig_to_filtered:
                continue  # Invalid SMILES, skip
            fi = orig_to_filtered[i]
        else:
            fi = i

        solute_dps.append(
            data.MoleculeDatapoint.from_smi(solutes[fi], y=np.array([y[fi]]))
        )
        solvent_dps.append(
            data.MoleculeDatapoint.from_smi(solvents[fi])
        )
        valid_orig.append(i)

    return solute_dps, solvent_dps, np.array(valid_orig)


def build_loaders(solute_train, solvent_train, solute_val, solvent_val,
                  solute_test, solvent_test, solute_only=False):
    """Build Chemprop dataloaders and return (train_loader, val_loader, test_loader, scaler)."""
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    if solute_only:
        train_dset = data.MoleculeDataset(solute_train, feat)
        val_dset = data.MoleculeDataset(solute_val, feat)
        test_dset = data.MoleculeDataset(solute_test, feat)

        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

        train_loader = data.build_dataloader(
            train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )
        val_loader = data.build_dataloader(
            val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        test_loader = data.build_dataloader(
            test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
    else:
        train_dsets = [
            data.MoleculeDataset(solute_train, feat),
            data.MoleculeDataset(solvent_train, feat),
        ]
        val_dsets = [
            data.MoleculeDataset(solute_val, feat),
            data.MoleculeDataset(solvent_val, feat),
        ]
        test_dsets = [
            data.MoleculeDataset(solute_test, feat),
            data.MoleculeDataset(solvent_test, feat),
        ]

        train_mcdset = data.MulticomponentDataset(train_dsets)
        val_mcdset = data.MulticomponentDataset(val_dsets)
        test_mcdset = data.MulticomponentDataset(test_dsets)

        scaler = train_mcdset.normalize_targets()
        val_mcdset.normalize_targets(scaler)

        train_loader = data.build_dataloader(
            train_mcdset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )
        val_loader = data.build_dataloader(
            val_mcdset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        test_loader = data.build_dataloader(
            test_mcdset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )

    return train_loader, val_loader, test_loader, scaler


def build_model(scaler, solute_only=False):
    """Build Chemprop MPNN model (single or multicomponent)."""
    from chemprop.models import MPNN as SingleMPNN

    agg = nn.MeanAggregation()
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    if solute_only:
        mp = nn.BondMessagePassing(d_h=D_H, depth=DEPTH)
        ffn = nn.RegressionFFN(
            input_dim=mp.output_dim,
            output_transform=output_transform,
        )
        model = SingleMPNN(
            mp, agg, ffn,
            metrics=[ChempropRMSE(), ChempropMAE()],
            warmup_epochs=WARMUP_EPOCHS,
            init_lr=INIT_LR,
            max_lr=MAX_LR,
            final_lr=FINAL_LR,
        )
        n_components = 1
    else:
        mcmp = nn.MulticomponentMessagePassing(
            blocks=[nn.BondMessagePassing(d_h=D_H, depth=DEPTH) for _ in range(2)],
            n_components=2,
        )
        ffn = nn.RegressionFFN(
            input_dim=mcmp.output_dim,
            output_transform=output_transform,
        )
        model = multi.MulticomponentMPNN(
            mcmp, agg, ffn,
            metrics=[ChempropRMSE(), ChempropMAE()],
            warmup_epochs=WARMUP_EPOCHS,
            init_lr=INIT_LR,
            max_lr=MAX_LR,
            final_lr=FINAL_LR,
        )
        n_components = 2

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  >> Chemprop D-MPNN: {n_params:,} params "
          f"(d_h={D_H}, depth={DEPTH}, {n_components} component{'s' if n_components > 1 else ''})")
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_fold(fold_idx, solutes, solvents, y, folds, orig_to_filtered, solute_only=False):
    """Train a single fold. Returns metrics dict or None if interrupted/skipped."""
    prefix = "chemprop_v2_nosolvent" if solute_only else "chemprop_v2"
    name = f"{prefix}_fold{fold_idx}"

    # Check if already done
    metrics_path = os.path.join(RESULTS_DIR, f"{name}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"  [SKIP] Fold {fold_idx} already complete (RMSE={metrics['RMSE']:.2f})")
        return metrics

    train_idx, val_idx, test_idx = folds[fold_idx]

    print(f"\n{'='*70}")
    print(f"  FOLD {fold_idx}/{N_FOLDS-1} — "
          f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"{'='*70}")

    # Create datapoints (skipping any invalid SMILES)
    solute_train, solvent_train, train_valid = make_chemprop_data(
        solutes, solvents, y, train_idx, orig_to_filtered)
    solute_val, solvent_val, val_valid = make_chemprop_data(
        solutes, solvents, y, val_idx, orig_to_filtered)
    solute_test, solvent_test, test_valid = make_chemprop_data(
        solutes, solvents, y, test_idx, orig_to_filtered)

    n_skipped = (len(train_idx) - len(train_valid) +
                 len(val_idx) - len(val_valid) +
                 len(test_idx) - len(test_valid))
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} invalid SMILES entries")

    # Build loaders
    train_loader, val_loader, test_loader, scaler = build_loaders(
        solute_train, solvent_train,
        solute_val, solvent_val,
        solute_test, solvent_test,
        solute_only=solute_only,
    )

    # Build model
    model = build_model(scaler, solute_only=solute_only)

    # Callbacks
    best_ckpt_path = os.path.join(RESULTS_DIR, f"{name}_best")
    latest_ckpt_path = os.path.join(RESULTS_DIR, f"{name}_latest")

    checkpoint_best = ModelCheckpoint(
        dirpath=RESULTS_DIR,
        filename=f"{name}_best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    checkpoint_periodic = ModelCheckpoint(
        dirpath=RESULTS_DIR,
        filename=f"{name}_latest",
        every_n_epochs=5,
        save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        min_delta=1e-5,
    )
    progress = ProgressCallback(name=name, report_every=5)

    # Check for existing checkpoint to resume from
    ckpt_resume = None
    for suffix in ["_latest.ckpt", "_best.ckpt"]:
        candidate = os.path.join(RESULTS_DIR, f"{name}{suffix}")
        if os.path.exists(candidate):
            ckpt_resume = candidate
            print(f"  >> Resuming from {os.path.basename(candidate)}")
            break

    # Train
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_best, checkpoint_periodic, early_stop, progress],
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=(fold_idx == 0),
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_resume)
    elapsed = time.time() - t0
    print(f"  [{name}] Training complete in {elapsed:.0f}s "
          f"({trainer.current_epoch + 1} epochs)")

    # Save training history
    history = progress.get_history()
    hist_path = os.path.join(RESULTS_DIR, f"{name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    # Load best model and predict on test set
    best_ckpt = checkpoint_best.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"  >> Loading best model from {os.path.basename(best_ckpt)} "
              f"(val_loss={checkpoint_best.best_model_score:.4f})")
    else:
        # Fallback: use current model
        print("  >> No best checkpoint found, using final model")
        best_ckpt = None

    # Predict
    if best_ckpt:
        if solute_only:
            from chemprop.models import MPNN as SingleMPNN
            best_model = SingleMPNN.load_from_checkpoint(best_ckpt)
        else:
            best_model = multi.MulticomponentMPNN.load_from_checkpoint(best_ckpt)
    else:
        best_model = model

    preds = trainer.predict(best_model, test_loader)
    y_pred = torch.cat(preds, dim=0).squeeze(-1).numpy()
    # Get y_test from filtered indices
    y_test = np.array([y[orig_to_filtered[i]] for i in test_valid])

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    print(f"  >> {name} — RMSE={metrics['RMSE']:.2f} MAE={metrics['MAE']:.2f} "
          f"R²={metrics['R2']:.4f} r={metrics['Pearson_r']:.4f}")

    # Save results
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, f"{name}_predictions.npy"), y_pred)
    np.save(os.path.join(RESULTS_DIR, f"{name}_y_test.npy"), y_test)
    np.save(os.path.join(RESULTS_DIR, f"{name}_test_indices.npy"), test_valid)

    # Cleanup periodic checkpoint (keep best)
    latest_ckpt = os.path.join(RESULTS_DIR, f"{name}_latest.ckpt")
    if os.path.exists(latest_ckpt):
        os.remove(latest_ckpt)

    # Free GPU memory
    del model, best_model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


# ─── Primary CV ───────────────────────────────────────────────────────────────

def run_primary_cv(fold_id=None, solute_only=False):
    """Run 5-fold CV on primary dataset."""
    solutes, solvents, y, orig_indices = load_primary_data()
    folds = load_fold_indices()

    # Build mapping: original dataset index → filtered dataset index
    orig_to_filtered = {orig: filt for filt, orig in enumerate(orig_indices)}

    pl.seed_everything(SEED, workers=True)

    fold_range = [fold_id] if fold_id is not None else list(range(N_FOLDS))
    all_metrics = []

    for fold in fold_range:
        metrics = train_fold(fold, solutes, solvents, y, folds, orig_to_filtered,
                            solute_only=solute_only)
        if metrics is None:
            return
        all_metrics.append(metrics)

        # Cool down GPU between folds
        if fold != fold_range[-1] and len(fold_range) > 1:
            gpu_cooldown()

    if len(all_metrics) == N_FOLDS:
        prefix = "chemprop_v2_nosolvent" if solute_only else "chemprop_v2"
        label = "solute only" if solute_only else "solute + solvent"
        print_aggregate(all_metrics, prefix=prefix, label=label)


def print_aggregate(all_metrics, prefix="chemprop_v2", label="solute + solvent"):
    """Print and save aggregate metrics across folds."""
    print(f"\n{'='*70}")
    print(f"  AGGREGATE: Chemprop D-MPNN (v2, {label}, 5-fold CV)")
    print(f"{'='*70}")

    agg = {"n_folds": len(all_metrics)}
    for key in ["RMSE", "MAE", "R2", "Pearson_r"]:
        vals = [m[key] for m in all_metrics]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        agg[f"{key}_mean"] = round(mean, 4)
        agg[f"{key}_std"] = round(std, 4)
        agg[f"{key}_values"] = vals
        print(f"  {key}: {mean:.2f} ± {std:.2f}")

    agg_path = os.path.join(RESULTS_DIR, f"{prefix}_cv_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  Saved to {agg_path}")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary():
    """Print summary of all Chemprop results."""
    print(f"\n{'='*70}")
    print("  CHEMPROP D-MPNN RESULTS SUMMARY")
    print(f"{'='*70}")

    # Check aggregate
    agg_path = os.path.join(RESULTS_DIR, "chemprop_v2_cv_aggregate.json")
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            agg = json.load(f)
        print(f"\n  Primary (v2, {agg.get('n_folds', '?')}-fold CV):")
        for key in ["RMSE", "MAE", "R2", "Pearson_r"]:
            print(f"    {key}: {agg[f'{key}_mean']:.2f} ± {agg[f'{key}_std']:.2f}")
    else:
        # Try per-fold
        fold_metrics = []
        for fold in range(N_FOLDS):
            path = os.path.join(RESULTS_DIR, f"chemprop_v2_fold{fold}_metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    fold_metrics.append(json.load(f))
        if fold_metrics:
            print(f"\n  Primary (v2, {len(fold_metrics)}/{N_FOLDS} folds):")
            for key in ["RMSE", "MAE", "R2", "Pearson_r"]:
                vals = [m[key] for m in fold_metrics]
                print(f"    {key}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")
            if len(fold_metrics) == N_FOLDS:
                print_aggregate(fold_metrics)
        else:
            print("\n  Primary: No results found")

    # Cross-dataset
    for ds_key, ds_info in CROSS_DATASET_INFO.items():
        path = os.path.join(RESULTS_DIR, ds_key, "chemprop_solvent_random_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            print(f"\n  {ds_info['display']}:")
            print(f"    RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, "
                  f"R²={m['R2']:.4f}, r={m['Pearson_r']:.4f}")


# ─── Cross-dataset ────────────────────────────────────────────────────────────

CROSS_DATASET_INFO = {
    "deep4chem": {
        "file": os.path.join(DATA_DIR, "deep4chem_processed.csv"),
        "display": "Deep4Chem (Joung 2020)",
        "random_split": {"train": 0.8, "val": 0.1, "test": 0.1},
    },
}


def run_cross_dataset(dataset_key):
    """Train Chemprop on a cross-dataset benchmark."""
    info = CROSS_DATASET_INFO[dataset_key]
    path = info["file"]

    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        print(f"  Run: python3 download_datasets.py --dataset {dataset_key}")
        sys.exit(1)

    print(f"\n[DATA] Loading {info['display']} from {path}...")
    df = pd.read_csv(path)
    df = df.dropna(subset=["smiles", "solvent_smiles", "lambda_max"])

    solutes = df["smiles"].values
    solvents_arr = df["solvent_smiles"].values
    y = df["lambda_max"].values.astype(np.float64)

    print(f"  Samples: {len(df)}, λ_max: {y.min():.1f} – {y.max():.1f} nm")

    results_dir = os.path.join(RESULTS_DIR, dataset_key)
    os.makedirs(results_dir, exist_ok=True)

    prefix = "chemprop_solvent_random"

    # Check if already done
    metrics_path = os.path.join(results_dir, f"{prefix}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"  [SKIP] Already complete (RMSE={metrics['RMSE']:.2f})")
        return metrics

    # Load or create split indices
    split_path = os.path.join(results_dir, "random_split_indices.npz")
    if os.path.exists(split_path):
        print(f"  Loading existing split indices from {split_path}")
        splits = np.load(split_path)
        train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    else:
        from sklearn.model_selection import train_test_split
        n = len(y)
        indices = np.arange(n)
        split_cfg = info["random_split"]
        test_frac = split_cfg["test"]
        val_frac = split_cfg["val"]

        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_frac, random_state=SEED,
        )
        val_frac_of_trainval = val_frac / (1.0 - test_frac)
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_frac_of_trainval, random_state=SEED,
        )
        np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)

    print(f"\n{'─'*70}")
    print(f"  {info['display']} — Chemprop random split "
          f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
    print(f"{'─'*70}")

    pl.seed_everything(SEED, workers=True)

    # Create datapoints
    solute_train, solvent_train, _ = make_chemprop_data(solutes, solvents_arr, y, train_idx)
    solute_val, solvent_val, _ = make_chemprop_data(solutes, solvents_arr, y, val_idx)
    solute_test, solvent_test, test_orig_idx = make_chemprop_data(solutes, solvents_arr, y, test_idx)

    # Build loaders
    train_loader, val_loader, test_loader, scaler = build_loaders(
        solute_train, solvent_train,
        solute_val, solvent_val,
        solute_test, solvent_test,
    )

    # Build model
    model = build_model(scaler)

    # Callbacks
    checkpoint_best = ModelCheckpoint(
        dirpath=results_dir,
        filename=f"{prefix}_best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        min_delta=1e-5,
    )
    progress = ProgressCallback(name=prefix, report_every=5)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_best, early_stop, progress],
        enable_progress_bar=False,
        logger=False,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    elapsed = time.time() - t0
    print(f"  [{prefix}] Training complete in {elapsed:.0f}s")

    # Save history
    history = progress.get_history()
    with open(os.path.join(results_dir, f"{prefix}_history.json"), "w") as f:
        json.dump(history, f)

    # Load best model and predict
    best_ckpt = checkpoint_best.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        best_model = multi.MulticomponentMPNN.load_from_checkpoint(best_ckpt)
    else:
        best_model = model

    preds = trainer.predict(best_model, test_loader)
    y_pred = torch.cat(preds, dim=0).squeeze(-1).numpy()
    y_test = y[test_idx]

    metrics = compute_metrics(y_test, y_pred)
    print(f"  >> {prefix} — RMSE={metrics['RMSE']:.2f} MAE={metrics['MAE']:.2f} "
          f"R²={metrics['R2']:.4f} r={metrics['Pearson_r']:.4f}")

    # Save results
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, f"{prefix}_predictions.npy"), y_pred)
    np.save(os.path.join(results_dir, f"{prefix}_y_test.npy"), y_test)
    np.save(os.path.join(results_dir, f"{prefix}_test_indices.npy"), test_idx)

    del model, best_model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chemprop (D-MPNN) for UV λ_max prediction",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run single fold (0-4) for primary dataset",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary of all results",
    )
    parser.add_argument(
        "--dataset", choices=["primary", "deep4chem"], default="primary",
        help="Dataset to train on (default: primary)",
    )
    parser.add_argument(
        "--no-solvent", action="store_true",
        help="Train solute-only model (single-component MPNN, no solvent graph)",
    )
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.dataset == "primary":
        run_primary_cv(fold_id=args.fold, solute_only=args.no_solvent)
    else:
        run_cross_dataset(args.dataset)


if __name__ == "__main__":
    main()
