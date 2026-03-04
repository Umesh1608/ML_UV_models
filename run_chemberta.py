#!/usr/bin/env python3
"""
Fine-tune ChemBERTa (pretrained Transformer) for UV λ_max prediction.

Model: seyonec/ChemBERTa-zinc-base-v1 (RoBERTa, 44.1M params, pretrained on ZINC)
Training: AdamW, lr=2e-5, MAE loss, early stopping on val MAE, fp16

Modes:
  1. Primary dataset (Joung+Beard 18,755) — 5-fold stratified CV (same folds as BiGRU/RF)
  2. Cross-dataset (Deep4Chem, Jung 2024) — single random splits matching published protocols

Usage:
  python3 run_chemberta.py --dataset primary --v2           # 5-fold CV on primary dataset
  python3 run_chemberta.py --dataset primary --v2 --fold 0  # single fold
  python3 run_chemberta.py --dataset deep4chem              # cross-dataset
  python3 run_chemberta.py --dataset jung2024               # cross-dataset
  python3 run_chemberta.py --summary                        # aggregate results
"""

import argparse
import gc
import json
import os
import sys
import time

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paper1_new_cl.models import compute_metrics

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_PATH = os.path.join(SCRIPT_DIR, "previous_code", "UV_canonical_full_dataset.csv")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
MAX_LEN = 512  # ChemBERTa max position embeddings
SEED = 7
N_FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 32
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
PATIENCE = 10

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Dataset class ────────────────────────────────────────────────────────────

class SMILESDataset(Dataset):
    """PyTorch Dataset for SMILES regression."""

    def __init__(self, smiles_list, targets, tokenizer, max_len=MAX_LEN):
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


# ─── Training loop ────────────────────────────────────────────────────────────

def create_model(device):
    """Load ChemBERTa and configure for regression."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="regression"
    )
    model.to(device)
    return model


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then linear decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler):
    """Train for one epoch, return mean MAE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.squeeze(-1)
            loss = nn.SmoothL1Loss(beta=10.0)(preds, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model, return (predictions, mae_loss)."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.squeeze(-1).cpu()

        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    mae = np.mean(np.abs(all_preds - all_targets))
    return all_preds, mae


def train_chemberta(train_smiles, val_smiles, test_smiles,
                    y_train, y_val, y_test,
                    name="chemberta", results_dir=RESULTS_DIR):
    """Full training loop with early stopping. Returns (y_pred, metrics)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  >> {name} — device: {device}")

    if torch.cuda.is_available():
        print(f"  >> GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = SMILESDataset(train_smiles, y_train, tokenizer)
    val_ds = SMILESDataset(val_smiles, y_val, tokenizer)
    test_ds = SMILESDataset(test_smiles, y_test, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = create_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  >> Parameters: {n_params:,} total, {n_trainable:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_mae = float("inf")
    best_epoch = -1
    patience_counter = 0
    checkpoint_path = os.path.join(results_dir, f"{name}_best.pt")
    history = {"train_mae": [], "val_mae": []}

    t0 = time.time()
    for epoch in range(EPOCHS):
        epoch_t0 = time.time()
        train_mae = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        _, val_mae = evaluate(model, val_loader, device)

        history["train_mae"].append(float(train_mae))
        history["val_mae"].append(float(val_mae))

        epoch_time = time.time() - epoch_t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{name}] Epoch {epoch+1}/{EPOCHS} — "
                  f"train_mae={train_mae:.2f}, val_mae={val_mae:.2f}, "
                  f"lr={scheduler.get_last_lr()[0]:.2e}, {epoch_time:.1f}s",
                  flush=True)

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [{name}] Early stopping at epoch {epoch+1} "
                      f"(best val_mae={best_val_mae:.2f} at epoch {best_epoch})",
                      flush=True)
                break

    elapsed = time.time() - t0
    print(f"  [{name}] Training complete — {epoch+1} epochs in {elapsed:.0f}s",
          flush=True)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    y_pred, test_mae = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_test, y_pred)

    print(f"  >> {name} — RMSE={metrics['RMSE']:.2f} MAE={metrics['MAE']:.2f} "
          f"R²={metrics['R2']:.4f} r={metrics['Pearson_r']:.4f}")

    # Save history
    hist_path = os.path.join(results_dir, f"{name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    # Cleanup
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return y_pred, metrics


# ─── Primary dataset (5-fold CV) ─────────────────────────────────────────────

def load_primary_data():
    """Load Joung+Beard primary dataset."""
    print("\n[DATA] Loading primary dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "solvents", "lambda_max"]].dropna()

    # Combined SMILES with "." separator (standard SMILES fragment separator)
    combined = (df["canon"] + "." + df["solvents"]).values
    y = df["lambda_max"].values.astype(np.float64)

    print(f"  Samples: {len(df)}")
    print(f"  λ_max: {y.min():.1f} – {y.max():.1f} nm (mean {y.mean():.1f})")

    return combined, y, df["solvents"].values


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


def run_primary_cv(fold_id=None):
    """Run 5-fold CV on primary dataset."""
    combined, y, solvents = load_primary_data()
    folds = load_fold_indices()

    fold_range = [fold_id] if fold_id is not None else range(N_FOLDS)
    all_metrics = []

    for fold in fold_range:
        train_idx, val_idx, test_idx = folds[fold]

        print(f"\n{'='*70}")
        print(f"  FOLD {fold}/{N_FOLDS-1} — "
              f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        print(f"{'='*70}")

        name = f"chemberta_fold{fold}"

        # Check if already done
        metrics_path = os.path.join(RESULTS_DIR, f"{name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"  [SKIP] Fold {fold} already complete (RMSE={metrics['RMSE']:.2f})")
            all_metrics.append(metrics)
            continue

        y_pred, metrics = train_chemberta(
            train_smiles=combined[train_idx].tolist(),
            val_smiles=combined[val_idx].tolist(),
            test_smiles=combined[test_idx].tolist(),
            y_train=y[train_idx],
            y_val=y[val_idx],
            y_test=y[test_idx],
            name=name,
        )

        # Save outputs
        np.save(os.path.join(RESULTS_DIR, f"{name}_predictions.npy"), y_pred)
        np.save(os.path.join(RESULTS_DIR, f"{name}_y_test.npy"), y[test_idx])
        np.save(os.path.join(RESULTS_DIR, f"{name}_test_indices.npy"), test_idx)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        all_metrics.append(metrics)

    if len(all_metrics) == N_FOLDS:
        print_aggregate(all_metrics, "ChemBERTa (primary v2)")


def print_aggregate(all_metrics, label):
    """Print aggregate metrics across folds."""
    print(f"\n{'='*70}")
    print(f"  AGGREGATE: {label}")
    print(f"{'='*70}")

    for key in ["RMSE", "MAE", "R2", "Pearson_r"]:
        vals = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")

    # Save aggregate
    agg = {}
    for key in ["RMSE", "MAE", "R2", "Pearson_r"]:
        vals = [m[key] for m in all_metrics]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 2)
        agg[f"{key}_std"] = round(float(np.std(vals)), 2)
    agg["n_folds"] = len(all_metrics)

    agg_path = os.path.join(RESULTS_DIR, "chemberta_cv_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"  Saved to {agg_path}")


# ─── Cross-dataset benchmarks ────────────────────────────────────────────────

CROSS_DATASET_INFO = {
    "deep4chem": {
        "file": os.path.join(DATA_DIR, "deep4chem_processed.csv"),
        "display": "Deep4Chem (Joung 2020)",
        "random_split": {"train": 0.8, "val": 0.1, "test": 0.1},
    },
    "jung2024": {
        "file": os.path.join(DATA_DIR, "jung2024_processed.csv"),
        "display": "Jung et al. 2024",
        "random_split": {"train": 0.72, "val": 0.18, "test": 0.10},
    },
}


def load_cross_dataset(dataset_key):
    """Load cross-dataset CSV."""
    info = CROSS_DATASET_INFO[dataset_key]
    path = info["file"]

    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        print(f"  Run: python3 download_datasets.py --dataset {dataset_key}")
        sys.exit(1)

    print(f"\n[DATA] Loading {info['display']} from {path}...")
    df = pd.read_csv(path)
    df = df.dropna(subset=["smiles", "solvent_smiles", "lambda_max"])

    combined = (df["smiles"] + "." + df["solvent_smiles"]).values
    y = df["lambda_max"].values.astype(np.float64)

    print(f"  Samples: {len(df)}, λ_max: {y.min():.1f} – {y.max():.1f} nm")
    return combined, y


def run_cross_dataset(dataset_key):
    """Train ChemBERTa on a cross-dataset benchmark."""
    info = CROSS_DATASET_INFO[dataset_key]
    combined, y = load_cross_dataset(dataset_key)

    results_dir = os.path.join(RESULTS_DIR, dataset_key)
    os.makedirs(results_dir, exist_ok=True)

    prefix = "chemberta_solvent_random"

    # Check if already done
    metrics_path = os.path.join(results_dir, f"{prefix}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"  [SKIP] Already complete (RMSE={metrics['RMSE']:.2f})")
        return metrics

    # Load existing split indices (reuse from BiGRU/RF run)
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
            indices, test_size=test_frac, random_state=SEED
        )
        val_frac_of_trainval = val_frac / (1.0 - test_frac)
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_frac_of_trainval, random_state=SEED
        )
        np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)

    print(f"\n{'─'*70}")
    print(f"  {info['display']} — ChemBERTa random split "
          f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
    print(f"{'─'*70}")

    y_pred, metrics = train_chemberta(
        train_smiles=combined[train_idx].tolist(),
        val_smiles=combined[val_idx].tolist(),
        test_smiles=combined[test_idx].tolist(),
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        name=prefix,
        results_dir=results_dir,
    )

    # Save outputs
    np.save(os.path.join(results_dir, f"{prefix}_predictions.npy"), y_pred)
    np.save(os.path.join(results_dir, f"{prefix}_y_test.npy"), y[test_idx])
    np.save(os.path.join(results_dir, f"{prefix}_test_indices.npy"), test_idx)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary():
    """Print summary of all ChemBERTa results."""
    print(f"\n{'='*70}")
    print("  CHEMBERTA RESULTS SUMMARY")
    print(f"{'='*70}")

    # Primary CV
    agg_path = os.path.join(RESULTS_DIR, "chemberta_cv_aggregate.json")
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            agg = json.load(f)
        print(f"\n  Primary (v2, 5-fold CV):")
        print(f"    RMSE: {agg['RMSE_mean']:.2f} ± {agg['RMSE_std']:.2f}")
        print(f"    MAE:  {agg['MAE_mean']:.2f} ± {agg['MAE_std']:.2f}")
        print(f"    R²:   {agg['R2_mean']:.4f} ± {agg['R2_std']:.4f}")
        print(f"    r:    {agg['Pearson_r_mean']:.4f} ± {agg['Pearson_r_std']:.4f}")
    else:
        # Try per-fold metrics
        fold_metrics = []
        for fold in range(N_FOLDS):
            path = os.path.join(RESULTS_DIR, f"chemberta_fold{fold}_metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    fold_metrics.append(json.load(f))
        if fold_metrics:
            print(f"\n  Primary (v2, {len(fold_metrics)}/{N_FOLDS} folds):")
            for key in ["RMSE", "MAE", "R2", "Pearson_r"]:
                vals = [m[key] for m in fold_metrics]
                print(f"    {key}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")
            if len(fold_metrics) == N_FOLDS:
                print_aggregate(fold_metrics, "ChemBERTa (primary v2)")
        else:
            print("\n  Primary: No results found")

    # Cross-dataset
    for ds in ["deep4chem", "jung2024"]:
        path = os.path.join(RESULTS_DIR, ds, "chemberta_solvent_random_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            display = CROSS_DATASET_INFO[ds]["display"]
            print(f"\n  {display}:")
            print(f"    RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, "
                  f"R²={m['R2']:.4f}, r={m['Pearson_r']:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChemBERTa fine-tuning for UV λ_max")
    parser.add_argument("--dataset", choices=["primary", "deep4chem", "jung2024"],
                        default="primary", help="Dataset to train on")
    parser.add_argument("--v2", action="store_true",
                        help="Use v2 stratified splits (required for primary)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run single fold (0-4) for primary dataset")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of all results")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.summary:
        print_summary()
        return

    if args.dataset == "primary":
        run_primary_cv(fold_id=args.fold)
    else:
        run_cross_dataset(args.dataset)


if __name__ == "__main__":
    main()
