#!/usr/bin/env python3
"""
Transfer learning: pretrain BiGRU+Solvent on Reaxys, fine-tune on Joung+Beard.

Workflow:
  1. Build superset charset from Reaxys + Joung+Beard
  2. Pretrain on mamede_regression_dataset.csv (90/10 train/val)
  3. Fine-tune on Joung+Beard 5-fold CV (reuse stratified splits)

Usage:
  python3 run_transfer.py --pretrain                    # Pretrain on Reaxys (~2-4 hours)
  python3 run_transfer.py --finetune --fold 0           # Fine-tune fold 0 (~3-5 hours)
  python3 run_transfer.py --finetune                    # All 5 folds
  python3 run_transfer.py --finetune --strategy freeze  # Freeze embedding + first BiGRU
  python3 run_transfer.py --summary                     # Aggregate results
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

from paper1_new_cl.models import (
    build_char_maps,
    compute_metrics,
    create_dl_model,
    get_tqdm_callback,
    vectorize_smiles,
    verify_gpu_available,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

REAXYS_PATH = os.path.join(DATA_DIR, "mamede_regression_dataset.csv")
JB_PATH = os.path.join(SCRIPT_DIR, "previous_code", "UV_canonical_full_dataset.csv")

SEED = 7
PRETRAIN_EPOCHS = 250
FINETUNE_EPOCHS = 250
BATCH_SIZE = 32
PRETRAIN_PATIENCE = 25
FINETUNE_PATIENCE = 25
N_FOLDS = 5


# ─── Data loading ──────────────────────────────────────────────────────────

def load_reaxys():
    """Load Reaxys regression dataset."""
    if not os.path.exists(REAXYS_PATH):
        print(f"ERROR: {REAXYS_PATH} not found.")
        print("  Run: python3 postprocess_reaxys.py --solvent all")
        sys.exit(1)

    df = pd.read_csv(REAXYS_PATH)
    df = df.dropna(subset=["smiles", "solvent_smiles", "lambda_max"])

    # Filter reasonable wavelength range
    df = df[(df["lambda_max"] >= 100) & (df["lambda_max"] <= 1000)]

    combined = (df["smiles"] + "!" + df["solvent_smiles"]).values
    y = df["lambda_max"].values.astype(np.float64)

    print(f"[Reaxys] Loaded {len(df)} samples")
    print(f"  λ_max range: {y.min():.1f} – {y.max():.1f} nm (mean {y.mean():.1f})")
    return combined, y


def load_joung_beard():
    """Load Joung+Beard dataset."""
    if not os.path.exists(JB_PATH):
        print(f"ERROR: {JB_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(JB_PATH)
    combined = (df["canon"] + "!" + df["solvents"]).values
    y = df["lambda_max"].values.astype(np.float64)
    solvents = df["solvents"].values

    print(f"[Joung+Beard] Loaded {len(df)} samples")
    print(f"  λ_max range: {y.min():.1f} – {y.max():.1f} nm (mean {y.mean():.1f})")
    return combined, y, solvents


def build_superset_charset(reaxys_combined, jb_combined):
    """Build charset from union of both datasets."""
    all_chars = set()
    for s in reaxys_combined:
        all_chars.update(s)
    for s in jb_combined:
        all_chars.update(s)
    all_chars.add("!")
    all_chars.add("E")
    charset = sorted(all_chars)

    max_len_reaxys = max(len(s) for s in reaxys_combined)
    max_len_jb = max(len(s) for s in jb_combined)
    embed_len = max(max_len_reaxys, max_len_jb) + 5

    print(f"[Charset] Superset: {len(charset)} tokens, embed_len: {embed_len}")
    return charset, embed_len


# ─── Pretraining ──────────────────────────────────────────────────────────

def pretrain(charset, embed_len, reaxys_combined, reaxys_y):
    """Pretrain BiGRU+Solvent on Reaxys data with 90/10 train/val split."""
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[GPU] Mixed precision enabled")

    verify_gpu_available()

    c2i = build_char_maps(charset)
    num_words = len(charset)
    input_length = embed_len - 1

    # 90/10 split for pretraining
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(reaxys_combined))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=SEED)

    print(f"\n[Pretrain] Vectorizing {len(train_idx)} train + {len(val_idx)} val...")
    X_train = vectorize_smiles(reaxys_combined[train_idx], c2i, embed_len)
    X_val = vectorize_smiles(reaxys_combined[val_idx], c2i, embed_len)
    y_train = reaxys_y[train_idx].reshape(-1, 1)
    y_val = reaxys_y[val_idx].reshape(-1, 1)

    # Build model
    model = create_dl_model("bigru", num_words, input_length)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8)
    model.compile(loss="mae", optimizer=optimizer, metrics=["mse"])

    model.build(input_shape=(None, input_length))
    print(f"\n  >> Pretrain model — {model.count_params():,} params")

    # Callbacks
    pretrain_dir = os.path.join(RESULTS_DIR, "transfer")
    os.makedirs(pretrain_dir, exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=PRETRAIN_PATIENCE,
        verbose=0, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(pretrain_dir, "pretrain_best.keras"),
        monitor="val_loss", save_best_only=True, verbose=0
    )
    TqdmCB = get_tqdm_callback()
    progress = TqdmCB(PRETRAIN_EPOCHS, model_name="Pretrain", report_interval=10)

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, progress], verbose=0
    )
    elapsed = time.time() - t0

    # Evaluate on val
    y_pred_val = model.predict(X_val, verbose=0).flatten()
    val_metrics = compute_metrics(y_val.flatten(), y_pred_val)

    print(f"\n  >> Pretrain done in {elapsed:.0f}s — "
          f"RMSE={val_metrics['RMSE']} MAE={val_metrics['MAE']} R²={val_metrics['R2']}")

    # Save pretrained model
    model_path = os.path.join(pretrain_dir, "pretrain_model.keras")
    model.save(model_path)
    print(f"  >> Saved to {model_path}")

    # Save config
    config = {
        "charset": charset,
        "embed_len": embed_len,
        "num_words": num_words,
        "input_length": input_length,
        "reaxys_samples": len(reaxys_combined),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "val_metrics": val_metrics,
        "epochs_trained": len(history.history["loss"]),
        "elapsed_seconds": round(elapsed),
    }
    with open(os.path.join(pretrain_dir, "pretrain_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save history
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(pretrain_dir, "pretrain_history.json"), "w") as f:
        json.dump(hist_data, f)

    del model, X_train, X_val
    tf.keras.backend.clear_session()
    gc.collect()


# ─── Fine-tuning ──────────────────────────────────────────────────────────

def finetune_fold(fold_idx, charset, embed_len, jb_combined, jb_y, jb_solvents,
                  strategy="all"):
    """Fine-tune pretrained model on one fold of Joung+Beard."""
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    pretrain_dir = os.path.join(RESULTS_DIR, "transfer")
    model_path = os.path.join(pretrain_dir, "pretrain_model.keras")
    if not os.path.exists(model_path):
        print(f"ERROR: Pretrained model not found at {model_path}")
        print("  Run: python3 run_transfer.py --pretrain")
        sys.exit(1)

    # Check if already done
    metrics_path = os.path.join(pretrain_dir, f"transfer_{strategy}_fold{fold_idx}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing = json.load(f)
        print(f"\n  [Fold {fold_idx}] Already completed (RMSE={existing['RMSE']}). Skipping.")
        return existing

    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    verify_gpu_available()

    # Load fold indices (reuse from run_baselines.py v2 stratified splits)
    fold_path = os.path.join(RESULTS_DIR, "cv_fold_indices_stratified.npz")
    if not os.path.exists(fold_path):
        print(f"  Fold indices not found at {fold_path}, creating...")
        from paper1_new_cl.splits import create_stratified_folds
        folds = create_stratified_folds(len(jb_y), jb_solvents, n_folds=N_FOLDS, seed=SEED)
        np.savez(fold_path,
                 **{f"fold{i}_train": f[0] for i, f in enumerate(folds)},
                 **{f"fold{i}_val": f[1] for i, f in enumerate(folds)},
                 **{f"fold{i}_test": f[2] for i, f in enumerate(folds)})

    fold_data = np.load(fold_path)
    train_idx = fold_data[f"fold{fold_idx}_train"]
    val_idx = fold_data[f"fold{fold_idx}_val"]
    test_idx = fold_data[f"fold{fold_idx}_test"]

    print(f"\n{'─'*70}")
    print(f"  Transfer ({strategy}) — Fold {fold_idx}/{N_FOLDS-1}  "
          f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
    print(f"{'─'*70}")

    c2i = build_char_maps(charset)

    X_train = vectorize_smiles(jb_combined[train_idx], c2i, embed_len)
    X_val = vectorize_smiles(jb_combined[val_idx], c2i, embed_len)
    X_test = vectorize_smiles(jb_combined[test_idx], c2i, embed_len)
    y_train = jb_y[train_idx].reshape(-1, 1)
    y_val = jb_y[val_idx].reshape(-1, 1)
    y_test = jb_y[test_idx].reshape(-1, 1)

    # Load pretrained model
    model = tf.keras.models.load_model(model_path)

    # Apply fine-tuning strategy
    if strategy == "freeze":
        # Freeze Embedding + first BiGRU layer
        for layer in model.layers[:3]:  # Embedding + first Bidirectional(GRU)
            layer.trainable = False
            print(f"  Frozen: {layer.name}")
        lr = 0.001
    else:  # strategy == "all"
        lr = 1e-4  # Lower LR for all-layer fine-tuning

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-8)
    model.compile(loss="mae", optimizer=optimizer, metrics=["mse"])

    trainable = sum(p.numpy().size for p in model.trainable_weights)
    total = model.count_params()
    print(f"  >> Transfer ({strategy}, lr={lr}) — {trainable:,}/{total:,} trainable params")

    name = f"transfer_{strategy}_fold{fold_idx}"
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=FINETUNE_PATIENCE,
        verbose=0, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(pretrain_dir, f"{name}_best.keras"),
        monitor="val_loss", save_best_only=True, verbose=0
    )
    TqdmCB = get_tqdm_callback()
    progress = TqdmCB(FINETUNE_EPOCHS, model_name=name, report_interval=10)

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=FINETUNE_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, progress], verbose=0
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = compute_metrics(y_test.flatten(), y_pred)

    print(f"\n  >> {name} done in {elapsed:.0f}s — RMSE={metrics['RMSE']} "
          f"MAE={metrics['MAE']} R²={metrics['R2']} r={metrics['Pearson_r']}")

    # Save results
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(pretrain_dir, f"{name}_predictions.npy"), y_pred)
    np.save(os.path.join(pretrain_dir, f"{name}_y_test.npy"), y_test.flatten())
    np.save(os.path.join(pretrain_dir, f"{name}_test_indices.npy"), test_idx)

    # Save history
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(pretrain_dir, f"{name}_history.json"), "w") as f:
        json.dump(hist_data, f)

    # Save model
    model.save(os.path.join(pretrain_dir, f"{name}_model.keras"))

    del model, X_train, X_val, X_test
    tf.keras.backend.clear_session()
    gc.collect()

    return metrics


# ─── Summary ──────────────────────────────────────────────────────────────

def generate_summary():
    """Aggregate transfer learning results across folds."""
    pretrain_dir = os.path.join(RESULTS_DIR, "transfer")

    print("\n" + "=" * 70)
    print("  TRANSFER LEARNING SUMMARY")
    print("=" * 70)

    # Show pretrain info
    config_path = os.path.join(pretrain_dir, "pretrain_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"\n  Pretrain: {config['reaxys_samples']} Reaxys samples, "
              f"{config['epochs_trained']} epochs, {config['elapsed_seconds']}s")
        vm = config.get("val_metrics", {})
        print(f"  Val: RMSE={vm.get('RMSE', '?')}, MAE={vm.get('MAE', '?')}, "
              f"R²={vm.get('R2', '?')}")

    for strategy in ["all", "freeze"]:
        all_metrics = []
        for i in range(N_FOLDS):
            path = os.path.join(pretrain_dir, f"transfer_{strategy}_fold{i}_metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    all_metrics.append(json.load(f))

        if not all_metrics:
            print(f"\n  [{strategy}] No fold results found.")
            continue

        n_found = len(all_metrics)
        metric_keys = ["RMSE", "MAE", "R2", "Pearson_r"]
        agg = {"n_folds_found": n_found, "strategy": strategy}
        for mk in metric_keys:
            vals = [m[mk] for m in all_metrics]
            agg[f"{mk}_mean"] = round(np.mean(vals), 4)
            agg[f"{mk}_std"] = round(np.std(vals), 4)
            agg[f"{mk}_values"] = vals

        print(f"\n  Transfer ({strategy}): {n_found}/{N_FOLDS} folds")
        print(f"    RMSE: {agg['RMSE_mean']:.2f} +/- {agg['RMSE_std']:.2f}")
        print(f"    MAE:  {agg['MAE_mean']:.2f} +/- {agg['MAE_std']:.2f}")
        print(f"    R²:   {agg['R2_mean']:.4f} +/- {agg['R2_std']:.4f}")
        print(f"    r:    {agg['Pearson_r_mean']:.4f} +/- {agg['Pearson_r_std']:.4f}")

        agg_path = os.path.join(pretrain_dir, f"transfer_{strategy}_cv_aggregate.json")
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"    Saved to {agg_path}")

    print("\n  Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transfer learning: Reaxys → Joung+Beard")
    parser.add_argument("--pretrain", action="store_true",
                        help="Pretrain on Reaxys regression dataset")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune on Joung+Beard (requires pretrained model)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold to fine-tune (0-4)")
    parser.add_argument("--strategy", choices=["all", "freeze"], default="all",
                        help="Fine-tuning strategy: 'all' (lr=1e-4) or 'freeze' (freeze emb+gru1)")
    parser.add_argument("--summary", action="store_true",
                        help="Generate summary from saved results")
    args = parser.parse_args()

    if not any([args.pretrain, args.finetune, args.summary]):
        parser.error("Specify --pretrain, --finetune, or --summary")

    if args.summary:
        generate_summary()
        return

    # Load both datasets to build superset charset
    reaxys_combined, reaxys_y = load_reaxys()
    jb_combined, jb_y, jb_solvents = load_joung_beard()
    charset, embed_len = build_superset_charset(reaxys_combined, jb_combined)

    if args.pretrain:
        print("\n" + "=" * 70)
        print("  PRETRAINING on Reaxys")
        print("=" * 70)
        pretrain(charset, embed_len, reaxys_combined, reaxys_y)

    if args.finetune:
        print("\n" + "=" * 70)
        print(f"  FINE-TUNING on Joung+Beard (strategy={args.strategy})")
        print("=" * 70)

        fold_range = [args.fold] if args.fold is not None else range(N_FOLDS)
        for fold_idx in fold_range:
            finetune_fold(fold_idx, charset, embed_len,
                          jb_combined, jb_y, jb_solvents,
                          strategy=args.strategy)

        # Auto-aggregate if all folds done
        print("\n  Aggregating...")
        generate_summary()


if __name__ == "__main__":
    main()
