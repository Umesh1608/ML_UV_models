#!/usr/bin/env python3
"""
SELFIES experiment: Train BiGRU+Solvent on nablaColors using SELFIES representation
instead of SMILES, using the precomputed (author-defined) train/val/test split.

Compares token-level SELFIES encoding vs character-level SMILES encoding.

Usage:
  python3 run_selfies_experiment.py
"""

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import selfies as sf

from paper1_new_cl.models import compute_metrics, get_tqdm_callback

# ─── Config ──────────────────────────────────────────────────────────────────

SEED = 7
EPOCHS = 250
BATCH_SIZE = 32  # matches nablaColors cross-dataset config
PATIENCE = 25
LR = 0.001

DATA_PATH = "data/nablacolors_processed.csv"
SPLITS_PATH = "data/nablacolors_splits.npz"
RESULTS_DIR = "results/nablacolors_selfies"

# ─── GPU setup ───────────────────────────────────────────────────────────────

cuda_compat = os.path.expanduser("~/.local/cuda_compat/nvvm/libdevice")
if os.path.isdir(cuda_compat):
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.path.dirname(os.path.dirname(cuda_compat))}"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── SELFIES conversion ─────────────────────────────────────────────────────

def smiles_to_selfies(smiles_arr):
    """Convert array of SMILES to SELFIES. Returns (selfies_list, failed_indices)."""
    selfies_list = []
    failed = []
    for i, smi in enumerate(smiles_arr):
        se = sf.encoder(smi)
        if se is None:
            failed.append(i)
            selfies_list.append(None)
        else:
            selfies_list.append(se)
    return selfies_list, failed


def build_selfies_vocab(selfies_combined):
    """Build token vocabulary from combined SELFIES strings."""
    vocab = set()
    for se in selfies_combined:
        if se is not None:
            vocab.update(sf.split_selfies(se))
    # Add special tokens
    vocab.add("[PAD]")    # padding
    vocab.add("[SEP]")    # delimiter between solute and solvent
    vocab.add("[START]")  # start token
    return sorted(vocab)


def vectorize_selfies(selfies_arr, token_to_int, max_len):
    """Convert SELFIES strings to integer-encoded arrays (token-level).

    Each sequence: [START] + [tokens] + [PAD padding]
    Output shape: (n, max_len)
    """
    n = len(selfies_arr)
    X = np.zeros((n, max_len), dtype=np.int32)
    pad_idx = token_to_int.get("[PAD]", 0)
    start_idx = token_to_int.get("[START]", 0)

    for i, se in enumerate(selfies_arr):
        X[i, 0] = start_idx
        tokens = list(sf.split_selfies(se))
        for j, tok in enumerate(tokens):
            if j + 1 < max_len:
                X[i, j + 1] = token_to_int.get(tok, pad_idx)
        # Remaining positions are already 0 (PAD)
    return X


# ─── Model builder (same architecture, different vocab) ─────────────────────

def create_selfies_model(num_tokens, input_length, embed_dim=50):
    """BiGRU model for SELFIES token sequences."""
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Bidirectional, Dense, Dropout, Embedding, GRU,
    )
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Embedding(num_tokens, embed_dim, input_length=input_length))
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(Bidirectional(GRU(128)))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear", dtype="float32"))
    return model


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                num_tokens, input_length, name="selfies_bigru"):
    """Train BiGRU on SELFIES data with early stopping on val set."""
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = create_selfies_model(num_tokens, input_length)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LR, rho=0.9, epsilon=1e-8)
    model.compile(loss="mae", optimizer=optimizer, metrics=["mse"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=PATIENCE,
        verbose=0, mode="auto", restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(RESULTS_DIR, f"{name}_best.keras"),
        monitor="val_loss", save_best_only=True, verbose=0
    )
    TqdmProgressCallback = get_tqdm_callback()
    progress = TqdmProgressCallback(EPOCHS, model_name=name)

    model.build(input_shape=(None, input_length))
    params = model.count_params()
    print(f"\n  >> {name} — {params:,} params")

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, progress], verbose=0
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = compute_metrics(y_test.flatten(), y_pred)
    print(f"  >> Done in {elapsed:.0f}s — RMSE={metrics['RMSE']} "
          f"MAE={metrics['MAE']} R²={metrics['R2']} r={metrics['Pearson_r']}")

    # Save history
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(RESULTS_DIR, f"{name}_history.json"), "w") as f:
        json.dump(hist_data, f)

    return y_pred, metrics, model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SELFIES EXPERIMENT: BiGRU+Solvent on nablaColors")
    print("  Representation: SELFIES (token-level) vs SMILES (char-level)")
    print("  Split: precomputed (author-defined train/val/test)")
    print("=" * 70)

    # Check GPU
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n  GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("  Mixed precision: float16")
    else:
        print("\n  [WARN] No GPU detected — training will be slow")

    # Load data
    print(f"\n[DATA] Loading nablaColors from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    print(f"  Samples: {len(df)}")

    # Load precomputed splits
    splits = np.load(SPLITS_PATH)
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Convert SMILES → SELFIES
    print("\n[SELFIES] Converting SMILES to SELFIES...")
    t0 = time.time()

    solute_selfies, solute_failed = smiles_to_selfies(df["smiles"].values)
    solvent_selfies, solvent_failed = smiles_to_selfies(df["solvent_smiles"].values)

    n_failed = len(solute_failed) + len(solvent_failed)
    print(f"  Failed conversions: {n_failed} ({n_failed/len(df):.2%})")

    # Build combined SELFIES: solute [SEP] solvent
    combined_selfies = []
    valid_mask = []
    for i in range(len(df)):
        if solute_selfies[i] is not None and solvent_selfies[i] is not None:
            combined = solute_selfies[i] + "[SEP]" + solvent_selfies[i]
            combined_selfies.append(combined)
            valid_mask.append(True)
        else:
            combined_selfies.append(None)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    if not valid_mask.all():
        n_drop = (~valid_mask).sum()
        print(f"  Dropping {n_drop} rows with failed SELFIES conversion")

    y = df["lambda_max"].values.astype(np.float64)
    print(f"  Conversion time: {time.time() - t0:.1f}s")

    # Build vocabulary from ALL valid combined SELFIES
    print("\n[VOCAB] Building SELFIES token vocabulary...")
    vocab = build_selfies_vocab([c for c in combined_selfies if c is not None])
    token_to_int = {tok: i for i, tok in enumerate(vocab)}
    num_tokens = len(vocab)

    # Compute max sequence length (in tokens)
    max_tokens = 0
    for se in combined_selfies:
        if se is not None:
            toks = list(sf.split_selfies(se))
            max_tokens = max(max_tokens, len(toks))
    max_len = max_tokens + 5  # +5 for START + padding buffer

    print(f"  Vocabulary size: {num_tokens} tokens")
    print(f"  Max sequence length: {max_len} tokens")
    print(f"  λ_max: {y.min():.1f} – {y.max():.1f} nm (mean {y.mean():.1f})")

    # Filter split indices to valid rows only
    valid_indices = set(np.where(valid_mask)[0])
    train_idx = np.array([i for i in train_idx if i in valid_indices])
    val_idx = np.array([i for i in val_idx if i in valid_indices])
    test_idx = np.array([i for i in test_idx if i in valid_indices])
    print(f"  After filtering: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Vectorize
    print("\n[VECTORIZE] Encoding SELFIES sequences...")
    combined_arr = np.array(combined_selfies)

    X_train = vectorize_selfies(combined_arr[train_idx], token_to_int, max_len)
    X_val = vectorize_selfies(combined_arr[val_idx], token_to_int, max_len)
    X_test = vectorize_selfies(combined_arr[test_idx], token_to_int, max_len)

    y_train = y[train_idx].reshape(-1, 1)
    y_val = y[val_idx].reshape(-1, 1)
    y_test = y[test_idx].reshape(-1, 1)

    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # Save config
    config = {
        "representation": "SELFIES",
        "dataset": "nablaColors",
        "split": "precomputed",
        "vocab_size": num_tokens,
        "max_seq_len": max_len,
        "embed_dim": 50,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "vocab": vocab,
    }
    with open(os.path.join(RESULTS_DIR, "selfies_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train
    print(f"\n{'─' * 70}")
    print(f"  nablaColors SELFIES — Precomputed split")
    print(f"  (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
    print(f"{'─' * 70}")

    y_pred, metrics, model = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_tokens=num_tokens, input_length=max_len,
        name="bigru_selfies_precomputed"
    )

    # Save results
    with open(os.path.join(RESULTS_DIR, "bigru_selfies_precomputed_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "bigru_selfies_precomputed_predictions.npy"), y_pred)
    np.save(os.path.join(RESULTS_DIR, "bigru_selfies_precomputed_y_test.npy"), y_test.flatten())
    np.save(os.path.join(RESULTS_DIR, "bigru_selfies_precomputed_test_indices.npy"), test_idx)

    # Save model
    model.save(os.path.join(RESULTS_DIR, "bigru_selfies_precomputed_model.keras"))

    # Print comparison
    print(f"\n{'=' * 70}")
    print("  RESULTS COMPARISON")
    print(f"{'=' * 70}")
    print(f"  SELFIES BiGRU+Solvent:  RMSE={metrics['RMSE']}  MAE={metrics['MAE']}  "
          f"R²={metrics['R2']}  r={metrics['Pearson_r']}")

    # Load SMILES result if available
    smiles_path = "results/nablacolors/bigru_solvent_precomputed_metrics.json"
    if os.path.exists(smiles_path):
        with open(smiles_path) as f:
            sm = json.load(f)
        print(f"  SMILES BiGRU+Solvent:   RMSE={sm['RMSE']}  MAE={sm['MAE']}  "
              f"R²={sm['R2']}  r={sm['Pearson_r']}")
    else:
        print(f"  SMILES BiGRU+Solvent:   (not yet run — use run_cross_dataset.py --dataset nablacolors --split precomputed)")

    print(f"  Published UniProp:      MAE=15.97 (3D GNN, 27.7M params)")
    print(f"{'=' * 70}")

    # Cleanup
    del model, X_train, X_val, X_test
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
