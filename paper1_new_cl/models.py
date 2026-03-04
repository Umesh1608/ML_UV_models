"""
Shared model/training functions for UV absorption (lambda_max) prediction.

Extracted from run_baselines.py so that run_baselines.py, eval_external.py,
and run_cross_dataset.py all share the same code.
"""

import gc
import json
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Return dict of RMSE, MAE, R², Pearson r."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2),
            "R2": round(r2, 4), "Pearson_r": round(r, 4)}


# ─── SMILES tokenization ───────────────────────────────────────────────────

def build_char_maps(charset):
    """Build char-to-int mapping from a sorted charset list."""
    return {c: i for i, c in enumerate(charset)}


def vectorize_smiles(smiles_arr, char_to_int, embed_len):
    """Convert SMILES strings to integer-encoded arrays.

    Each sequence is: [!] + [chars] + [E padding].
    Output shape: (n, embed_len - 1).
    """
    n = len(smiles_arr)
    X = np.zeros((n, embed_len - 1), dtype=np.int32)
    for i, smi in enumerate(smiles_arr):
        X[i, 0] = char_to_int.get("!", 0)
        for j, c in enumerate(smi):
            if j + 1 < embed_len - 1:
                X[i, j + 1] = char_to_int.get(c, 0)
        end_pos = len(smi) + 1
        if end_pos < embed_len - 1:
            X[i, end_pos:] = char_to_int.get("E", 0)
    return X


# ─── GPU verification ─────────────────────────────────────────────────────

def verify_gpu_available():
    """Verify GPU is available and can run computations. Call before DL training."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[GPU] WARNING: No GPU detected! DL training will be very slow on CPU.")
        return False

    # Quick benchmark: matmul on GPU
    try:
        with tf.device("/GPU:0"):
            a = tf.random.normal([500, 500])
            b = tf.random.normal([500, 500])
            t0 = time.time()
            c = tf.matmul(a, b)
            c.numpy()
            gpu_time = time.time() - t0
        print(f"[GPU] Verified: {gpus[0].name} — benchmark matmul in {gpu_time:.3f}s")
        return True
    except Exception as e:
        print(f"[GPU] WARNING: GPU computation failed: {e}")
        return False


# ─── tqdm Keras callback ───────────────────────────────────────────────────

def get_tqdm_callback():
    """Return TqdmProgressCallback class (lazy TF import)."""
    import tensorflow as tf

    class TqdmProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, total_epochs, model_name="", report_interval=10):
            super().__init__()
            self.total_epochs = total_epochs
            self.model_name = model_name
            self.report_interval = report_interval
            self.pbar = None
            self.epoch_start_time = None
            self.epoch_times = []

        def on_train_begin(self, logs=None):
            self.pbar = tqdm(
                total=self.total_epochs, desc=f"  {self.model_name}",
                unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}",
                file=sys.stdout
            )

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss", 0)
            loss = logs.get("loss", 0)
            epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
            self.epoch_times.append(epoch_time)
            self.pbar.set_postfix(
                loss=f"{loss:.2f}", val_loss=f"{val_loss:.2f}",
                t=f"{epoch_time:.0f}s"
            )
            self.pbar.update(1)

            # Detailed report every N epochs
            if (epoch + 1) % self.report_interval == 0:
                avg_time = np.mean(self.epoch_times[-self.report_interval:])
                print(f"\n  [{self.model_name}] Epoch {epoch+1}/{self.total_epochs} — "
                      f"loss={loss:.3f}, val_loss={val_loss:.3f}, "
                      f"avg {avg_time:.1f}s/epoch", flush=True)

        def on_train_end(self, logs=None):
            if self.pbar:
                self.pbar.close()
            if self.epoch_times:
                avg = np.mean(self.epoch_times)
                print(f"  [{self.model_name}] Training complete — "
                      f"{len(self.epoch_times)} epochs, avg {avg:.1f}s/epoch")

    return TqdmProgressCallback


# ─── Deep learning model builder ───────────────────────────────────────────

def create_dl_model(model_type, num_words, input_length, task="regression"):
    """Build a Keras Sequential model for SMILES-based prediction.

    Args:
        model_type: One of "bigru", "bilstm", "cnn_bigru".
        num_words: Vocabulary size (charset length).
        input_length: Sequence length (embed_len - 1).
        task: "regression" (linear output) or "classification" (sigmoid output).
    """
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Bidirectional, Conv1D, Dense, Dropout, Embedding, GRU, LSTM,
    )
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Embedding(num_words, 50, input_length=input_length))

    if model_type == "bigru":
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Bidirectional(GRU(128)))
    elif model_type == "bilstm":
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(128)))
    elif model_type == "cnn_bigru":
        model.add(Conv1D(192, 3, activation="relu"))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Bidirectional(GRU(128)))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))

    if task == "classification":
        model.add(Dense(1, activation="sigmoid", dtype="float32"))
    else:
        model.add(Dense(1, activation="linear", dtype="float32"))
    return model


def train_dl_model(model_type, X_train, X_test, y_train, y_test,
                   num_words, input_length, name="model",
                   results_dir="results", seed=7, epochs=250,
                   batch_size=80, patience=25,
                   X_val=None, y_val=None, task="regression"):
    """Train a DL model with early stopping. Returns (y_pred, metrics, model).

    Args:
        results_dir: Directory for checkpoint/history files.
        seed: Random seed for TF.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        patience: Early stopping patience.
        X_val: Optional separate validation set for early stopping.
               If provided, early stopping uses (X_val, y_val) instead of (X_test, y_test).
        y_val: Validation targets (required if X_val is provided).
        task: "regression" (MAE loss) or "classification" (BCE loss, sigmoid output).
    """
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)

    # Verify GPU is available before DL training
    verify_gpu_available()

    model = create_dl_model(model_type, num_words, input_length, task=task)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8)

    if task == "classification":
        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                      metrics=["accuracy"])
    else:
        model.compile(loss="mae", optimizer=optimizer, metrics=["mse"])

    # Use separate val set for early stopping if provided, otherwise use test set
    val_data = (X_val, y_val) if X_val is not None else (X_test, y_test)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=patience,
        verbose=0, mode="auto", restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(results_dir, f"{name}_best.keras"),
        monitor="val_loss", save_best_only=True, verbose=0
    )
    TqdmProgressCallback = get_tqdm_callback()
    progress = TqdmProgressCallback(epochs, model_name=name, report_interval=10)

    model.build(input_shape=(None, input_length))
    params = model.count_params()
    print(f"\n  >> {name} ({model_type}, {task}) — {params:,} params")

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=val_data,
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop, checkpoint, progress], verbose=0
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test, verbose=0).flatten()

    if task == "classification":
        # Return raw probabilities; caller handles thresholding and metrics
        print(f"  >> {name} done in {elapsed:.0f}s — "
              f"val_loss={history.history['val_loss'][-1]:.4f}")
        metrics = None
    else:
        metrics = compute_metrics(y_test.flatten(), y_pred)
        print(f"  >> {name} done in {elapsed:.0f}s — RMSE={metrics['RMSE']} "
              f"MAE={metrics['MAE']} R²={metrics['R2']} r={metrics['Pearson_r']}")

    # Save training history
    hist_path = os.path.join(results_dir, f"{name}_history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f)

    return y_pred, metrics, model


# ─── Hybrid model (multi-input: SMILES + Morgan FP) ──────────────────────

def create_hybrid_model(num_words, input_length, fp_dim=2048):
    """Build a multi-input Keras model: BiGRU on SMILES + Dense on Morgan FP.

    Args:
        num_words: Vocabulary size (charset length).
        input_length: SMILES sequence length (embed_len - 1).
        fp_dim: Morgan fingerprint dimension (default 2048).

    Returns:
        Compiled Keras Functional API model with two inputs.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Bidirectional, Concatenate, Dense, Dropout, Embedding, GRU, Input,
    )
    from tensorflow.keras.models import Model

    # SMILES branch
    smiles_input = Input(shape=(input_length,), name="smiles_input")
    x = Embedding(num_words, 50)(smiles_input)
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Bidirectional(GRU(128))(x)  # [batch, 256]

    # Morgan FP branch
    fp_input = Input(shape=(fp_dim,), name="fp_input")
    y = Dense(256, activation="relu")(fp_input)
    y = Dropout(0.2)(y)  # [batch, 256]

    # Merge
    merged = Concatenate()([x, y])  # [batch, 512]
    z = Dense(128, activation="relu")(merged)
    z = Dropout(0.2)(z)
    output = Dense(1, activation="sigmoid", dtype="float32")(z)

    model = Model(inputs=[smiles_input, fp_input], outputs=output)
    return model


# ─── Morgan fingerprints ─────────────────────────────────────────────────

def compute_morgan_fps(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan fingerprints for a list of SMILES strings.

    Returns:
        fps: np.ndarray of shape (n, n_bits)
        valid_mask: np.ndarray of booleans indicating valid molecules
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    valid_mask = []
    for smi in tqdm(smiles_list, desc="    Fingerprints", leave=False, file=sys.stdout):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
            valid_mask.append(True)
        else:
            fps.append(np.zeros(n_bits))
            valid_mask.append(False)
    return np.array(fps), np.array(valid_mask)
