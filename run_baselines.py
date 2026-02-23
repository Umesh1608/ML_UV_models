#!/usr/bin/env python3
"""
Run baseline experiments for UV absorption (lambda_max) prediction.

Models:
  1. BiGRU (with solvent)        — primary model
  2. BiGRU (without solvent)     — ablation
  3. BiLSTM (with solvent)       — architecture comparison
  4. CNN-BiGRU (with solvent)    — architecture comparison
  5. Random Forest + Morgan FP   — classical ML baseline
  6. XGBoost + Morgan FP         — classical ML baseline

All models use the same random train/test split (90/10, seed=7).
Metrics: RMSE, MAE, R², Pearson r
"""

import os
import sys
import time

# ─── GPU / CUDA setup ────────────────────────────────────────────────────────
# Point XLA at the libdevice file so TF can JIT-compile GPU kernels
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/home/umesh/.local/cuda_compat"
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress INFO/WARNING spam

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

# Verify GPU
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"[GPU] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Enable mixed precision for faster training on RTX 4090
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("[GPU] Mixed precision (float16) enabled")
else:
    print("[WARN] No GPU detected — training will be slow on CPU")

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "previous_code", "UV_canonical_full_dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 7
TEST_SIZE = 0.1
EPOCHS = 250
BATCH_SIZE = 256    # larger batch → fewer steps/epoch → much faster on GPU
PATIENCE = 25

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─── Metrics helper ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Return dict of RMSE, MAE, R², Pearson r."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2),
            "R2": round(r2, 4), "Pearson_r": round(r, 4)}


# ─── tqdm Keras callback ─────────────────────────────────────────────────────

class TqdmProgressCallback(tf.keras.callbacks.Callback):
    """Keras callback that shows a tqdm progress bar across epochs."""

    def __init__(self, total_epochs, model_name=""):
        super().__init__()
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(
            total=self.total_epochs, desc=f"  {self.model_name}",
            unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}",
            file=sys.stdout
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss", 0)
        loss = logs.get("loss", 0)
        self.pbar.set_postfix(loss=f"{loss:.2f}", val_loss=f"{val_loss:.2f}")
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()


# ─── Load and prepare data ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  UV λ_max BASELINE EXPERIMENTS")
print("=" * 70)

print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
df = df[["canon", "solvents", "lambda_max"]].dropna()

# Combined solute+solvent SMILES (with delimiter)
df["combined"] = df["canon"] + "!" + df["solvents"]

# Build charset and embedding length from combined column
all_smiles_combined = df["combined"].values
charset_combined = set("".join(all_smiles_combined) + "!E")
embed_combined = max(len(s) for s in all_smiles_combined) + 5

# Also for solute-only
all_smiles_solute = df["canon"].values
charset_solute = set("".join(all_smiles_solute) + "!E")
embed_solute = max(len(s) for s in all_smiles_solute) + 5

print(f"  Dataset: {len(df)} samples")
print(f"  Combined charset: {len(charset_combined)} tokens, max seq len: {embed_combined}")
print(f"  Solute-only charset: {len(charset_solute)} tokens, max seq len: {embed_solute}")

# Train/test split (same as original paper)
y = df["lambda_max"].values

smiles_combined = df["combined"].values
smiles_solute = df["canon"].values

(X_train_comb, X_test_comb,
 X_train_sol, X_test_sol,
 y_train, y_test) = train_test_split(
    smiles_combined, smiles_solute, y,
    test_size=TEST_SIZE, random_state=SEED
)

print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

# ─── Vectorization for deep learning models ──────────────────────────────────

def build_char_maps(charset):
    char_to_int = {c: i for i, c in enumerate(charset)}
    return char_to_int

def vectorize_smiles(smiles_arr, char_to_int, embed_len):
    """Convert SMILES strings to integer sequences."""
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

# Build maps
c2i_combined = build_char_maps(charset_combined)
c2i_solute = build_char_maps(charset_solute)

print("\n[2/6] Vectorizing sequences...")
X_train_comb_vec = vectorize_smiles(X_train_comb, c2i_combined, embed_combined)
X_test_comb_vec = vectorize_smiles(X_test_comb, c2i_combined, embed_combined)
X_train_sol_vec = vectorize_smiles(X_train_sol, c2i_solute, embed_solute)
X_test_sol_vec = vectorize_smiles(X_test_sol, c2i_solute, embed_solute)

y_train_r = y_train.reshape(-1, 1)
y_test_r = y_test.reshape(-1, 1)
print("  Done.")


# ─── Deep learning model builder ─────────────────────────────────────────────

def create_dl_model(model_type, num_words, input_length):
    """Create a Keras sequential model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, Bidirectional, GRU, LSTM, Dense, Dropout, Conv1D
    )

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
    model.add(Dense(1, activation="linear", dtype="float32"))  # output in fp32 for stability
    return model


def train_dl_model(model_type, X_train, X_test, y_train, y_test,
                   num_words, input_length, name="model"):
    """Train a DL model and return predictions + metrics."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = create_dl_model(model_type, num_words, input_length)
    # Scale LR with batch size: 0.001 * (256/32) = 0.008, capped at 0.005
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.005, rho=0.9, epsilon=1e-8)
    model.compile(loss="mae", optimizer=optimizer, metrics=["mse"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=PATIENCE,
        verbose=0, mode="auto", restore_best_weights=True
    )
    progress = TqdmProgressCallback(EPOCHS, model_name=name)

    model.build(input_shape=(None, input_length))
    params = model.count_params()
    print(f"\n  >> {name} ({model_type}) — {params:,} params")

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop, progress], verbose=0   # verbose=0 since tqdm handles it
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = compute_metrics(y_test.flatten(), y_pred)
    print(f"  >> {name} done in {elapsed:.0f}s — RMSE={metrics['RMSE']} MAE={metrics['MAE']} R²={metrics['R2']} r={metrics['Pearson_r']}")

    # Save training history
    hist_path = os.path.join(RESULTS_DIR, f"{name}_history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f)

    return y_pred, metrics


# ─── Fingerprint-based model builder ─────────────────────────────────────────

def compute_morgan_fps(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan fingerprints for a list of SMILES."""
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


def prepare_fp_features(solute_smiles, solvent_smiles):
    """Compute concatenated Morgan FPs for solute + solvent."""
    print("  Computing solute fingerprints...")
    fp_solute, mask1 = compute_morgan_fps(solute_smiles)
    print("  Computing solvent fingerprints...")
    fp_solvent, mask2 = compute_morgan_fps(solvent_smiles)
    features = np.hstack([fp_solute, fp_solvent])
    valid = mask1 & mask2
    return features, valid


def train_fp_model(model_class, model_name, X_train_fp, X_test_fp,
                   y_train_fp, y_test_fp, **kwargs):
    """Train a fingerprint-based model and return predictions + metrics."""
    print(f"\n  >> Training {model_name}...")
    t0 = time.time()
    model = model_class(**kwargs)
    model.fit(X_train_fp, y_train_fp.flatten())
    elapsed = time.time() - t0
    y_pred = model.predict(X_test_fp)
    metrics = compute_metrics(y_test_fp.flatten(), y_pred.flatten())
    print(f"  >> {model_name} done in {elapsed:.0f}s — RMSE={metrics['RMSE']} MAE={metrics['MAE']} R²={metrics['R2']} r={metrics['Pearson_r']}")
    return y_pred, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

all_results = {}

# Overall progress tracker
experiments = [
    ("BiGRU + Solvent",     "bigru",    True),
    ("BiGRU (no solvent)",  "bigru",    False),
    ("BiLSTM + Solvent",    "bilstm",   True),
    ("CNN-BiGRU + Solvent", "cnn_bigru", True),
    ("RF + Morgan FP",      "rf",       None),
    ("XGBoost + Morgan FP", "xgb",      None),
]

overall_bar = tqdm(experiments, desc="Overall progress",
                   unit="model", bar_format="{l_bar}{bar:30}{r_bar}",
                   file=sys.stdout)

print("\n[3/6] Training deep learning models (GPU)...")

for exp_name, model_type, with_solvent in overall_bar:
    overall_bar.set_postfix(current=exp_name)

    if model_type in ("bigru", "bilstm", "cnn_bigru"):
        if with_solvent:
            y_pred, metrics = train_dl_model(
                model_type,
                X_train_comb_vec, X_test_comb_vec, y_train_r, y_test_r,
                num_words=len(charset_combined), input_length=embed_combined - 1,
                name=exp_name.replace(" ", "_").replace("+", "w").replace("(", "").replace(")", "")
            )
        else:
            y_pred, metrics = train_dl_model(
                model_type,
                X_train_sol_vec, X_test_sol_vec, y_train_r, y_test_r,
                num_words=len(charset_solute), input_length=embed_solute - 1,
                name=exp_name.replace(" ", "_").replace("+", "w").replace("(", "").replace(")", "")
            )
        all_results[exp_name] = metrics

        # Store predictions by name for later
        if exp_name == "BiGRU + Solvent":
            y_pred_bigru = y_pred
        elif exp_name == "BiGRU (no solvent)":
            y_pred_bigru_nosol = y_pred
        elif exp_name == "BiLSTM + Solvent":
            y_pred_bilstm = y_pred
        elif exp_name == "CNN-BiGRU + Solvent":
            y_pred_cnnbigru = y_pred

    elif model_type in ("rf", "xgb"):
        # Prepare fingerprints on first classical model
        if model_type == "rf":
            print("\n[4/6] Preparing fingerprint features for classical ML...")
            solvent_train_smi = []
            solute_train_smi = []
            for s in X_train_comb:
                parts = s.split("!")
                solute_train_smi.append(parts[0])
                solvent_train_smi.append(parts[1] if len(parts) > 1 else "")

            solvent_test_smi = []
            solute_test_smi = []
            for s in X_test_comb:
                parts = s.split("!")
                solute_test_smi.append(parts[0])
                solvent_test_smi.append(parts[1] if len(parts) > 1 else "")

            X_train_fp, mask_train = prepare_fp_features(solute_train_smi, solvent_train_smi)
            X_test_fp, mask_test = prepare_fp_features(solute_test_smi, solvent_test_smi)

            X_train_fp_valid = X_train_fp[mask_train]
            y_train_fp_valid = y_train[mask_train]
            X_test_fp_valid = X_test_fp[mask_test]
            y_test_fp_valid = y_test[mask_test]
            print(f"  Valid: {mask_train.sum()}/{len(mask_train)} train, {mask_test.sum()}/{len(mask_test)} test")

            print("\n[5/6] Training classical ML baselines...")
            from sklearn.ensemble import RandomForestRegressor
            y_pred_rf, metrics_rf = train_fp_model(
                RandomForestRegressor, "RF + Morgan FP",
                X_train_fp_valid, X_test_fp_valid,
                y_train_fp_valid.reshape(-1, 1), y_test_fp_valid.reshape(-1, 1),
                n_estimators=500, max_depth=None, n_jobs=-1, random_state=SEED
            )
            all_results["RF + Morgan FP"] = metrics_rf
            y_pred = y_pred_rf

        elif model_type == "xgb":
            import xgboost as xgb
            y_pred_xgb, metrics_xgb = train_fp_model(
                xgb.XGBRegressor, "XGBoost + Morgan FP",
                X_train_fp_valid, X_test_fp_valid,
                y_train_fp_valid.reshape(-1, 1), y_test_fp_valid.reshape(-1, 1),
                n_estimators=500, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=SEED, tree_method="hist"
            )
            all_results["XGBoost + Morgan FP"] = metrics_xgb
            y_pred = y_pred_xgb


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(all_results).T
results_df = results_df[["RMSE", "MAE", "R2", "Pearson_r"]]
results_df = results_df.sort_values("RMSE")
print(results_df.to_string())

results_df.to_csv(os.path.join(RESULTS_DIR, "baseline_comparison.csv"))
print(f"\nResults saved to {os.path.join(RESULTS_DIR, 'baseline_comparison.csv')}")

# Save predictions for error analysis
np.savez(
    os.path.join(RESULTS_DIR, "predictions.npz"),
    y_test=y_test,
    y_test_fp=y_test_fp_valid if mask_test.sum() > 0 else y_test,
    y_pred_bigru=y_pred_bigru,
    y_pred_bigru_nosol=y_pred_bigru_nosol,
    y_pred_bilstm=y_pred_bilstm,
    y_pred_cnnbigru=y_pred_cnnbigru,
    y_pred_rf=y_pred_rf,
    y_pred_xgb=y_pred_xgb,
    test_smiles_combined=X_test_comb,
    solute_test_smi=np.array(solute_test_smi),
    solvent_test_smi=np.array(solvent_test_smi),
)
print("Predictions saved to predictions.npz")


# ═══════════════════════════════════════════════════════════════════════════════
# [6/6] GENERATE PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[6/6] Generating plots...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Parity plot for best model (BiGRU + Solvent) ---
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(y_test, y_pred_bigru, alpha=0.3, s=10, c="steelblue", edgecolors="none")
lims = [min(y_test.min(), y_pred_bigru.min()) - 10,
        max(y_test.max(), y_pred_bigru.max()) + 10]
ax.plot(lims, lims, "k--", lw=1, label="Ideal")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Experimental $\\lambda_{\\max}$ (nm)", fontsize=12)
ax.set_ylabel("Predicted $\\lambda_{\\max}$ (nm)", fontsize=12)
ax.set_title(f"BiGRU + Solvent (RMSE = {all_results['BiGRU + Solvent']['RMSE']:.1f} nm)", fontsize=13)
ax.set_aspect("equal")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "parity_plot.png"), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, "parity_plot.pdf"))
print("  Parity plot saved.")

# --- Error distribution histogram ---
errors_bigru = y_pred_bigru - y_test
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.hist(errors_bigru, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", lw=1)
ax.set_xlabel("Prediction Error (nm)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Error Distribution — BiGRU + Solvent", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.png"), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, "error_distribution.pdf"))
print("  Error distribution saved.")

# --- Error vs wavelength range ---
abs_errors = np.abs(errors_bigru)
ranges = [("< 300", y_test < 300),
          ("300-400", (y_test >= 300) & (y_test < 400)),
          ("400-500", (y_test >= 400) & (y_test < 500)),
          ("> 500", y_test >= 500)]

range_labels = []
range_means = []
range_stds = []
range_counts = []
for label, mask in ranges:
    if mask.sum() > 0:
        range_labels.append(label)
        range_means.append(abs_errors[mask].mean())
        range_stds.append(abs_errors[mask].std())
        range_counts.append(mask.sum())

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
bars = ax.bar(range_labels, range_means, yerr=range_stds, capsize=5,
              color="steelblue", edgecolor="white", alpha=0.85)
for bar, count in zip(bars, range_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"n={count}", ha="center", fontsize=9)
ax.set_xlabel("$\\lambda_{\\max}$ Range (nm)", fontsize=12)
ax.set_ylabel("Mean Absolute Error (nm)", fontsize=12)
ax.set_title("Error by Wavelength Range — BiGRU + Solvent", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "error_by_wavelength.png"), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, "error_by_wavelength.pdf"))
print("  Error by wavelength saved.")

# --- Error by top-5 solvents ---
solvent_test_arr = np.array(solvent_test_smi)
unique_solvents, counts = np.unique(solvent_test_arr, return_counts=True)
top5_idx = np.argsort(-counts)[:5]
top5_solvents = unique_solvents[top5_idx]

solvent_names = {
    "CCO": "Ethanol", "CO": "Methanol", "ClCCl": "DCM",
    "CC(C)=O": "Acetone", "CCCCCC": "Hexane", "CS(C)=O": "DMSO",
    "CC#N": "Acetonitrile", "O": "Water", "Cc1ccccc1": "Toluene",
    "C1CCOC1": "THF", "ClC(Cl)Cl": "Chloroform",
    "CCOCC": "Diethyl ether", "c1ccccc1": "Benzene",
    "C(Cl)(Cl)Cl": "Chloroform2",
}

sol_labels = []
sol_errors = []
for sol in top5_solvents:
    mask = solvent_test_arr == sol
    sol_labels.append(solvent_names.get(sol, sol[:15]))
    sol_errors.append(abs_errors[mask])

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
bp = ax.boxplot(sol_errors, labels=sol_labels, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("steelblue")
    patch.set_alpha(0.7)
ax.set_xlabel("Solvent", fontsize=12)
ax.set_ylabel("Absolute Error (nm)", fontsize=12)
ax.set_title("Error by Solvent — BiGRU + Solvent", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "error_by_solvent.png"), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, "error_by_solvent.pdf"))
print("  Error by solvent saved.")

print("\n" + "=" * 70)
print("  ALL EXPERIMENTS COMPLETE")
print("=" * 70)
