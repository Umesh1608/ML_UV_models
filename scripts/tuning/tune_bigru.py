#!/usr/bin/env python3
"""
BiGRU hyperparameter tuning using Optuna TPE on fold 0 validation set.

Search space (from CLAUDE.md plan):
  - GRU units: {64, 128, 256}
  - GRU layers: {1, 2, 3}
  - Embedding dim: {32, 50, 100}
  - Dropout rate: [0.1, 0.4]
  - Learning rate: [1e-4, 3e-3] (log scale)
  - Batch size: {32, 64, 128}

Uses solvent-stratified fold 0 (train/val/test) from v2 splits.
Evaluates on val set; final best config evaluated on all 5 folds' test sets.

Usage:
  python3 tune_bigru.py                    # Run 30-trial Optuna search
  python3 tune_bigru.py --n-trials 10      # Quick test with 10 trials
  python3 tune_bigru.py --full-cv          # Run best config on all 5 folds
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
import optuna
import pandas as pd

from paper1_new_cl.models import (
    build_char_maps,
    compute_metrics,
    get_tqdm_callback,
    vectorize_smiles,
    verify_gpu_available,
)
from paper1_new_cl.splits import create_stratified_folds

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_PATH = os.path.join(SCRIPT_DIR, "previous_code", "UV_canonical_full_dataset.csv")
SEED = 7
EPOCHS = 250
PATIENCE = 25


def load_data():
    """Load dataset, compute charset/embed_len, return global data dict."""
    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "solvents", "lambda_max"]].dropna().reset_index(drop=True)
    df["combined"] = df["canon"] + "!" + df["solvents"]

    all_smiles = df["combined"].values
    charset = sorted(set("".join(all_smiles) + "!E"))
    embed_len = max(len(s) for s in all_smiles) + 5

    print(f"  Dataset: {len(df)} samples, charset: {len(charset)}, embed_len: {embed_len}")
    return df, charset, embed_len


def get_fold_splits(df):
    """Load or create stratified fold splits."""
    splits_path = os.path.join(RESULTS_DIR, "cv_fold_indices_stratified.npz")
    if os.path.exists(splits_path):
        splits = np.load(splits_path)
        folds = []
        for i in range(5):
            folds.append((splits[f"train_{i}"], splits[f"val_{i}"], splits[f"test_{i}"]))
        print(f"  Loaded existing stratified splits from {splits_path}")
        return folds
    else:
        return create_stratified_folds(len(df), df["solvents"].values, n_folds=5, seed=SEED)


def build_model(num_words, input_length, n_units, n_layers, embed_dim, dropout, lr):
    """Build a configurable BiGRU model."""
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Bidirectional, Dense, Dropout as DropoutLayer, Embedding, GRU,
    )
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Embedding(num_words, embed_dim, input_length=input_length))

    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        model.add(Bidirectional(GRU(n_units, return_sequences=return_seq)))

    model.add(Dense(n_units, activation="relu"))
    model.add(DropoutLayer(dropout))
    model.add(Dense(1, activation="linear", dtype="float32"))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-8)
    model.compile(loss="mae", optimizer=optimizer, metrics=["mse"])
    return model


def objective(trial, X_train, X_val, y_train, y_val, num_words, input_length):
    """Optuna objective: train BiGRU with suggested params, return val RMSE."""
    import tensorflow as tf

    n_units = trial.suggest_categorical("n_units", [64, 128, 256])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    embed_dim = trial.suggest_categorical("embed_dim", [32, 50, 100])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = build_model(num_words, input_length, n_units, n_layers, embed_dim, dropout, lr)
    model.build(input_shape=(None, input_length))
    params = model.count_params()
    print(f"\n  Trial {trial.number}: units={n_units}, layers={n_layers}, "
          f"embed={embed_dim}, dropout={dropout:.2f}, lr={lr:.5f}, "
          f"batch={batch_size}, params={params:,}")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=PATIENCE,
        verbose=0, mode="auto", restore_best_weights=True
    )

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=batch_size,
        callbacks=[early_stop], verbose=0,
    )
    elapsed = time.time() - t0
    n_epochs = len(history.history["loss"])

    # Evaluate on val set
    y_pred_val = model.predict(X_val, verbose=0).flatten()
    metrics = compute_metrics(y_val.flatten(), y_pred_val)
    val_rmse = metrics["RMSE"]

    print(f"  Trial {trial.number}: val_RMSE={val_rmse:.2f}, MAE={metrics['MAE']:.2f}, "
          f"epochs={n_epochs}, time={elapsed:.0f}s")

    # Report for pruning
    trial.set_user_attr("n_epochs", n_epochs)
    trial.set_user_attr("train_time_s", round(elapsed, 1))
    trial.set_user_attr("n_params", params)
    trial.set_user_attr("val_MAE", metrics["MAE"])

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return val_rmse


def run_search(n_trials=30):
    """Run Optuna TPE search on fold 0 val set."""
    verify_gpu_available()

    df, charset, embed_len = load_data()
    folds = get_fold_splits(df)
    train_idx, val_idx, test_idx = folds[0]

    c2i = build_char_maps(charset)
    num_words = len(charset)
    input_length = embed_len - 1

    print(f"  Vectorizing fold 0: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    X_train = vectorize_smiles(df["combined"].values[train_idx], c2i, embed_len)
    X_val = vectorize_smiles(df["combined"].values[val_idx], c2i, embed_len)
    y_train = df["lambda_max"].values[train_idx].reshape(-1, 1)
    y_val = df["lambda_max"].values[val_idx].reshape(-1, 1)

    # Use SQLite storage so trials persist across runs (stop/resume safe)
    db_path = os.path.join(RESULTS_DIR, "bigru_hpo.db")
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="bigru_hpo",
        storage=storage,
        load_if_exists=True,  # Resume from previous trials if DB exists
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    if completed > 0:
        print(f"  Resuming: {completed} trials already done, {remaining} remaining")
        print(f"  Current best: val RMSE = {study.best_value:.2f}")

    if remaining > 0:
        study.optimize(
            lambda trial: objective(trial, X_train, X_val, y_train, y_val, num_words, input_length),
            n_trials=remaining,
            show_progress_bar=True,
        )
    else:
        print(f"  All {n_trials} trials already complete.")

    # Report results
    print(f"\n{'='*60}")
    print(f"  OPTUNA SEARCH COMPLETE ({n_trials} trials)")
    print(f"{'='*60}")
    print(f"  Best val RMSE: {study.best_value:.2f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Also evaluate best trial on test set
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    bp = study.best_params
    model = build_model(num_words, input_length,
                        bp["n_units"], bp["n_layers"], bp["embed_dim"],
                        bp["dropout"], bp["lr"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-5, patience=PATIENCE,
        verbose=0, mode="auto", restore_best_weights=True
    )
    TqdmProgressCallback = get_tqdm_callback()
    progress = TqdmProgressCallback(EPOCHS, model_name="best_retrain", report_interval=10)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=bp["batch_size"],
              callbacks=[early_stop, progress], verbose=0)

    X_test = vectorize_smiles(df["combined"].values[test_idx], c2i, embed_len)
    y_test = df["lambda_max"].values[test_idx]
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    test_metrics = compute_metrics(y_test, y_pred_test)
    print(f"\n  Best config TEST metrics (fold 0):")
    print(f"    RMSE={test_metrics['RMSE']}, MAE={test_metrics['MAE']}, "
          f"R²={test_metrics['R2']}, r={test_metrics['Pearson_r']}")

    # Save
    best_config = {**study.best_params, "best_val_rmse": study.best_value,
                   "test_metrics_fold0": test_metrics}
    config_path = os.path.join(RESULTS_DIR, "bigru_tuned_best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"  Saved best config to {config_path}")

    # Save all trial results
    trials_data = []
    for t in study.trials:
        row = {**t.params, "val_rmse": t.value}
        row.update(t.user_attrs)
        trials_data.append(row)
    trials_df = pd.DataFrame(trials_data).sort_values("val_rmse")
    trials_df.to_csv(os.path.join(RESULTS_DIR, "bigru_hpo_trials.csv"), index=False)
    print(f"  Saved all trial results to results/bigru_hpo_trials.csv")

    # Compare with default
    print(f"\n  Default BiGRU v2:  RMSE 36.45 ± 1.12 (5-fold)")
    print(f"  Best tuned (fold 0 test): RMSE {test_metrics['RMSE']}")

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return best_config


def _get_periodic_checkpoint_callback(save_path, save_freq=10):
    """Callback that saves full model every save_freq epochs for mid-fold resume."""
    import tensorflow as tf

    class PeriodicCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, save_path, save_freq):
            super().__init__()
            self.save_path = save_path
            self.save_freq = save_freq
            self.state_path = save_path.replace(".keras", "_state.json")

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.save_freq == 0:
                self.model.save(self.save_path)
                state = {
                    "epoch": epoch + 1,
                    "val_loss": float(logs.get("val_loss", 0)),
                    "loss": float(logs.get("loss", 0)),
                }
                with open(self.state_path, "w") as f:
                    json.dump(state, f)
                print(f"\n    [checkpoint] Saved at epoch {epoch+1}, "
                      f"val_loss={state['val_loss']:.3f}", flush=True)

    return PeriodicCheckpoint(save_path, save_freq)


def _save_partial_aggregate(fold_metrics, config):
    """Save partial aggregate after each completed fold so progress is visible."""
    if not fold_metrics:
        return
    agg = {"n_folds_complete": len(fold_metrics)}
    for key in ["RMSE", "MAE", "R2", "Pearson_r"]:
        vals = [m[key] for m in fold_metrics]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 2)
        if len(vals) > 1:
            agg[f"{key}_std"] = round(float(np.std(vals)), 2)
    agg["config"] = {k: config[k] for k in ["n_units", "n_layers", "embed_dim",
                                              "dropout", "lr", "batch_size"]}
    path = os.path.join(RESULTS_DIR, "bigru_tuned_cv_aggregate.json")
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)


def run_full_cv(config=None):
    """Run best config on all 5 folds.

    STOP/RESUME SAFE:
      - Completed folds are detected via _metrics.json and skipped automatically.
      - Mid-fold resume: periodic checkpoint every 10 epochs to _latest.keras;
        on restart, training resumes from the last checkpoint epoch.
      - SIGINT (Ctrl+C) triggers a graceful save of the current model before exit.
      - Partial aggregate saved after each fold so progress is always visible.
    """
    if config is None:
        config_path = os.path.join(RESULTS_DIR, "bigru_tuned_best_config.json")
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Loaded config from {config_path}")

    import signal
    import tensorflow as tf

    verify_gpu_available()

    df, charset, embed_len = load_data()
    folds = get_fold_splits(df)
    c2i = build_char_maps(charset)
    num_words = len(charset)
    input_length = embed_len - 1

    # Also save config for eval_wetlab.py compatibility
    config_out = {
        "charset": charset,
        "embed_len": embed_len,
        "char_to_int": c2i,
        "model_type": "bigru_tuned",
        "with_solvent": True,
        "n_folds": 5,
        "seed": SEED,
        "n_units": config["n_units"],
        "n_layers": config["n_layers"],
        "embed_dim": config["embed_dim"],
        "dropout": config["dropout"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
    }
    config_out_path = os.path.join(RESULTS_DIR, "bigru_tuned_config.json")
    with open(config_out_path, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"  Saved full config to {config_out_path}")

    n_units = config["n_units"]
    n_layers = config["n_layers"]
    embed_dim = config["embed_dim"]
    dropout = config["dropout"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    print(f"\n  Config: units={n_units}, layers={n_layers}, embed={embed_dim}, "
          f"dropout={dropout:.2f}, lr={lr:.5f}, batch={batch_size}")

    # ── Collect already-completed folds ──
    fold_metrics = []
    all_preds = []
    all_y_test = []
    all_test_idx = []

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(folds):
        prefix = f"bigru_tuned_fold{fold_i}"
        metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.json")

        # ── Skip completed folds ──
        if os.path.exists(metrics_path):
            model_path = os.path.join(RESULTS_DIR, f"{prefix}_model.keras")
            pred_path = os.path.join(RESULTS_DIR, f"{prefix}_predictions.npy")
            ytest_path = os.path.join(RESULTS_DIR, f"{prefix}_y_test.npy")
            tidx_path = os.path.join(RESULTS_DIR, f"{prefix}_test_indices.npy")

            if all(os.path.exists(p) for p in [model_path, pred_path, ytest_path, tidx_path]):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                fold_metrics.append(metrics)
                all_preds.append(np.load(pred_path))
                all_y_test.append(np.load(ytest_path))
                all_test_idx.append(np.load(tidx_path))
                print(f"\n  FOLD {fold_i}/4 — ALREADY COMPLETE "
                      f"(RMSE={metrics['RMSE']}, MAE={metrics['MAE']}). Skipping.")
                continue

        print(f"\n{'='*60}")
        print(f"  FOLD {fold_i}/4")
        print(f"{'='*60}")

        X_train = vectorize_smiles(df["combined"].values[train_idx], c2i, embed_len)
        X_val = vectorize_smiles(df["combined"].values[val_idx], c2i, embed_len)
        X_test = vectorize_smiles(df["combined"].values[test_idx], c2i, embed_len)
        y_train = df["lambda_max"].values[train_idx].reshape(-1, 1)
        y_val = df["lambda_max"].values[val_idx].reshape(-1, 1)
        y_test = df["lambda_max"].values[test_idx]

        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED + fold_i)

        model = build_model(num_words, input_length, n_units, n_layers, embed_dim, dropout, lr)
        model.build(input_shape=(None, input_length))
        params = model.count_params()
        print(f"  Model: {params:,} params")

        # ── Check for mid-fold checkpoint to resume from ──
        latest_path = os.path.join(RESULTS_DIR, f"{prefix}_latest.keras")
        state_path = os.path.join(RESULTS_DIR, f"{prefix}_latest_state.json")
        initial_epoch = 0

        if os.path.exists(latest_path) and os.path.exists(state_path):
            try:
                model = tf.keras.models.load_model(latest_path)
                with open(state_path) as f:
                    state = json.load(f)
                initial_epoch = state["epoch"]
                print(f"  RESUMING from epoch {initial_epoch} "
                      f"(val_loss={state.get('val_loss', '?')})")
            except Exception as e:
                print(f"  WARNING: Could not load checkpoint ({e}), starting fresh")
                initial_epoch = 0
                model = build_model(num_words, input_length, n_units, n_layers,
                                    embed_dim, dropout, lr)

        # ── SIGINT handler: save checkpoint before exit ──
        _interrupted = [False]

        def _sigint_handler(sig, frame):
            if _interrupted[0]:
                print("\n  FORCE QUIT (second Ctrl+C)")
                sys.exit(1)
            _interrupted[0] = True
            print(f"\n  SIGINT received — saving checkpoint for fold {fold_i}...")
            try:
                model.save(latest_path)
                # We don't know the exact epoch here, but the periodic checkpoint
                # already saved the latest state. Just save the model.
                print(f"  Emergency checkpoint saved to {latest_path}")
            except Exception as e:
                print(f"  WARNING: Could not save checkpoint: {e}")
            print("  Exiting. Re-run `--full-cv` to resume from this fold.")
            sys.exit(0)

        old_handler = signal.signal(signal.SIGINT, _sigint_handler)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-5, patience=PATIENCE,
            verbose=0, mode="auto", restore_best_weights=True
        )
        best_ckpt = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, f"{prefix}_best.keras"),
            monitor="val_loss", save_best_only=True, verbose=0
        )
        periodic_ckpt = _get_periodic_checkpoint_callback(latest_path, save_freq=10)
        TqdmProgressCallback = get_tqdm_callback()
        progress = TqdmProgressCallback(EPOCHS, model_name=f"bigru_tuned_fold{fold_i}",
                                        report_interval=10)

        t0 = time.time()
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=EPOCHS, batch_size=batch_size,
                  initial_epoch=initial_epoch,
                  callbacks=[early_stop, best_ckpt, periodic_ckpt, progress],
                  verbose=0)
        elapsed = time.time() - t0

        # Restore SIGINT handler
        signal.signal(signal.SIGINT, old_handler)

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = compute_metrics(y_test, y_pred)
        fold_metrics.append(metrics)

        print(f"\n  Fold {fold_i}: RMSE={metrics['RMSE']:.2f} MAE={metrics['MAE']:.2f} "
              f"R²={metrics['R2']:.4f} r={metrics['Pearson_r']:.4f} ({elapsed:.0f}s)")

        # Save fold results (marks fold as COMPLETE)
        np.save(os.path.join(RESULTS_DIR, f"{prefix}_predictions.npy"), y_pred)
        np.save(os.path.join(RESULTS_DIR, f"{prefix}_y_test.npy"), y_test)
        np.save(os.path.join(RESULTS_DIR, f"{prefix}_test_indices.npy"), test_idx)
        model.save(os.path.join(RESULTS_DIR, f"{prefix}_model.keras"))

        metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        all_preds.append(y_pred)
        all_y_test.append(y_test)
        all_test_idx.append(test_idx)

        # Clean up mid-fold checkpoints (fold is done)
        for tmp in [latest_path, state_path]:
            if os.path.exists(tmp):
                os.remove(tmp)

        # Save partial aggregate after each fold
        _save_partial_aggregate(fold_metrics, config)
        n_done = len(fold_metrics)
        print(f"\n  Progress: {n_done}/5 folds complete")
        if n_done > 1:
            rmse_vals = [m["RMSE"] for m in fold_metrics]
            mae_vals = [m["MAE"] for m in fold_metrics]
            print(f"  Running avg — RMSE: {np.mean(rmse_vals):.2f} ± {np.std(rmse_vals):.2f}, "
                  f"MAE: {np.mean(mae_vals):.2f} ± {np.std(mae_vals):.2f}")

        del model, X_train, X_val, X_test
        tf.keras.backend.clear_session()
        gc.collect()

    # ── Final aggregate ──
    np.savez(os.path.join(RESULTS_DIR, "bigru_tuned_cv_pooled.npz"),
             y_pred=np.concatenate(all_preds),
             y_true=np.concatenate(all_y_test),
             test_indices=np.concatenate(all_test_idx))

    agg = {}
    for key in ["RMSE", "MAE", "R2", "Pearson_r"]:
        vals = [m[key] for m in fold_metrics]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 2)
        agg[f"{key}_std"] = round(float(np.std(vals)), 2)

    agg["n_folds_complete"] = len(fold_metrics)
    agg["config"] = {k: config[k] for k in ["n_units", "n_layers", "embed_dim",
                                              "dropout", "lr", "batch_size"]}
    with open(os.path.join(RESULTS_DIR, "bigru_tuned_cv_aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  AGGREGATE (5-fold CV)")
    print(f"{'='*60}")
    print(f"  RMSE: {agg['RMSE_mean']:.2f} ± {agg['RMSE_std']:.2f}")
    print(f"  MAE:  {agg['MAE_mean']:.2f} ± {agg['MAE_std']:.2f}")
    print(f"  R²:   {agg['R2_mean']:.4f} ± {agg['R2_std']:.4f}")
    print(f"  r:    {agg['Pearson_r_mean']:.4f} ± {agg['Pearson_r_std']:.4f}")

    print(f"\n  Default BiGRU v2:  RMSE 36.45 ± 1.12, MAE 20.70 ± 0.51")
    print(f"  Tuned BiGRU:       RMSE {agg['RMSE_mean']:.2f} ± {agg['RMSE_std']:.2f}, "
          f"MAE {agg['MAE_mean']:.2f} ± {agg['MAE_std']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiGRU hyperparameter tuning (Optuna)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--full-cv", action="store_true",
                        help="Run full 5-fold CV with best config")
    args = parser.parse_args()

    if args.full_cv:
        run_full_cv()
    else:
        run_search(n_trials=args.n_trials)
