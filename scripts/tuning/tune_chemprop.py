#!/usr/bin/env python3
"""
R1-12 Step 0 — Optuna probe of Chemprop hyperparameters.

The reviewer (R1-12) flagged that BiGRU received 25 Optuna trials of HPO while
Chemprop was left at its published defaults, and asked whether the comparison
is fair. This script runs a comparable 25-trial Optuna probe on Chemprop v2 on
fold 0 of the v3 cleaned Joung+Beard dataset, reporting whether the best-found
configuration meaningfully beats the published defaults.

Search space (Chemprop v2 multicomponent solute+solvent):
  d_h           ∈ {200, 300 (default), 400, 600, 900}      # MPNN hidden dim
  depth         ∈ {3 (default), 4, 5, 6}                    # message-passing steps T
  dropout       ∈ {0.0 (default), 0.1, 0.2, 0.3}            # MPNN dropout
  ffn_n_layers  ∈ {1 (default), 2, 3}                       # regression head depth
  ffn_h         ∈ {300 (default), 500, 800}                 # regression head width
  batch_size    ∈ {32, 50, 64 (default), 128}

Training per trial:
  max 100 epochs, early stopping on val MAE (patience=15),
  Noam LR schedule (warmup 1e-4→1e-3→1e-4), Huber-like loss internally,
  MedianPruner (n_startup_trials=5, n_warmup_steps=20) to kill clearly-bad runs.

Storage: SQLite study file → resume-safe if pod restarts.

Outputs:
  results/r1_12_chemprop_optuna.db        Optuna SQLite storage
  results/r1_12_chemprop_probe.json       best config + trial table
  results/r1_12_chemprop_probe_summary.md human-readable
  logs/r1_12_chemprop_trial_<N>.log       per-trial chemprop log
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

DATA_PATH = SCRIPT_DIR / "previous_code" / "UV_canonical_v3_dedup.csv"
RESULTS = SCRIPT_DIR / "results"
LOGS = SCRIPT_DIR / "logs"
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

FOLD = 0
N_TRIALS = 25
SEED = 7
MAX_EPOCHS = 100
PATIENCE = 15
WARMUP_EPOCHS = 2
INIT_LR = 1e-4
MAX_LR = 1e-3
FINAL_LR = 1e-4

STUDY_DB = RESULTS / "r1_12_chemprop_optuna.db"
STUDY_NAME = "chemprop_v3_fold0"
OUT_JSON = RESULTS / "r1_12_chemprop_probe.json"
OUT_MD = RESULTS / "r1_12_chemprop_probe_summary.md"

# Chemprop v2 published defaults — used as the comparison baseline
DEFAULT_CFG = {
    "d_h": 300,
    "depth": 3,
    "dropout": 0.0,
    "ffn_n_layers": 1,
    "ffn_h": 300,
    "batch_size": 64,
}


# ────────────────────────────────────────────────────────────────────────────
# Data and fold loading
# ────────────────────────────────────────────────────────────────────────────

def load_data_fold0():
    """Load v3 dataset and fold-0 split. Returns (solutes, solvents, y, fold)."""
    from rdkit import Chem
    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "solvents", "lambda_max"]].dropna()
    # Pre-validate SMILES (Chemprop raises on invalid)
    mask = np.array([
        Chem.MolFromSmiles(s) is not None and Chem.MolFromSmiles(v) is not None
        for s, v in zip(df["canon"], df["solvents"])
    ])
    n_invalid = int((~mask).sum())
    if n_invalid > 0:
        print(f"  Skipping {n_invalid} invalid-SMILES rows")
        df = df[mask].reset_index(drop=True)
    splits = np.load(RESULTS / "cv_fold_indices_v3.npz")
    fold = (splits[f"train_{FOLD}"], splits[f"val_{FOLD}"], splits[f"test_{FOLD}"])
    print(f"  v3 dataset rows (after SMILES validation): {len(df)}")
    print(f"  fold {FOLD}: train={len(fold[0])}, val={len(fold[1])}, test={len(fold[2])}")
    return df["canon"].values, df["solvents"].values, \
           df["lambda_max"].values.astype(np.float64), fold


def build_loaders(solutes, solvents, y, fold, batch_size):
    """Build Chemprop multicomponent dataloaders for fold 0.

    Mirrors the construction in run_chemprop.py: MoleculeDataset takes a list
    of MoleculeDatapoints plus a SimpleMoleculeMolGraphFeaturizer; y values
    are attached to the solute datapoint and the solvent datapoint carries
    no y. The training MulticomponentDataset returns its scaler from
    normalize_targets(); the val dataset applies the same scaler.
    """
    from chemprop import data, featurizers

    train_idx, val_idx, _ = fold
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def _datapoints(idx):
        solute_dps = [data.MoleculeDatapoint.from_smi(solutes[i], y=np.array([y[i]]))
                      for i in idx]
        solvent_dps = [data.MoleculeDatapoint.from_smi(solvents[i]) for i in idx]
        return solute_dps, solvent_dps

    solute_tr, solvent_tr = _datapoints(train_idx)
    solute_va, solvent_va = _datapoints(val_idx)

    train_mc = data.MulticomponentDataset([
        data.MoleculeDataset(solute_tr, feat),
        data.MoleculeDataset(solvent_tr, feat),
    ])
    val_mc = data.MulticomponentDataset([
        data.MoleculeDataset(solute_va, feat),
        data.MoleculeDataset(solvent_va, feat),
    ])
    scaler = train_mc.normalize_targets()
    val_mc.normalize_targets(scaler)

    train_loader = data.build_dataloader(
        train_mc, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = data.build_dataloader(
        val_mc, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader, scaler, y[val_idx]


def build_model(scaler, d_h, depth, dropout, ffn_n_layers, ffn_h):
    """Build Chemprop multicomponent MPNN with the given hyperparameters."""
    from chemprop import nn
    from chemprop.models import multi
    from chemprop.nn.transforms import UnscaleTransform
    from chemprop.nn.metrics import RMSE as ChempropRMSE, MAE as ChempropMAE

    agg = nn.MeanAggregation()
    output_transform = UnscaleTransform.from_standard_scaler(scaler)
    mcmp = nn.MulticomponentMessagePassing(
        blocks=[nn.BondMessagePassing(d_h=d_h, depth=depth, dropout=dropout)
                for _ in range(2)],
        n_components=2,
    )
    ffn = nn.RegressionFFN(
        input_dim=mcmp.output_dim,
        hidden_dim=ffn_h,
        n_layers=ffn_n_layers,
        dropout=dropout,
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
    return model


# ────────────────────────────────────────────────────────────────────────────
# Single-trial training and evaluation
# ────────────────────────────────────────────────────────────────────────────

def evaluate_val_rmse(model, val_loader, y_val_true):
    """Run model.predict on val_loader and return val RMSE in nm."""
    import torch
    import lightning.pytorch as pl
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False,
    )
    preds = trainer.predict(model, val_loader)
    y_pred = torch.cat(preds, dim=0).squeeze(-1).numpy()
    return float(np.sqrt(np.mean((y_pred - y_val_true) ** 2)))


def train_one_trial(trial_id, cfg, solutes, solvents, y, fold, log_path):
    """Train Chemprop on fold 0 with the given config; return val RMSE."""
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping

    train_loader, val_loader, scaler, y_val_true = build_loaders(
        solutes, solvents, y, fold, cfg["batch_size"]
    )
    model = build_model(
        scaler,
        d_h=cfg["d_h"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
        ffn_n_layers=cfg["ffn_n_layers"],
        ffn_h=cfg["ffn_h"],
    )
    n_params = sum(p.numel() for p in model.parameters())

    early = EarlyStopping(
        monitor="val_loss", patience=PATIENCE,
        mode="min", min_delta=1e-4,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[early],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=200,
    )
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    dt = time.time() - t0
    val_rmse = evaluate_val_rmse(model, val_loader, y_val_true)
    # Append trial summary to its log
    with open(log_path, "a") as f:
        f.write(f"\n--- trial {trial_id} ---\n")
        f.write(f"  config: {cfg}\n")
        f.write(f"  n_params: {n_params}\n")
        f.write(f"  epochs_trained: {trainer.current_epoch}\n")
        f.write(f"  wallclock_s: {dt:.0f}\n")
        f.write(f"  val_RMSE: {val_rmse:.4f}\n")
    del model, trainer
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return val_rmse, dt, n_params


# ────────────────────────────────────────────────────────────────────────────
# Optuna study driver
# ────────────────────────────────────────────────────────────────────────────

def objective_factory(solutes, solvents, y, fold):
    def objective(trial):
        cfg = {
            "d_h":          trial.suggest_categorical("d_h",          [200, 300, 400, 600, 900]),
            "depth":        trial.suggest_categorical("depth",        [3, 4, 5, 6]),
            "dropout":      trial.suggest_categorical("dropout",      [0.0, 0.1, 0.2, 0.3]),
            "ffn_n_layers": trial.suggest_categorical("ffn_n_layers", [1, 2, 3]),
            "ffn_h":        trial.suggest_categorical("ffn_h",        [300, 500, 800]),
            "batch_size":   trial.suggest_categorical("batch_size",   [32, 50, 64, 128]),
        }
        log_path = LOGS / f"r1_12_chemprop_trial_{trial.number}.log"
        print(f"\n[TRIAL {trial.number}/{N_TRIALS}] {cfg}")
        try:
            val_rmse, dt, n_params = train_one_trial(
                trial.number, cfg, solutes, solvents, y, fold, log_path
            )
            trial.set_user_attr("wallclock_s", round(dt, 1))
            trial.set_user_attr("n_params", n_params)
            print(f"  -> val RMSE = {val_rmse:.3f} nm  ({dt:.0f}s, {n_params:,} params)")
            return val_rmse
        except Exception as e:
            print(f"  -> FAILED: {type(e).__name__}: {e}")
            with open(log_path, "a") as f:
                f.write(f"\n--- trial {trial.number} FAILED ---\n{e}\n")
            raise
    return objective


def main():
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    print(f"[LOAD] {DATA_PATH}")
    solutes, solvents, y, fold = load_data_fold0()

    storage = f"sqlite:///{STUDY_DB.as_posix()}"
    sampler = TPESampler(seed=SEED, n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Enqueue the published default config as trial 0 so we always have a
    # baseline measurement in the study even if TPE never proposes it.
    if len(study.trials) == 0:
        study.enqueue_trial(DEFAULT_CFG)
        print("  enqueued DEFAULT_CFG as trial 0")

    done = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    print(f"  study has {done} completed trials; running up to {N_TRIALS} total")

    objective = objective_factory(solutes, solvents, y, fold)
    study.optimize(
        objective,
        n_trials=max(0, N_TRIALS - len(study.trials)),
        show_progress_bar=False,
        catch=(Exception,),
    )

    # Save best config + trial table
    best = study.best_trial
    print(f"\n[BEST] trial {best.number}: val RMSE = {best.value:.3f} nm")
    print(f"  config: {best.params}")
    # Compare to default
    default_val_rmse = None
    for t in study.trials:
        if t.state.name == "COMPLETE" and dict(t.params) == DEFAULT_CFG:
            default_val_rmse = t.value
            break
    if default_val_rmse is not None:
        delta = default_val_rmse - best.value
        print(f"\n[COMPARE] default config val RMSE = {default_val_rmse:.3f} nm")
        print(f"          best Optuna   val RMSE = {best.value:.3f} nm")
        print(f"          improvement (default - best) = {delta:+.3f} nm")
    else:
        delta = None
        print(f"\n[COMPARE] default config trial result not found in study")

    trial_rows = []
    for t in study.trials:
        row = {
            "trial": t.number, "state": t.state.name, "value": t.value,
            **{k: v for k, v in t.params.items()},
            "wallclock_s": t.user_attrs.get("wallclock_s"),
            "n_params": t.user_attrs.get("n_params"),
        }
        trial_rows.append(row)
    summary = {
        "study_name": STUDY_NAME,
        "data": str(DATA_PATH.name),
        "fold": FOLD,
        "n_trials_completed": done,
        "default_cfg": DEFAULT_CFG,
        "default_val_rmse": default_val_rmse,
        "best_trial": best.number,
        "best_val_rmse": best.value,
        "best_cfg": dict(best.params),
        "best_minus_default_nm": delta,
        "trials": trial_rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[OUT] {OUT_JSON}")

    # Human-readable markdown
    lines = []
    lines.append("# R1-12 Step 0 — Chemprop HPO Probe (v3 fold 0)\n")
    lines.append(f"_25-trial Optuna TPE probe on Chemprop v2 hyperparameters_\n")
    lines.append(f"## Headline")
    lines.append(f"- Trials completed: **{done} / {N_TRIALS}**")
    lines.append(f"- Default config val RMSE: **{default_val_rmse if default_val_rmse is None else f'{default_val_rmse:.3f}'}** nm")
    lines.append(f"- Best Optuna config val RMSE: **{best.value:.3f}** nm")
    if delta is not None:
        lines.append(f"- Improvement (default − best): **{delta:+.3f} nm**")
        if delta < 1.0:
            bucket = "**NEAR-OPTIMAL DEFAULTS** (<1 nm improvement). Keep defaults; SI-only update + literature backstop."
        elif 1.0 <= delta < 2.5:
            bucket = "**MODEST IMPROVEMENT** (1--2.5 nm). Judgment call; lean keep defaults but report probe transparently."
        else:
            bucket = "**SUBSTANTIAL IMPROVEMENT** (≥2.5 nm). Trigger full 5-fold retrain with best config."
        lines.append(f"- Decision-rule bucket: {bucket}")
    lines.append("")
    lines.append(f"## Best config")
    for k, v in best.params.items():
        lines.append(f"  - `{k}` = {v}")
    lines.append(f"  - n_params: {best.user_attrs.get('n_params'):,}" if best.user_attrs.get('n_params') else "  - n_params: ?")
    lines.append(f"  - wallclock: {best.user_attrs.get('wallclock_s'):.0f}s")
    lines.append("")
    lines.append(f"## All completed trials (sorted by val RMSE)\n")
    lines.append("| trial | val RMSE | d_h | depth | dropout | ffn_n_layers | ffn_h | batch | wallclock | params |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    rows_ok = [r for r in trial_rows if r["state"] == "COMPLETE" and r["value"] is not None]
    rows_ok.sort(key=lambda r: r["value"])
    for r in rows_ok:
        lines.append(
            f"| {r['trial']} | {r['value']:.3f} | {r.get('d_h','-')} | {r.get('depth','-')} | "
            f"{r.get('dropout','-')} | {r.get('ffn_n_layers','-')} | {r.get('ffn_h','-')} | "
            f"{r.get('batch_size','-')} | {r.get('wallclock_s','-')}s | "
            f"{r.get('n_params','-'):,}" if isinstance(r.get('n_params'), int) else
            f"| {r['trial']} | {r['value']:.3f} | {r.get('d_h','-')} | {r.get('depth','-')} | "
            f"{r.get('dropout','-')} | {r.get('ffn_n_layers','-')} | {r.get('ffn_h','-')} | "
            f"{r.get('batch_size','-')} | {r.get('wallclock_s','-')}s | - |"
        )
    pruned_or_failed = [r for r in trial_rows if r["state"] != "COMPLETE"]
    if pruned_or_failed:
        lines.append("")
        lines.append(f"## Pruned / failed trials: {len(pruned_or_failed)}")
        for r in pruned_or_failed:
            lines.append(f"  - trial {r['trial']}: {r['state']}")
    OUT_MD.write_text("\n".join(lines))
    print(f"[OUT] {OUT_MD}")


if __name__ == "__main__":
    main()
