#!/usr/bin/env python3
"""Ensemble predict wetlab λmax using Chemprop checkpoints from F4/F1 solute CV.

Loads all 5 fold checkpoints, averages predictions, compares to experimental.
"""

import argparse
import json
import os
import glob

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from chemprop import data, featurizers
from chemprop.models import multi
import lightning.pytorch as pl

from paper1_new_cl.models import compute_metrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WETLAB_PATH = os.path.join(SCRIPT_DIR, "data", "wetlab_experimental.csv")

SEED = 7


def canonicalize(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi


def train_solute_set(data_path):
    """Return set of canonical SMILES used in training."""
    df = pd.read_csv(data_path).dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    return set(canonicalize(s) for s in df["smiles"].unique())


def ensemble_predict(checkpoints, solute_smis, solvent_smis):
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    solute_dps = [data.MoleculeDatapoint.from_smi(s, y=np.array([0.0])) for s in solute_smis]
    solvent_dps = [data.MoleculeDatapoint.from_smi(v) for v in solvent_smis]
    mc = data.MulticomponentDataset([
        data.MoleculeDataset(solute_dps, feat),
        data.MoleculeDataset(solvent_dps, feat),
    ])
    loader = data.build_dataloader(mc, batch_size=64, shuffle=False, num_workers=0)
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, enable_progress_bar=False)
    preds = []
    for ckpt in checkpoints:
        model = multi.MulticomponentMPNN.load_from_checkpoint(ckpt)
        with torch.no_grad():
            p = trainer.predict(model, loader)
        preds.append(torch.cat(p, dim=0).squeeze(-1).numpy())
    preds = np.stack(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["f4", "f1", "both"], default="both")
    args = ap.parse_args()

    wet = pd.read_csv(WETLAB_PATH)
    print(f"[WETLAB] {len(wet)} measurements, {wet['molecule'].nunique()} molecules, "
          f"{wet['solvent_smiles'].nunique()} solvents ({sorted(wet['solvent_smiles'].unique())})")

    targets = {"f4": ("results/chemprop_f4_solute/*fold*_best.ckpt",
                      "data/filtered_uv/f4_uvab_sunscreen.csv", "F4"),
               "f1": ("results/chemprop_f1_solute/*fold*_best.ckpt",
                      "data/filtered_uv/f1_uvab.csv", "F1")}
    models_to_run = ["f4", "f1"] if args.model == "both" else [args.model]

    results = {}
    for key in models_to_run:
        ckpt_pat, data_path, label = targets[key]
        ckpts = sorted(glob.glob(os.path.join(SCRIPT_DIR, ckpt_pat)))
        if not ckpts:
            print(f"[{label}] no checkpoints; skip")
            continue
        print(f"\n[{label}] {len(ckpts)} fold checkpoints")

        train_set = train_solute_set(os.path.join(SCRIPT_DIR, data_path))
        wet_canon = [canonicalize(s) for s in wet["smiles"].values]
        in_train = np.array([c in train_set for c in wet_canon])
        print(f"  wetlab molecules found in training set: {in_train.sum()}/{len(wet)} "
              f"(these are leaky — biased predictions)")

        y_pred, y_std = ensemble_predict(ckpts, wet["smiles"].values, wet["solvent_smiles"].values)
        y_true = wet["lambda_max_exp"].values.astype(float)

        # All predictions
        m_all = compute_metrics(y_true, y_pred)
        # Unseen molecules only (clean external)
        unseen_idx = ~in_train
        m_clean = (compute_metrics(y_true[unseen_idx], y_pred[unseen_idx])
                   if unseen_idx.sum() >= 2 else None)

        print(f"\n  [{label}] ALL (n={len(wet)}):   "
              f"MAE={m_all['MAE']:.2f}  RMSE={m_all['RMSE']:.2f}  "
              f"R²={m_all['R2']:.3f}  r={m_all['Pearson_r']:.3f}")
        if m_clean:
            print(f"  [{label}] UNSEEN (n={unseen_idx.sum()}): "
                  f"MAE={m_clean['MAE']:.2f}  RMSE={m_clean['RMSE']:.2f}  "
                  f"R²={m_clean['R2']:.3f}  r={m_clean['Pearson_r']:.3f}")

        # Per-molecule table
        print(f"\n  Per-sample predictions ({label}):")
        print(f"    {'molecule':<22s} {'solvent':<6s}  exp   pred (±σ)  err   seen?")
        for i in range(len(wet)):
            seen = "✓" if in_train[i] else "NEW"
            print(f"    {wet['molecule'].iloc[i]:<22s} "
                  f"{wet['solvent_name'].iloc[i]:<6s}  "
                  f"{y_true[i]:>3.0f}   {y_pred[i]:>5.1f} (±{y_std[i]:>4.1f})  "
                  f"{y_pred[i]-y_true[i]:>+6.1f}  {seen}")

        results[key] = {"label": label, "all": m_all, "unseen": m_clean,
                        "n_seen": int(in_train.sum()),
                        "predictions": y_pred.tolist(), "std": y_std.tolist()}

        out_path = os.path.join(SCRIPT_DIR, "results", f"wetlab_chemprop_{key}.json")
        with open(out_path, "w") as f:
            json.dump(results[key], f, indent=2)
        print(f"  saved: {out_path}")

    # Summary comparison
    if len(results) == 2:
        print(f"\n{'='*70}\n  WETLAB SUMMARY  (32 pairs, 16 mols × 2 solvents)\n{'='*70}")
        for key in ["f4", "f1"]:
            r = results[key]
            print(f"  {r['label']:<4s} ALL   MAE={r['all']['MAE']:<5.2f}  RMSE={r['all']['RMSE']:<5.2f}  R²={r['all']['R2']:.3f}")
            if r["unseen"]:
                print(f"  {r['label']:<4s} unseen MAE={r['unseen']['MAE']:<5.2f}  "
                      f"RMSE={r['unseen']['RMSE']:<5.2f}  R²={r['unseen']['R2']:.3f}  "
                      f"(n={32 - r['n_seen']})")


if __name__ == "__main__":
    main()
