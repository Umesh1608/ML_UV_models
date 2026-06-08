#!/usr/bin/env python3
"""Ensemble predict wetlab λmax with Hybrid (Chemprop + PaiNN features) checkpoints."""
import argparse, json, os, glob, pickle, sys
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

from chemprop import data, featurizers
from chemprop.models import multi
import lightning.pytorch as pl

from paper1_new_cl.models import compute_metrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WETLAB_PATH = os.path.join(SCRIPT_DIR, "data", "wetlab_experimental.csv")
PAINN_FEAT_PATH = "/home/umesh/smallmol_platform/data/painn_cache/painn_features.pkl"


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="f4")
    ap.add_argument("--train-csv", default="data/filtered_uv/f4_uvab_sunscreen.csv")
    args = ap.parse_args()

    wet = pd.read_csv(WETLAB_PATH)
    wet["canon_smi"] = wet["smiles"].map(canon)
    wet["canon_solvent"] = wet["solvent_smiles"].map(canon)
    print(f"Wetlab: {len(wet)} pairs")

    # Load PaiNN features
    with open(PAINN_FEAT_PATH, "rb") as f:
        painn = pickle.load(f)
    missing = [s for s in wet["canon_smi"].unique() if painn.get(s) is None]
    if missing:
        print(f"MISSING PaiNN features: {missing}")

    # Recompute training set X_d mean/std (needed to normalize wetlab X_d)
    train_df = pd.read_csv(os.path.join(SCRIPT_DIR, args.train_csv))
    train_df["canon_smi"] = train_df["smiles"].map(canon)
    train_df = train_df[train_df["canon_smi"].map(lambda s: painn.get(s) is not None)]
    train_solutes = train_df["canon_smi"].drop_duplicates().values
    xd_train = np.stack([painn[s] for s in train_solutes])
    xd_mean = xd_train.mean(axis=0)
    xd_std = xd_train.std(axis=0) + 1e-6
    print(f"Train X_d stats: mean_range=[{xd_mean.min():.2f},{xd_mean.max():.2f}]")

    # Build wetlab datapoints with normalized X_d
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    wet_ok_idx = [i for i in range(len(wet)) if painn.get(wet["canon_smi"].iloc[i]) is not None]
    print(f"Wetlab with PaiNN features: {len(wet_ok_idx)}/{len(wet)}")
    solute_dps, solvent_dps = [], []
    for i in wet_ok_idx:
        row = wet.iloc[i]
        x_d = (painn[row["canon_smi"]] - xd_mean) / xd_std
        solute_dps.append(data.MoleculeDatapoint.from_smi(
            row["canon_smi"], y=np.array([0.0]), x_d=x_d.astype(np.float32)))
        solvent_dps.append(data.MoleculeDatapoint.from_smi(row["canon_solvent"]))
    mc = data.MulticomponentDataset([
        data.MoleculeDataset(solute_dps, feat),
        data.MoleculeDataset(solvent_dps, feat),
    ])
    loader = data.build_dataloader(mc, batch_size=64, shuffle=False, num_workers=0)

    # Ensemble predict
    ckpts = sorted(glob.glob(os.path.join(SCRIPT_DIR, f"results/hybrid_{args.tag}/{args.tag}_fold*_best.ckpt")))
    print(f"Found {len(ckpts)} fold checkpoints")
    all_preds = []
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False)
    for c in ckpts:
        model = multi.MulticomponentMPNN.load_from_checkpoint(c)
        with torch.no_grad():
            preds = trainer.predict(model, loader)
        all_preds.append(torch.cat(preds, dim=0).squeeze(-1).numpy())
        print(f"  {os.path.basename(c)}: pred mean {all_preds[-1].mean():.1f}")
    stacked = np.stack(all_preds)
    y_pred = stacked.mean(axis=0)
    y_std = stacked.std(axis=0)
    y_true = wet["lambda_max_exp"].values[wet_ok_idx].astype(float)

    m = compute_metrics(y_true, y_pred)
    print(f"\n=== HYBRID (Chemprop + PaiNN features) WETLAB RESULTS ===")
    print(f"n = {len(y_true)}")
    print(f"MAE  = {m['MAE']:.2f} nm")
    print(f"RMSE = {m['RMSE']:.2f} nm")
    print(f"R²   = {m['R2']:.3f}")
    print(f"r    = {m['Pearson_r']:.3f}")

    print(f"\n{'molecule':<22s} {'solv':<6s}  exp   pred (±σ)   err")
    for k, i in enumerate(wet_ok_idx):
        print(f"{wet['molecule'].iloc[i]:<22s} {wet['solvent_name'].iloc[i]:<6s}  "
              f"{y_true[k]:>3.0f}   {y_pred[k]:>5.1f} (±{y_std[k]:>4.1f})   {y_pred[k]-y_true[k]:>+6.1f}")

    out = {"model": f"hybrid_{args.tag}", "n": int(len(y_true)),
           **{k: float(v) for k, v in m.items()},
           "predictions": y_pred.tolist(), "std": y_std.tolist()}
    out_path = os.path.join(SCRIPT_DIR, f"results/hybrid_{args.tag}/wetlab_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
