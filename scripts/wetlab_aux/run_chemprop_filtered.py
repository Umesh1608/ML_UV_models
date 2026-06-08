#!/usr/bin/env python3
"""5-fold CV Chemprop D-MPNN on filtered UV datasets (F1, F4).

Mirrors run_chemprop.py but takes --data-path and does stratified K-fold CV on
the provided CSV (columns: smiles, solvent_smiles, lambda_max).

Usage:
  python3 run_chemprop_filtered.py --data-path data/filtered_uv/f1_uvab.csv --tag f1
  python3 run_chemprop_filtered.py --data-path data/filtered_uv/f4_uvab_sunscreen.csv --tag f4
"""

import argparse
import gc
import json
import os
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, GroupKFold

torch.set_float32_matmul_precision("medium")

from chemprop import data, featurizers, nn
from chemprop.models import multi
from chemprop.nn.metrics import RMSE as ChempropRMSE, MAE as ChempropMAE
from chemprop.nn.transforms import UnscaleTransform
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from paper1_new_cl.models import compute_metrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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


def load_data(path):
    from rdkit import Chem
    df = pd.read_csv(path).dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    valid = np.array([
        Chem.MolFromSmiles(s) is not None and Chem.MolFromSmiles(v) is not None
        for s, v in zip(df["smiles"], df["solvent_smiles"])
    ])
    df = df[valid].reset_index(drop=True)
    print(f"[DATA] {len(df):,} pairs  ({df['smiles'].nunique():,} solutes, "
          f"{df['solvent_smiles'].nunique()} solvents)")
    print(f"  λmax: {df['lambda_max'].min():.0f}–{df['lambda_max'].max():.0f} nm "
          f"(mean {df['lambda_max'].mean():.0f})")
    return df["smiles"].values, df["solvent_smiles"].values, df["lambda_max"].values.astype(np.float64)


def make_dps(solutes, solvents, y, idx):
    solute_dps = [data.MoleculeDatapoint.from_smi(solutes[i], y=np.array([y[i]])) for i in idx]
    solvent_dps = [data.MoleculeDatapoint.from_smi(solvents[i]) for i in idx]
    return solute_dps, solvent_dps


def build_loaders(sol_tr, slv_tr, sol_vl, slv_vl, sol_te, slv_te):
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr = data.MulticomponentDataset([data.MoleculeDataset(sol_tr, feat), data.MoleculeDataset(slv_tr, feat)])
    vl = data.MulticomponentDataset([data.MoleculeDataset(sol_vl, feat), data.MoleculeDataset(slv_vl, feat)])
    te = data.MulticomponentDataset([data.MoleculeDataset(sol_te, feat), data.MoleculeDataset(slv_te, feat)])
    scaler = tr.normalize_targets()
    vl.normalize_targets(scaler)
    return (
        data.build_dataloader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        data.build_dataloader(vl, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        data.build_dataloader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        scaler,
    )


def build_model(scaler):
    agg = nn.MeanAggregation()
    output_transform = UnscaleTransform.from_standard_scaler(scaler)
    mcmp = nn.MulticomponentMessagePassing(
        blocks=[nn.BondMessagePassing(d_h=D_H, depth=DEPTH) for _ in range(2)],
        n_components=2,
    )
    ffn = nn.RegressionFFN(input_dim=mcmp.output_dim, output_transform=output_transform)
    model = multi.MulticomponentMPNN(
        mcmp, agg, ffn,
        metrics=[ChempropRMSE(), ChempropMAE()],
        warmup_epochs=WARMUP_EPOCHS, init_lr=INIT_LR, max_lr=MAX_LR, final_lr=FINAL_LR,
    )
    return model


class Progress(pl.Callback):
    def __init__(self, tag, report_every=5):
        super().__init__(); self.tag = tag; self.report_every = report_every
        self.t0 = None; self.epoch_t0 = None; self.times = []
    def on_train_start(self, t, p): self.t0 = time.time()
    def on_train_epoch_start(self, t, p): self.epoch_t0 = time.time()
    def on_train_epoch_end(self, t, p):
        if self.epoch_t0: self.times.append(time.time() - self.epoch_t0)
    def on_validation_end(self, t, p):
        ep = t.current_epoch
        tl = t.callback_metrics.get("train_loss")
        vl = t.callback_metrics.get("val_loss")
        if (ep + 1) % self.report_every == 0 or ep == 0:
            avg = np.mean(self.times[-self.report_every:]) if self.times else 0
            patience = 0; best = float("inf")
            for cb in t.callbacks:
                if isinstance(cb, EarlyStopping):
                    patience = cb.wait_count
                    best = float(cb.best_score) if cb.best_score is not None else float("inf")
                    break
            tl_s = f"{float(tl):.2f}" if tl is not None else "N/A"
            vl_s = f"{float(vl):.2f}" if vl is not None else "N/A"
            bv_s = f"{best:.2f}" if best < float("inf") else "N/A"
            print(f"  [{self.tag}] ep {ep+1}/{MAX_EPOCHS}  tr={tl_s}  vl={vl_s}  "
                  f"best={bv_s}  pat={patience}/{PATIENCE}  {avg:.1f}s/ep", flush=True)


def scaffold_of(smi):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smi)
    if m is None: return smi
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        csmi = Chem.MolToSmiles(s)
        return csmi if csmi else smi
    except Exception:
        return smi


def build_fold_splits(solutes, y, mode, n_folds, seed):
    """Return list of (train_idx, val_idx, test_idx) for each fold."""
    idx_all = np.arange(len(y))
    if mode == "random":
        kf_outer = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_outer = list(kf_outer.split(idx_all))
        folds = []
        for tr_val_idx, te_idx in splits_outer:
            kf_inner = KFold(n_splits=9, shuffle=True, random_state=seed)
            tr_sub, vl_sub = next(kf_inner.split(tr_val_idx))
            folds.append((tr_val_idx[tr_sub], tr_val_idx[vl_sub], te_idx))
        return folds
    if mode == "solute":
        groups = np.array(solutes)
    elif mode == "scaffold":
        print("  Computing Bemis-Murcko scaffolds...")
        uniq = {s: scaffold_of(s) for s in set(solutes)}
        groups = np.array([uniq[s] for s in solutes])
        n_scaffolds = len(set(groups.tolist()))
        print(f"  {n_scaffolds} unique scaffolds across {len(set(solutes))} solutes")
    else:
        raise ValueError(f"unknown split mode: {mode}")
    gkf_outer = GroupKFold(n_splits=n_folds)
    folds = []
    for tr_val_idx, te_idx in gkf_outer.split(idx_all, groups=groups):
        tr_val_groups = groups[tr_val_idx]
        gkf_inner = GroupKFold(n_splits=9)
        tr_sub, vl_sub = next(gkf_inner.split(tr_val_idx, groups=tr_val_groups))
        folds.append((tr_val_idx[tr_sub], tr_val_idx[vl_sub], te_idx))
    return folds


def train_fold(fold_idx, sol, slv, y, tr_idx, vl_idx, te_idx, out_dir, tag):
    name = f"{tag}_fold{fold_idx}"
    mpath = os.path.join(out_dir, f"{name}_metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f: m = json.load(f)
        print(f"  [SKIP] fold {fold_idx}  RMSE={m['RMSE']:.2f}")
        return m
    print(f"\n{'='*70}\n  {tag}  fold {fold_idx}/{N_FOLDS-1}  "
          f"tr={len(tr_idx)}  vl={len(vl_idx)}  te={len(te_idx)}\n{'='*70}")
    sol_tr, slv_tr = make_dps(sol, slv, y, tr_idx)
    sol_vl, slv_vl = make_dps(sol, slv, y, vl_idx)
    sol_te, slv_te = make_dps(sol, slv, y, te_idx)
    tr_ld, vl_ld, te_ld, scaler = build_loaders(sol_tr, slv_tr, sol_vl, slv_vl, sol_te, slv_te)
    model = build_model(scaler)
    if fold_idx == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
    ckpt = ModelCheckpoint(dirpath=out_dir, filename=f"{name}_best",
                           monitor="val_loss", save_top_k=1, mode="min")
    early = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", min_delta=1e-5)
    prog = Progress(tag=name, report_every=5)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, accelerator="auto", devices=1,
        callbacks=[ckpt, early, prog],
        enable_progress_bar=False, logger=False, enable_model_summary=False,
    )
    t0 = time.time()
    trainer.fit(model, tr_ld, vl_ld)
    print(f"  [{name}] trained in {time.time()-t0:.0f}s ({trainer.current_epoch+1} ep)")
    best_model = multi.MulticomponentMPNN.load_from_checkpoint(ckpt.best_model_path) if ckpt.best_model_path else model
    preds = trainer.predict(best_model, te_ld)
    y_pred = torch.cat(preds, dim=0).squeeze(-1).numpy()
    y_te = y[te_idx]
    m = compute_metrics(y_te, y_pred)
    print(f"  >> RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  R²={m['R2']:.4f}  r={m['Pearson_r']:.4f}")
    with open(mpath, "w") as f: json.dump(m, f, indent=2)
    np.save(os.path.join(out_dir, f"{name}_predictions.npy"), y_pred)
    np.save(os.path.join(out_dir, f"{name}_y_test.npy"), y_te)
    del model, best_model, trainer; torch.cuda.empty_cache(); gc.collect()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--tag", required=True, help="prefix for output files (e.g. f1, f4)")
    ap.add_argument("--fold", type=int, default=None, help="single fold to run")
    ap.add_argument("--split-mode", choices=["random", "solute", "scaffold"],
                    default="solute", help="cross-validation grouping")
    args = ap.parse_args()

    run_tag = f"{args.tag}_{args.split_mode}"
    out_dir = os.path.join(SCRIPT_DIR, "results", f"chemprop_{run_tag}")
    os.makedirs(out_dir, exist_ok=True)
    sol, slv, y = load_data(args.data_path)
    pl.seed_everything(SEED, workers=True)
    print(f"  split mode: {args.split_mode}")
    fold_splits = build_fold_splits(sol, y, args.split_mode, N_FOLDS, SEED)

    fold_range = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    metrics = []
    for i in fold_range:
        tr, vl, te = fold_splits[i]
        m = train_fold(i, sol, slv, y, tr, vl, te, out_dir, run_tag)
        metrics.append(m)
        if i != fold_range[-1]:
            torch.cuda.empty_cache(); gc.collect(); time.sleep(5)
    if len(metrics) == N_FOLDS:
        agg = {"n_folds": N_FOLDS, "tag": run_tag, "data_path": args.data_path,
               "split_mode": args.split_mode}
        print(f"\n{'='*70}\n  AGGREGATE  ({run_tag}, {N_FOLDS}-fold CV)\n{'='*70}")
        for k in ["RMSE", "MAE", "R2", "Pearson_r"]:
            vals = [m[k] for m in metrics]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
            agg[f"{k}_values"] = vals
            print(f"  {k}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        with open(os.path.join(out_dir, f"{run_tag}_cv_aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()
