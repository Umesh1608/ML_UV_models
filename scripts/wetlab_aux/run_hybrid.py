#!/usr/bin/env python3
"""Hybrid D-MPNN + PaiNN features training.

Architecture:
  Solute SMILES   → Chemprop BondMessagePassing (d_h=300, depth=3) → [B, 300]
  Solvent SMILES  → Chemprop BondMessagePassing (d_h=200, depth=2) → [B, 200]
  PaiNN features  → (no network) attached as solute X_d (53 dims)
  All three concat → RegressionFFN → λmax (nm)

Uses chemprop's X_d machinery to inject PaiNN features directly into FFN input.
"""
import argparse, gc, json, os, pickle, sys, time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

torch.set_float32_matmul_precision("medium")

from chemprop import data, featurizers, nn
from chemprop.models import multi
from chemprop.nn.metrics import RMSE as ChempropRMSE, MAE as ChempropMAE
from chemprop.nn.transforms import UnscaleTransform
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from paper1_new_cl.models import compute_metrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAINN_FEAT_PATH = "/home/umesh/smallmol_platform/data/painn_cache/painn_features.pkl"
SEED = 7
N_FOLDS = 5
MAX_EPOCHS = 100
PATIENCE = 15
WARMUP_EPOCHS = 2
INIT_LR = 1e-4
MAX_LR = 1e-3
FINAL_LR = 1e-4
BATCH_SIZE = 64
SOLUTE_D = 300
SOLUTE_DEPTH = 3
SOLVENT_D = 200
SOLVENT_DEPTH = 2
PAINN_DIM = 53


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def load_data(csv_path, painn_feats):
    from rdkit import Chem
    df = pd.read_csv(csv_path).dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    df["canon_smi"] = df["smiles"].map(canon)
    df["canon_solvent"] = df["solvent_smiles"].map(canon)
    # Keep only molecules with PaiNN features
    df = df[df["canon_smi"].map(lambda s: painn_feats.get(s) is not None)].reset_index(drop=True)
    # Also drop invalid solute/solvent SMILES
    valid = np.array([
        Chem.MolFromSmiles(s) is not None and Chem.MolFromSmiles(v) is not None
        for s, v in zip(df["canon_smi"], df["canon_solvent"])
    ])
    df = df[valid].reset_index(drop=True)
    print(f"[DATA] {len(df):,} pairs with PaiNN features ({df['canon_smi'].nunique():,} solutes, "
          f"{df['canon_solvent'].nunique()} solvents)")
    print(f"  λmax: {df['lambda_max'].min():.0f}-{df['lambda_max'].max():.0f} nm  "
          f"(mean {df['lambda_max'].mean():.0f})")
    return df


def compute_xd_scaler(df, painn_feats, idx):
    """Compute mean/std of PaiNN features over the training solute set (dedup by molecule)."""
    unique_sol = df.iloc[idx]["canon_smi"].drop_duplicates()
    feats = np.stack([painn_feats[s] for s in unique_sol])
    return feats.mean(axis=0), feats.std(axis=0) + 1e-6


def make_dps(df, painn_feats, idx, xd_mean, xd_std):
    """Build solute + solvent Chemprop datapoints with normalized X_d on solute."""
    solute_dps, solvent_dps = [], []
    for i in idx:
        row = df.iloc[i]
        x_d = (painn_feats[row["canon_smi"]] - xd_mean) / xd_std
        solute_dps.append(data.MoleculeDatapoint.from_smi(
            row["canon_smi"], y=np.array([row["lambda_max"]]),
            x_d=x_d.astype(np.float32)))
        solvent_dps.append(data.MoleculeDatapoint.from_smi(row["canon_solvent"]))
    return solute_dps, solvent_dps


def build_loaders(sol_tr, slv_tr, sol_vl, slv_vl, sol_te, slv_te):
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr = data.MulticomponentDataset([data.MoleculeDataset(sol_tr, feat), data.MoleculeDataset(slv_tr, feat)])
    vl = data.MulticomponentDataset([data.MoleculeDataset(sol_vl, feat), data.MoleculeDataset(slv_vl, feat)])
    te = data.MulticomponentDataset([data.MoleculeDataset(sol_te, feat), data.MoleculeDataset(slv_te, feat)])
    y_scaler = tr.normalize_targets()
    vl.normalize_targets(y_scaler)
    return (
        data.build_dataloader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        data.build_dataloader(vl, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        data.build_dataloader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        y_scaler,
    )


def build_model(y_scaler):
    agg = nn.MeanAggregation()
    output_transform = UnscaleTransform.from_standard_scaler(y_scaler)

    solute_block = nn.BondMessagePassing(d_h=SOLUTE_D, depth=SOLUTE_DEPTH)
    solvent_block = nn.BondMessagePassing(d_h=SOLVENT_D, depth=SOLVENT_DEPTH)
    mcmp = nn.MulticomponentMessagePassing(blocks=[solute_block, solvent_block], n_components=2)

    # FFN input: concat of solute D-MPNN + solvent D-MPNN + PaiNN X_d (pre-normalized)
    ffn_input_dim = mcmp.output_dim + PAINN_DIM
    ffn = nn.RegressionFFN(input_dim=ffn_input_dim, output_transform=output_transform)

    model = multi.MulticomponentMPNN(
        mcmp, agg, ffn,
        metrics=[ChempropRMSE(), ChempropMAE()],
        warmup_epochs=WARMUP_EPOCHS, init_lr=INIT_LR, max_lr=MAX_LR, final_lr=FINAL_LR,
    )
    return model


class Progress(pl.Callback):
    def __init__(self, tag, report_every=5):
        super().__init__(); self.tag = tag; self.report_every = report_every
        self.t0 = None; self.times = []; self.epoch_t0 = None
    def on_train_start(self, t, p): self.t0 = time.time()
    def on_train_epoch_start(self, t, p): self.epoch_t0 = time.time()
    def on_train_epoch_end(self, t, p):
        if self.epoch_t0: self.times.append(time.time() - self.epoch_t0)
    def on_validation_end(self, t, p):
        ep = t.current_epoch
        if (ep + 1) % self.report_every == 0 or ep == 0:
            tl = t.callback_metrics.get("train_loss")
            vl = t.callback_metrics.get("val_loss")
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
            print(f"  [{self.tag}] ep {ep+1}/{MAX_EPOCHS}  tr={tl_s}  vl={vl_s}  best={bv_s}  "
                  f"pat={patience}/{PATIENCE}  {avg:.1f}s/ep", flush=True)


def train_fold(fold_idx, df, painn_feats, tr_idx, vl_idx, te_idx, out_dir, tag):
    name = f"{tag}_fold{fold_idx}"
    mpath = os.path.join(out_dir, f"{name}_metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f: m = json.load(f)
        print(f"  [SKIP] fold {fold_idx}  RMSE={m['RMSE']:.2f}")
        return m
    print(f"\n{'='*70}\n  {tag}  fold {fold_idx}/{N_FOLDS-1}  "
          f"tr={len(tr_idx)}  vl={len(vl_idx)}  te={len(te_idx)}\n{'='*70}")

    xd_mean, xd_std = compute_xd_scaler(df, painn_feats, tr_idx)
    sol_tr, slv_tr = make_dps(df, painn_feats, tr_idx, xd_mean, xd_std)
    sol_vl, slv_vl = make_dps(df, painn_feats, vl_idx, xd_mean, xd_std)
    sol_te, slv_te = make_dps(df, painn_feats, te_idx, xd_mean, xd_std)
    tr_ld, vl_ld, te_ld, y_scaler = build_loaders(
        sol_tr, slv_tr, sol_vl, slv_vl, sol_te, slv_te)

    model = build_model(y_scaler)
    if fold_idx == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    best = multi.MulticomponentMPNN.load_from_checkpoint(ckpt.best_model_path) if ckpt.best_model_path else model
    preds = trainer.predict(best, te_ld)
    y_pred = torch.cat(preds, dim=0).squeeze(-1).numpy()
    y_te = df.iloc[te_idx]["lambda_max"].values.astype(float)
    m = compute_metrics(y_te, y_pred)
    print(f"  >> RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  R²={m['R2']:.4f}  r={m['Pearson_r']:.4f}")
    with open(mpath, "w") as f: json.dump(m, f, indent=2)
    np.save(os.path.join(out_dir, f"{name}_predictions.npy"), y_pred)
    np.save(os.path.join(out_dir, f"{name}_y_test.npy"), y_te)
    del model, best, trainer; torch.cuda.empty_cache(); gc.collect()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()

    out_dir = os.path.join(SCRIPT_DIR, "results", f"hybrid_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading PaiNN features from {PAINN_FEAT_PATH}...")
    with open(PAINN_FEAT_PATH, "rb") as f:
        painn_feats = pickle.load(f)
    print(f"  {sum(1 for v in painn_feats.values() if v is not None):,} molecules with features")

    df = load_data(args.data_path, painn_feats)
    pl.seed_everything(SEED, workers=True)

    # Solute-stratified 5-fold CV
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["canon_smi"].values
    idx_all = np.arange(len(df))
    fold_splits = []
    for tr_val_idx, te_idx in gkf.split(idx_all, groups=groups):
        g2 = groups[tr_val_idx]
        gkf2 = GroupKFold(n_splits=9)
        tr_sub, vl_sub = next(gkf2.split(tr_val_idx, groups=g2))
        fold_splits.append((tr_val_idx[tr_sub], tr_val_idx[vl_sub], te_idx))

    fold_range = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    metrics = []
    for i in fold_range:
        tr, vl, te = fold_splits[i]
        m = train_fold(i, df, painn_feats, tr, vl, te, out_dir, args.tag)
        metrics.append(m)
        if i != fold_range[-1]:
            torch.cuda.empty_cache(); gc.collect(); time.sleep(3)

    if len(metrics) == N_FOLDS:
        agg = {"n_folds": N_FOLDS, "tag": args.tag, "data_path": args.data_path}
        print(f"\n{'='*70}\n  AGGREGATE  ({args.tag}, {N_FOLDS}-fold CV)\n{'='*70}")
        for k in ["RMSE", "MAE", "R2", "Pearson_r"]:
            vals = [m[k] for m in metrics]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
            agg[f"{k}_values"] = vals
            print(f"  {k}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        with open(os.path.join(out_dir, f"{args.tag}_cv_aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()
