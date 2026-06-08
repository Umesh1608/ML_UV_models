#!/usr/bin/env python3
"""Train HybridSpectrum on F1 with 5-fold solute-stratified CV.

HybridSpectrum = Chemprop solute D-MPNN + Chemprop solvent D-MPNN +
                 per-state (Δε_k, Δf_k) correction heads over PaiNN-SCA gas predictions.

Output: solvent-corrected 50-state spectrum, λmax derived via Gaussian broadening.
"""
import argparse, gc, json, os, pickle, sys, time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

torch.set_float32_matmul_precision("medium")

from chemprop import nn as chemprop_nn, data as chemprop_data, featurizers

sys.path.insert(0, "/home/umesh/smallmol_platform/src")
from uv_predict.models.hybrid_spectrum import HybridSpectrum

PAINN_CACHE = "/home/umesh/smallmol_platform/data/painn_cache/f12_solutes.pkl"
SEED = 7
N_FOLDS = 5
N_STATES = 50
MAX_EPOCHS = 100
PATIENCE = 15
LR = 5e-4
WD = 1e-5
BATCH_SIZE = 64
SOLUTE_D = 256
SOLUTE_DEPTH = 3
SOLVENT_D = 128
SOLVENT_DEPTH = 2
DELTA_REG = 0.02   # regularization on ||Δε||² + ||Δf||²

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


# ─── Chemprop-encoder wrapper ──────────────────────────────────────────────────
class ChempropEncoder(nn.Module):
    """Wraps BondMessagePassing + MeanAggregation into a one-call encoder."""
    def __init__(self, d_h=300, depth=3):
        super().__init__()
        self.mp = chemprop_nn.BondMessagePassing(d_h=d_h, depth=depth)
        self.agg = chemprop_nn.MeanAggregation()
        self.output_dim = self.mp.output_dim

    def forward(self, bmg):
        H = self.mp(bmg)  # atom-level [n_total_atoms, d_h]
        return self.agg(H, bmg.batch)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class SpectrumDataset(torch.utils.data.Dataset):
    """Stores (solute MolGraph, solvent MolGraph, E_gas, f_gas, y_nm) per row."""
    def __init__(self, df, painn, featurizer):
        self.df = df.reset_index(drop=True)
        self.painn = painn
        # Pre-featurize for speed (small datasets)
        self.solute_mgs = [featurizer(Chem.MolFromSmiles(s)) for s in self.df["canon_smi"]]
        self.solvent_mgs = [featurizer(Chem.MolFromSmiles(s)) for s in self.df["canon_solvent"]]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = self.painn[row["canon_smi"]]
        return {
            "solute_mg": self.solute_mgs[idx],
            "solvent_mg": self.solvent_mgs[idx],
            "E_gas": torch.from_numpy(p["E"]).float(),
            "f_gas": torch.from_numpy(p["f"]).float(),
            "y_nm": float(row["lambda_max"]),
        }


def collate(batch):
    solute_bmg = chemprop_data.BatchMolGraph([b["solute_mg"] for b in batch])
    solvent_bmg = chemprop_data.BatchMolGraph([b["solvent_mg"] for b in batch])
    E_gas = torch.stack([b["E_gas"] for b in batch])
    f_gas = torch.stack([b["f_gas"] for b in batch])
    y_nm = torch.tensor([b["y_nm"] for b in batch], dtype=torch.float32)
    return solute_bmg, solvent_bmg, E_gas, f_gas, y_nm


def is_clean(v):
    if v is None: return False
    E = v["E"]; ff = v["f"]; g = v["gap"]
    if not (np.all(np.isfinite(E)) and np.all(np.isfinite(ff)) and np.isfinite(g)):
        return False
    if (E < 0.1).any() or (E > 50).any(): return False
    if (ff < 0).any() or (ff > 100).any(): return False
    if abs(g) > 50: return False
    return True


# ─── Training ─────────────────────────────────────────────────────────────────
def train_fold(fold_idx, df, painn, tr_idx, vl_idx, te_idx, out_dir, tag):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"{tag}_fold{fold_idx}"
    mpath = os.path.join(out_dir, f"{name}_metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f: m = json.load(f)
        print(f"  [SKIP] fold {fold_idx}  MAE={m['MAE']:.2f}")
        return m

    print(f"\n{'='*70}\n  {tag}  fold {fold_idx}/{N_FOLDS-1}  "
          f"tr={len(tr_idx)}  vl={len(vl_idx)}  te={len(te_idx)}\n{'='*70}")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_ds = SpectrumDataset(df.iloc[tr_idx], painn, featurizer)
    vl_ds = SpectrumDataset(df.iloc[vl_idx], painn, featurizer)
    te_ds = SpectrumDataset(df.iloc[te_idx], painn, featurizer)

    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)
    vl_ld = DataLoader(vl_ds, batch_size=256, shuffle=False, collate_fn=collate, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False, collate_fn=collate, num_workers=0)

    solute_encoder = ChempropEncoder(d_h=SOLUTE_D, depth=SOLUTE_DEPTH)
    solvent_encoder = ChempropEncoder(d_h=SOLVENT_D, depth=SOLVENT_DEPTH)
    model = HybridSpectrum(solute_encoder, solvent_encoder,
                           d_solute=SOLUTE_D, d_solvent=SOLVENT_D, n_states=N_STATES).to(device)
    if fold_idx == 0:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  HybridSpectrum params: {n:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR*0.1)

    def step(batch):
        solute_bmg, solvent_bmg, E_gas, f_gas, y_nm = batch
        # BatchMolGraph.to() is in-place and returns None — don't reassign
        solute_bmg.to(device)
        solvent_bmg.to(device)
        E_gas = E_gas.to(device); f_gas = f_gas.to(device); y_nm = y_nm.to(device)
        out = model(E_gas, f_gas, solute_bmg, solvent_bmg)
        lam = out["lambda_max"]
        loss_mae = torch.nn.functional.l1_loss(lam, y_nm)
        loss_reg = (out["delta_e"] ** 2).mean() + 0.1 * (out["delta_f"] ** 2).mean()
        loss = loss_mae + DELTA_REG * loss_reg
        return loss, loss_mae.item(), lam.detach(), y_nm.detach()

    best_val = float("inf"); best_state = None; patience = 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        tr_maes = []
        for batch in tr_ld:
            opt.zero_grad()
            loss, mae, _, _ = step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_maes.append(mae)
        sched.step()

        model.eval()
        vl_preds, vl_y = [], []
        with torch.no_grad():
            for batch in vl_ld:
                _, _, pred, y = step(batch)
                vl_preds.append(pred.cpu()); vl_y.append(y.cpu())
        vl_preds = torch.cat(vl_preds); vl_y = torch.cat(vl_y)
        vl_mae = (vl_preds - vl_y).abs().mean().item()

        if vl_mae < best_val - 1e-3:
            best_val = vl_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch % 5 == 0 or patience >= PATIENCE:
            print(f"  [{name}] ep {epoch:3d}  tr_MAE={np.mean(tr_maes):.2f}  vl_MAE={vl_mae:.2f}  "
                  f"best={best_val:.2f}  pat={patience}/{PATIENCE}", flush=True)
        if patience >= PATIENCE:
            break

    print(f"  [{name}] trained {epoch+1} ep in {time.time()-t0:.0f}s  best_val={best_val:.2f}nm")

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    te_preds, te_y = [], []
    with torch.no_grad():
        for batch in te_ld:
            _, _, pred, y = step(batch)
            te_preds.append(pred.cpu()); te_y.append(y.cpu())
    te_preds = torch.cat(te_preds).numpy(); te_y = torch.cat(te_y).numpy()
    err = te_preds - te_y
    mae = float(np.mean(np.abs(err))); rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = np.sum(err**2); ss_tot = np.sum((te_y - te_y.mean())**2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-9))
    from scipy.stats import pearsonr
    r = float(pearsonr(te_y, te_preds)[0])
    m = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4), "Pearson_r": round(r, 4)}
    print(f"  >> {name}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  r={r:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(mpath, "w") as f: json.dump(m, f, indent=2)
    torch.save(best_state, os.path.join(out_dir, f"{name}_best.pt"))
    np.save(os.path.join(out_dir, f"{name}_predictions.npy"), te_preds)
    np.save(os.path.join(out_dir, f"{name}_y_test.npy"), te_y)
    del model, solute_encoder, solvent_encoder; torch.cuda.empty_cache(); gc.collect()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()

    out_dir = os.path.join(SCRIPT_DIR, "results", f"hybrid_spectrum_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    with open(PAINN_CACHE, "rb") as f: painn = pickle.load(f)
    clean = {k: v for k, v in painn.items() if is_clean(v)}
    print(f"PaiNN clean cache: {len(clean):,}/{len(painn):,}")

    df = pd.read_csv(args.data_path).dropna(subset=["smiles", "solvent_smiles", "lambda_max"])
    df["canon_smi"] = df["smiles"].map(canon)
    df["canon_solvent"] = df["solvent_smiles"].map(canon)
    df = df[df["canon_smi"].map(lambda s: s in clean)].reset_index(drop=True)
    print(f"[{args.tag}] pairs with clean PaiNN: {len(df):,} ({df['canon_smi'].nunique():,} solutes)")

    torch.manual_seed(SEED); np.random.seed(SEED)
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
        m = train_fold(i, df, clean, tr, vl, te, out_dir, args.tag)
        metrics.append(m)
        if i != fold_range[-1]:
            torch.cuda.empty_cache(); gc.collect(); time.sleep(3)

    if len(metrics) == N_FOLDS:
        agg = {"n_folds": N_FOLDS, "tag": args.tag}
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
