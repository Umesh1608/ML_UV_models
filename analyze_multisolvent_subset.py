#!/usr/bin/env python3
"""
R1-14 Step 0 -- Multi-solvent subset analysis of the solvent ablation.

The reviewer asked whether the solvent-encoding ablation (with-solvent vs.
solute-only) should be reported on the *multi-solvent* subset only, because
chromophores that appear in train with only one solvent cannot teach the
model a solvent-specific delta.

This script restricts each fold's test set to records whose canonical solute
SMILES appears in the corresponding (train+val) slice with at least 2 distinct
solvent SMILES, and recomputes per-model RMSE/MAE on that subset. The same
fold/model artefacts are used as the main-text Table~\\ref{tab:solvent_ablation},
so the only change is the subsetting.

Data: v2 dataset (UV_canonical_full_dataset.csv after dropna, 18,755 records),
because the existing +Solvent / Solute-only paired runs were trained on v2.

Outputs:
  results/r1_14_multisolvent_subset.json   -- per-fold and aggregate metrics
  results/r1_14_summary.md                 -- human-readable summary
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results"
DATA_PATH = SCRIPT_DIR / "previous_code" / "UV_canonical_full_dataset.csv"
N_FOLDS = 5

# Paired model artefacts. Each entry is (model_label, +Solvent_prefix, Solute-only_prefix).
PAIRS = [
    ("RF (tuned)",  "rf_tuned",       "rf_tuned_nosolvent"),
    ("Chemprop",    "chemprop_v2",    "chemprop_v2_nosolvent"),
    ("BiGRU",       "bigru_solvent_v2", "bigru_nosolvent_v2"),
]


def rmse(e):
    return float(np.sqrt(np.mean(e**2)))


def main():
    print(f"  loading v2 dataset {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df[["canon", "solvents", "lambda_max"]].dropna().reset_index(drop=True)
    print(f"  v2 size after dropna: {len(df)}")

    n_total = len(df)
    all_indices = set(range(n_total))

    per_fold = []  # list of dicts
    for f in range(N_FOLDS):
        fold_record = {"fold": f, "models": {}}

        for label, sol_pfx, no_pfx in PAIRS:
            for tag, pfx in (("with_solvent", sol_pfx), ("solute_only", no_pfx)):
                p_path = RESULTS / f"{pfx}_fold{f}_predictions.npy"
                y_path = RESULTS / f"{pfx}_fold{f}_y_test.npy"
                t_path = RESULTS / f"{pfx}_fold{f}_test_indices.npy"
                if not (p_path.exists() and y_path.exists() and t_path.exists()):
                    print(f"    [skip] missing artefacts for {pfx} fold {f}")
                    continue
                preds   = np.load(p_path).flatten()
                y       = np.load(y_path).flatten()
                test_ix = np.load(t_path)
                if not (len(preds) == len(y) == len(test_ix)):
                    print(f"    [WARN] {pfx} fold {f} length mismatch "
                          f"(preds {len(preds)}, y {len(y)}, idx {len(test_ix)}) "
                          f"-- skipping")
                    continue

                # Compute multi-solvent train-set restricted to THIS model's
                # complement (since BiGRU and RF v2 splits differ by 1 record).
                train_val_idx = np.array(sorted(all_indices - set(test_ix.tolist())))
                tv_df = df.iloc[train_val_idx]
                n_solv_per_chrom = tv_df.groupby("canon")["solvents"].nunique()
                multi_solute_set = set(n_solv_per_chrom[n_solv_per_chrom >= 2].index)
                test_df = df.iloc[test_ix].reset_index(drop=True)
                multi_mask = test_df["canon"].isin(multi_solute_set).values

                err_full   = preds - y
                err_multi  = err_full[multi_mask]
                err_single = err_full[~multi_mask]
                fold_record["models"].setdefault(label, {})[tag] = {
                    "n_test_total":   int(len(test_ix)),
                    "n_test_multi":   int(multi_mask.sum()),
                    "n_test_single":  int((~multi_mask).sum()),
                    "frac_multi":     round(float(multi_mask.mean()), 4),
                    "full_RMSE":      round(rmse(err_full),                    3),
                    "full_MAE":       round(float(np.mean(np.abs(err_full))),  3),
                    "multi_RMSE":     round(rmse(err_multi),                   3),
                    "multi_MAE":      round(float(np.mean(np.abs(err_multi))), 3),
                    "single_RMSE":    round(rmse(err_single),                  3),
                    "single_MAE":     round(float(np.mean(np.abs(err_single))),3),
                    "n_unique_multi_chrom_in_train": int(len(multi_solute_set)),
                }
        per_fold.append(fold_record)
        ms_chunk = next(iter(fold_record["models"].values()))
        msa = ms_chunk["with_solvent"]
        print(f"  fold {f}: |test|={msa['n_test_total']:4d}  "
              f"|multi-test|={msa['n_test_multi']:4d} ({msa['frac_multi']:.1%})  "
              f"|single-or-novel test|={msa['n_test_single']:4d}  "
              f"|unique-multi-chrom in train|={msa['n_unique_multi_chrom_in_train']:4d}")

    # Aggregate across folds. frac_multi varies fold-to-fold but barely
    # varies model-to-model (only 1 record's difference between RF and BiGRU
    # splits on fold 2), so we average across folds using the first model's
    # frac_multi as representative.
    frac_multi_means = []
    for pf in per_fold:
        for label_data in pf["models"].values():
            if "with_solvent" in label_data:
                frac_multi_means.append(label_data["with_solvent"]["frac_multi"])
                break
    agg = {"dataset": str(DATA_PATH.name),
           "n_records": n_total,
           "n_folds_aggregated": len(per_fold),
           "frac_multi_solvent_test_mean": round(float(np.mean(frac_multi_means)), 4),
           "models": {}}
    for label, _, _ in PAIRS:
        agg["models"][label] = {}
        for tag in ("with_solvent", "solute_only"):
            for subset in ("full", "multi", "single"):
                rmses = [pf["models"].get(label, {}).get(tag, {}).get(f"{subset}_RMSE")
                         for pf in per_fold]
                maes  = [pf["models"].get(label, {}).get(tag, {}).get(f"{subset}_MAE")
                         for pf in per_fold]
                rmses = [r for r in rmses if r is not None]
                maes  = [m for m in maes  if m is not None]
                if not rmses:
                    continue
                agg["models"][label][f"{tag}_{subset}_RMSE_mean"] = round(float(np.mean(rmses)), 3)
                agg["models"][label][f"{tag}_{subset}_RMSE_std"]  = round(float(np.std(rmses)),  3)
                agg["models"][label][f"{tag}_{subset}_MAE_mean"]  = round(float(np.mean(maes)),  3)
                agg["models"][label][f"{tag}_{subset}_MAE_std"]   = round(float(np.std(maes)),   3)
        # Solvent-encoding gain (with - solute_only) on each subset
        for subset in ("full", "multi", "single"):
            ws = agg["models"][label].get(f"with_solvent_{subset}_RMSE_mean")
            so = agg["models"][label].get(f"solute_only_{subset}_RMSE_mean")
            if ws is not None and so is not None:
                agg["models"][label][f"{subset}_solvent_gain_RMSE"] = round(so - ws, 3)
                agg["models"][label][f"{subset}_solvent_gain_pct"]  = round((so - ws) / so * 100, 2)

    out = {"per_fold": per_fold, "aggregate": agg}
    (RESULTS / "r1_14_multisolvent_subset.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote results/r1_14_multisolvent_subset.json")

    # --- Markdown summary ---------------------------------------------------
    md = []
    md.append("# R1-14 Step 0: Solvent Ablation on Multi-Solvent Subset Only")
    md.append("")
    md.append(f"- Dataset: v2 (`{DATA_PATH.name}`), {n_total} records after dropna")
    md.append(f"- Folds aggregated: {len(per_fold)} / {N_FOLDS}")
    md.append(f"- Fraction of test records whose chromophore is multi-solvent in train: "
              f"mean = {agg['frac_multi_solvent_test_mean']:.1%}")
    md.append("")
    md.append("## Aggregate Results (5-fold mean ± std)")
    md.append("")
    md.append("| Model | Solvent | Full RMSE | Multi-solv subset RMSE | Single-solv/novel subset RMSE | Δ Full | Δ Multi | Δ Single |")
    md.append("|---|---|---|---|---|---|---|---|")
    for label, _, _ in PAIRS:
        m = agg["models"][label]
        full_w = f"{m.get('with_solvent_full_RMSE_mean','?')} ± {m.get('with_solvent_full_RMSE_std','?')}"
        full_s = f"{m.get('solute_only_full_RMSE_mean','?')} ± {m.get('solute_only_full_RMSE_std','?')}"
        mult_w = f"{m.get('with_solvent_multi_RMSE_mean','?')} ± {m.get('with_solvent_multi_RMSE_std','?')}"
        mult_s = f"{m.get('solute_only_multi_RMSE_mean','?')} ± {m.get('solute_only_multi_RMSE_std','?')}"
        sing_w = f"{m.get('with_solvent_single_RMSE_mean','?')} ± {m.get('with_solvent_single_RMSE_std','?')}"
        sing_s = f"{m.get('solute_only_single_RMSE_mean','?')} ± {m.get('solute_only_single_RMSE_std','?')}"
        d_full = f"{m.get('full_solvent_gain_RMSE','?')} ({m.get('full_solvent_gain_pct','?')}%)"
        d_mult = f"{m.get('multi_solvent_gain_RMSE','?')} ({m.get('multi_solvent_gain_pct','?')}%)"
        d_sing = f"{m.get('single_solvent_gain_RMSE','?')} ({m.get('single_solvent_gain_pct','?')}%)"
        md.append(f"| {label} | +Solv | {full_w} | {mult_w} | {sing_w} | {d_full} | {d_mult} | {d_sing} |")
        md.append(f"| {label} | Solute-only | {full_s} | {mult_s} | {sing_s} |  |  |  |")
    md.append("")
    md.append("Δ = (Solute-only RMSE) − (+Solvent RMSE) on that subset; positive Δ means solvent encoding helps.")
    (RESULTS / "r1_14_summary.md").write_text("\n".join(md))
    print(f"  wrote results/r1_14_summary.md")

    # Print key numbers
    print("\n  === KEY NUMBERS ===")
    print(f"  Mean fraction multi-solvent test records: {agg['frac_multi_solvent_test_mean']:.1%}")
    for label, _, _ in PAIRS:
        m = agg["models"][label]
        print(f"  {label}:")
        print(f"     full   RMSE: +Solv {m.get('with_solvent_full_RMSE_mean')}  vs  Solute-only {m.get('solute_only_full_RMSE_mean')}  "
              f"Δ = {m.get('full_solvent_gain_RMSE')} ({m.get('full_solvent_gain_pct')}%)")
        print(f"     multi  RMSE: +Solv {m.get('with_solvent_multi_RMSE_mean')}  vs  Solute-only {m.get('solute_only_multi_RMSE_mean')}  "
              f"Δ = {m.get('multi_solvent_gain_RMSE')} ({m.get('multi_solvent_gain_pct')}%)")
        print(f"     single RMSE: +Solv {m.get('with_solvent_single_RMSE_mean')}  vs  Solute-only {m.get('solute_only_single_RMSE_mean')}  "
              f"Δ = {m.get('single_solvent_gain_RMSE')} ({m.get('single_solvent_gain_pct')}%)")


if __name__ == "__main__":
    main()
