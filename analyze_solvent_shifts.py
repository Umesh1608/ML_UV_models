#!/usr/bin/env python3
"""
R2-4: Per-solute solvent-induced lambda_max shifts (solvatochromism audit).

For each chromophore appearing in the v3 cleaned Joung+Beard dataset under
>= 2 distinct canonical solvents, compute the span (max - min) of measured
lambda_max values. This measures the magnitude of empirically observed
solvatochromism without invoking polarity scales (R2-5 separately groups
by polarity).

Outputs:
  results/r2_4_solvent_shifts.{json,csv,png}
  results/r2_4_summary.md
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results"
DATA = SCRIPT_DIR / "previous_code" / "UV_canonical_v3_dedup.csv"


def classify_chromophore(smi):
    """Lightweight class flag based on SMARTS fragments. For the top-10
    flagging only; not exhaustive."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return "other"
    smarts = {
        "azo":           "[$([NX2;v3]=[NX2;v3])]",
        "merocyanine":   "[$([#7]C=C(=O))]",
        "stilbene":      "[$(c-/C=C/-c)]",
        "BODIPY":        "[$([#5;D2]([F])([F]))]",
        "porphyrin":     "[$(c1[nH]c2cc1)]",
        "donor_acceptor": "[$([#7;X3;!$([#7]=*)]),$([#8;X2;!$([#8]=*)])].[$([N+](=O)[O-]),$(C#N)]",
        "hydroxycinnamate": "[$(c-/C=C/-C(=O)O)]",
        "indigo":        "O=C1Nc2ccccc2C1=Cc1ccccc1=O",  # rough indigo core
    }
    flags = []
    for name, smt in smarts.items():
        patt = Chem.MolFromSmarts(smt)
        if patt is None:
            continue
        if m.HasSubstructMatch(patt):
            flags.append(name)
    return ",".join(flags) if flags else "other"


def main():
    print(f"  loading {DATA}")
    df = pd.read_csv(DATA)
    df["lambda_max"] = pd.to_numeric(df["lambda_max"], errors="coerce")
    df = df.dropna(subset=["canon", "solvents", "lambda_max"]).reset_index(drop=True)
    print(f"  v3 size: {len(df)} records, "
          f"{df['canon'].nunique()} unique chromophores, "
          f"{df['solvents'].nunique()} unique solvents")

    # Aggregate per (canon, solvent) first (collapse intra-solvent duplicates
    # by averaging) so each (chrom, solvent) contributes one value.
    g = df.groupby(["canon", "solvents"])["lambda_max"].mean().reset_index()
    print(f"  unique (chrom, solvent) pairs: {len(g)}")

    # Per-chromophore: how many distinct solvents, what is the span?
    per_chrom = g.groupby("canon").agg(
        n_solvents=("solvents", "count"),
        lmax_min=("lambda_max", "min"),
        lmax_max=("lambda_max", "max"),
        lmax_median=("lambda_max", "median"),
    )
    per_chrom["span_nm"] = per_chrom["lmax_max"] - per_chrom["lmax_min"]

    multi = per_chrom[per_chrom["n_solvents"] >= 2].sort_values("span_nm", ascending=False)
    print(f"  chromophores with >=2 solvents: {len(multi)}")

    # Histogram of per-chromophore spans
    spans = multi["span_nm"].values
    stats = {
        "n_total_chromophores":      int(per_chrom.shape[0]),
        "n_multi_solvent_chromophores": int(len(multi)),
        "median_span_nm":  round(float(np.median(spans)),     3),
        "mean_span_nm":    round(float(np.mean(spans)),       3),
        "p75_span_nm":     round(float(np.percentile(spans, 75)), 3),
        "p90_span_nm":     round(float(np.percentile(spans, 90)), 3),
        "p95_span_nm":     round(float(np.percentile(spans, 95)), 3),
        "p99_span_nm":     round(float(np.percentile(spans, 99)), 3),
        "max_span_nm":     round(float(np.max(spans)),        3),
        "frac_span_eq_0":      round(float((spans == 0).mean()),    3),
        "frac_span_le_5nm":    round(float((spans <= 5).mean()),    3),
        "frac_span_le_10nm":   round(float((spans <= 10).mean()),   3),
        "frac_span_le_20nm":   round(float((spans <= 20).mean()),   3),
        "frac_span_gt_50nm":   round(float((spans > 50).mean()),    3),
        "frac_span_gt_100nm":  round(float((spans > 100).mean()),   3),
    }

    # Top 10 largest spans + chromophore class hint
    top = multi.head(15).reset_index()
    print("\n  Top 15 largest solvent-induced spans:")
    top_records = []
    for _, r in top.iterrows():
        cls = classify_chromophore(r["canon"])
        top_records.append({
            "canon": r["canon"],
            "n_solvents": int(r["n_solvents"]),
            "span_nm": round(float(r["span_nm"]), 1),
            "lmax_min": round(float(r["lmax_min"]), 1),
            "lmax_max": round(float(r["lmax_max"]), 1),
            "class_flags": cls,
        })
        print(f"    {r['span_nm']:6.1f} nm  (n_solvents={r['n_solvents']:2d})  "
              f"[{cls}]  {r['canon']}")

    out = {"stats": stats, "top_spans": top_records}
    (RESULTS / "r2_4_solvent_shifts.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote results/r2_4_solvent_shifts.json")

    multi.reset_index().to_csv(RESULTS / "r2_4_solvent_shifts.csv", index=False)
    print(f"  wrote results/r2_4_solvent_shifts.csv ({len(multi)} rows)")

    # Plot histogram
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Clip very long tail for readability
    span_clip = np.clip(spans, 0, 200)
    ax.hist(span_clip, bins=40, color="#7BAACE", edgecolor="#3D5B73", linewidth=0.4)
    ax.axvline(stats["median_span_nm"], color="crimson", linestyle="--", linewidth=1.4,
               label=f"median = {stats['median_span_nm']:.1f} nm")
    ax.axvline(stats["p95_span_nm"], color="orange", linestyle=":", linewidth=1.4,
               label=f"p95 = {stats['p95_span_nm']:.1f} nm")
    ax.set_xlabel(r"Per-chromophore solvent-induced span $\max - \min$ ($\lambda_{\max}$, nm)",
                  fontsize=11)
    ax.set_ylabel("Count (chromophores)", fontsize=11)
    n_overflow = int((spans > 200).sum())
    ax.set_title(f"Per-chromophore solvatochromic span "
                 f"(n = {len(multi):,} chromophores with $\\geq$2 solvents; "
                 f"{n_overflow} have span $>$ 200 nm, clipped)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_4_solvent_shifts.png", dpi=180, bbox_inches="tight")
    print(f"  wrote results/r2_4_solvent_shifts.png")

    # Markdown summary
    md = []
    md.append("# R2-4 Step 0: Per-Solute Solvent-Induced lambda_max Shifts")
    md.append("")
    md.append(f"- Total unique chromophores: {stats['n_total_chromophores']:,}")
    md.append(f"- Chromophores with >=2 solvents: **{stats['n_multi_solvent_chromophores']:,}** "
              f"({stats['n_multi_solvent_chromophores']/stats['n_total_chromophores']*100:.1f}%)")
    md.append("")
    md.append("## Span distribution (per-chromophore max - min)")
    md.append(f"- Median: **{stats['median_span_nm']:.1f} nm**")
    md.append(f"- Mean:   {stats['mean_span_nm']:.1f} nm")
    md.append(f"- 75th pct: {stats['p75_span_nm']:.1f} nm")
    md.append(f"- 90th pct: {stats['p90_span_nm']:.1f} nm")
    md.append(f"- 95th pct: **{stats['p95_span_nm']:.1f} nm**")
    md.append(f"- 99th pct: {stats['p99_span_nm']:.1f} nm")
    md.append(f"- Max:     {stats['max_span_nm']:.1f} nm")
    md.append("")
    md.append(f"- Fraction with span = 0 nm: {stats['frac_span_eq_0']*100:.1f}%")
    md.append(f"- Fraction with span <= 5 nm: {stats['frac_span_le_5nm']*100:.1f}%")
    md.append(f"- Fraction with span <= 10 nm: {stats['frac_span_le_10nm']*100:.1f}%")
    md.append(f"- Fraction with span <= 20 nm: {stats['frac_span_le_20nm']*100:.1f}%")
    md.append(f"- Fraction with span >  50 nm: {stats['frac_span_gt_50nm']*100:.1f}%")
    md.append(f"- Fraction with span > 100 nm: {stats['frac_span_gt_100nm']*100:.1f}%")
    md.append("")
    md.append("## Top 15 chromophores by solvatochromic span")
    md.append("")
    md.append("| Span (nm) | # solvents | Class flags | Canonical SMILES |")
    md.append("|---:|---:|:--|:--|")
    for r in top_records:
        md.append(f"| {r['span_nm']:.1f} | {r['n_solvents']} | {r['class_flags']} | `{r['canon']}` |")
    (RESULTS / "r2_4_summary.md").write_text("\n".join(md))
    print(f"  wrote results/r2_4_summary.md")

    print("\n  === KEY NUMBERS ===")
    print(f"  multi-solvent chromophores: {stats['n_multi_solvent_chromophores']:,}")
    print(f"  median span = {stats['median_span_nm']:.1f} nm")
    print(f"  p95 span = {stats['p95_span_nm']:.1f} nm")
    print(f"  frac with span > 50 nm: {stats['frac_span_gt_50nm']*100:.1f}%")


if __name__ == "__main__":
    main()
