#!/usr/bin/env python3
"""
R1-8 Step 0 — Quantify what the Greenman/Song duplicate-handling protocol
would do to our primary Joung+Beard dataset.

Reviewer R1-8 cites Greenman et al.\ (2022, Chem.\ Sci., DOI 10.1039/D1SC05677H)
and Song et al.\ (2025, Digital Discovery, DOI 10.1039/D5DD00402K) as having a
more rigorous protocol than our "keep first occurrence" approach:

    for each group of records with the same (canonical solute, canonical solvent):
        if max(lambda) - min(lambda)  <= 5 nm   ->  average
        if max(lambda) - min(lambda)   > 5 nm   ->  drop the whole group
        singletons unchanged

This script does *only* the analysis: it reports record counts, group statistics,
wavelength-binned stratification, and solvent breakdown. It does NOT modify the
manuscript or retrain anything. The output is read by the human to decide
whether to commit to retraining all five model families.

Outputs:
    results/r1_8_dedup_stats.json           machine-readable summary
    results/r1_8_spread_histogram.png       histogram of within-group spreads
    results/r1_8_wavelength_stratified.csv  per-bin breakdown
    results/r1_8_solvent_breakdown.csv      per-solvent duplicate behaviour
    results/r1_8_summary.md                 short human-readable report
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "previous_code" / "UV_canonical_full_dataset.csv"
OUT_DIR = SCRIPT_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

THRESHOLD_NM = 5.0  # Greenman / Song cutoff for "small" disagreements


def canon(smi):
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def main():
    print(f"[LOAD] {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    n_raw = len(raw)
    print(f"  raw rows: {n_raw}")

    # Drop rows missing any of (canon, solvents, lambda_max)
    df = raw[["canon", "solvents", "lambda_max"]].dropna().copy()
    n_after_dropna = len(df)
    print(f"  after dropna: {n_after_dropna}")

    # Re-canonicalise both columns through RDKit to make the dedup consistent
    # with what Greenman/Song would do. This catches edge cases where the
    # saved canonical column lost some normalisation.
    print(f"[CANON] re-canonicalising solute SMILES...")
    df["canon_rd"] = [canon(s) for s in df["canon"]]
    print(f"[CANON] re-canonicalising solvent SMILES...")
    df["solv_rd"] = [canon(s) for s in df["solvents"]]
    n_before_canon_drop = len(df)
    df = df.dropna(subset=["canon_rd", "solv_rd"]).copy()
    n_canon_dropped = n_before_canon_drop - len(df)
    print(f"  RDKit canon failures dropped: {n_canon_dropped}")
    print(f"  after canon: {len(df)}")

    # ── Current behaviour: pipeline takes all 18,755 rows as-is (no dedup).
    # The CLAUDE.md description ('retain first occurrence') is partial — the
    # actual training pipeline calls `dropna()` but NOT `drop_duplicates()`.
    # We report both numbers so the reviewer response is precise.
    n_after_first_occurrence = len(df.drop_duplicates(subset=["canon_rd", "solv_rd"], keep="first"))
    print(f"  hypothetical keep-first-occurrence dedup: {n_after_first_occurrence}")

    # ── Greenman / Song protocol -------------------------------------------
    print(f"[PROTOCOL] grouping by (canonical solute, canonical solvent)...")
    grouped = df.groupby(["canon_rd", "solv_rd"])["lambda_max"]

    sizes = grouped.size()
    n_groups_total = int(len(sizes))
    n_singletons = int((sizes == 1).sum())
    n_dup_groups = int((sizes >= 2).sum())

    spreads = (grouped.max() - grouped.min())
    means = grouped.mean()
    print(f"  total groups: {n_groups_total} (singletons {n_singletons}, duplicates {n_dup_groups})")

    # Classify duplicate groups
    dup_mask = sizes >= 2
    spread_dup = spreads[dup_mask]
    n_groups_dropped = int((spread_dup > THRESHOLD_NM).sum())   # discard entire group
    n_groups_averaged = int((spread_dup <= THRESHOLD_NM).sum())  # collapse to mean

    records_in_dropped = int(sizes[dup_mask][spread_dup > THRESHOLD_NM].sum())
    records_in_averaged = int(sizes[dup_mask][spread_dup <= THRESHOLD_NM].sum())
    records_singletons = int(sizes[~dup_mask].sum())

    # After protocol: singletons stay, averaged groups collapse to 1 record,
    # dropped groups contribute 0
    n_after_protocol = records_singletons + n_groups_averaged
    print(f"  -> after Greenman/Song protocol: {n_after_protocol} unique (solute, solvent) records")
    print(f"     - {records_singletons} singletons unchanged")
    print(f"     - {n_groups_averaged} duplicate groups averaged (collapsing {records_in_averaged} records)")
    print(f"     - {n_groups_dropped} duplicate groups dropped (discarding {records_in_dropped} records)")

    # Loss accounting from the perspective of what currently goes into training
    # (the 18,755 rows in df above; or equivalently the records actually present)
    n_lost_vs_current = n_after_dropna - n_after_protocol
    pct_lost_vs_current = 100.0 * n_lost_vs_current / max(n_after_dropna, 1)
    n_lost_vs_first_occ = n_after_first_occurrence - n_after_protocol
    pct_lost_vs_first_occ = 100.0 * n_lost_vs_first_occ / max(n_after_first_occurrence, 1)
    print(f"\n  vs current pipeline (no dedup, {n_after_dropna} rows): "
          f"lose {n_lost_vs_current} rows ({pct_lost_vs_current:.2f}%)")
    print(f"  vs first-occurrence dedup ({n_after_first_occurrence} rows): "
          f"lose {n_lost_vs_first_occ} rows ({pct_lost_vs_first_occ:.2f}%)")

    # ── Spread histogram --------------------------------------------------
    print(f"\n[FIG] within-group spread histogram (duplicate groups only)...")
    bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 1e4])
    hist, _ = np.histogram(spread_dup.values, bins=bins)
    bin_labels = ["0--5", "5--10", "10--15", "15--20", "20--25", "25--30",
                  "30--35", "35--40", "40--45", "45--50", ">50"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(bin_labels, hist, color=["#4a9d4a" if i == 0 else "#c54a4a"
                                            for i in range(len(bin_labels))],
                  edgecolor="black", linewidth=0.5)
    for b, h in zip(bars, hist):
        ax.text(b.get_x() + b.get_width() / 2, h,
                str(int(h)), ha="center", va="bottom", fontsize=8)
    ax.axvspan(-0.5, 0.5, alpha=0.10, color="green")
    ax.set_xlabel(r"within-group spread (max $-$ min, nm)")
    ax.set_ylabel("number of duplicate groups")
    ax.set_title(f"Within-group $\\lambda_{{\\max}}$ spread for {n_dup_groups} duplicate groups\n"
                 f"green = averaged ({n_groups_averaged}), red = dropped under 5 nm rule ({n_groups_dropped})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "r1_8_spread_histogram.png", dpi=200)
    print(f"  saved {OUT_DIR/'r1_8_spread_histogram.png'}")

    # ── Wavelength-binned stratification ---------------------------------
    print(f"\n[STRAT] wavelength-binned breakdown...")
    bin_edges = [-np.inf, 300, 400, 500, 600, np.inf]
    bin_names = ["<300", "300-400", "400-500", "500-600", ">600"]
    rows = []
    for name, lo, hi in zip(bin_names, bin_edges[:-1], bin_edges[1:]):
        mask = (means > lo) & (means <= hi)
        groups_here = sizes[mask]
        spreads_here = spreads[mask]

        dup_here = groups_here >= 2
        n_records = int(groups_here.sum())
        n_groups = int(len(groups_here))
        n_sing = int((groups_here == 1).sum())
        n_dup_grp = int(dup_here.sum())
        n_avg_grp = int(((spreads_here <= THRESHOLD_NM) & dup_here).sum())
        n_drop_grp = int(((spreads_here > THRESHOLD_NM) & dup_here).sum())
        n_rec_avg = int(groups_here[(spreads_here <= THRESHOLD_NM) & dup_here].sum())
        n_rec_drop = int(groups_here[(spreads_here > THRESHOLD_NM) & dup_here].sum())
        n_after_protocol_bin = n_sing + n_avg_grp
        rows.append({
            "wavelength_bin_nm": name,
            "records_now": n_records,
            "groups_total": n_groups,
            "groups_singleton": n_sing,
            "groups_duplicate": n_dup_grp,
            "groups_averaged_le5nm": n_avg_grp,
            "groups_dropped_gt5nm": n_drop_grp,
            "records_in_avg_groups": n_rec_avg,
            "records_in_drop_groups": n_rec_drop,
            "records_after_protocol": n_after_protocol_bin,
            "pct_lost": round(100 * (n_records - n_after_protocol_bin) / max(n_records, 1), 2),
        })
    strat_df = pd.DataFrame(rows)
    strat_df.to_csv(OUT_DIR / "r1_8_wavelength_stratified.csv", index=False)
    print(strat_df.to_string(index=False))

    # ── Solvent breakdown -------------------------------------------------
    print(f"\n[STRAT] top-10 solvent breakdown...")
    # Count records per solvent
    solv_counts = df["solv_rd"].value_counts().head(10)
    rows_solv = []
    for solv in solv_counts.index:
        sub = df[df["solv_rd"] == solv]
        sub_groups = sub.groupby(["canon_rd"])["lambda_max"]
        sub_sizes = sub_groups.size()
        sub_spreads = sub_groups.max() - sub_groups.min()
        dup_mask_s = sub_sizes >= 2
        n_records_s = int(sub_sizes.sum())
        n_groups_s = int(len(sub_sizes))
        n_dup_groups_s = int(dup_mask_s.sum())
        n_drop_groups_s = int(((sub_spreads > THRESHOLD_NM) & dup_mask_s).sum())
        rows_solv.append({
            "solvent_canon": solv,
            "records": n_records_s,
            "unique_solutes": n_groups_s,
            "duplicate_groups": n_dup_groups_s,
            "dropped_gt5nm_groups": n_drop_groups_s,
            "drop_rate_within_dup": round(n_drop_groups_s / max(n_dup_groups_s, 1), 3),
        })
    solv_df = pd.DataFrame(rows_solv)
    solv_df.to_csv(OUT_DIR / "r1_8_solvent_breakdown.csv", index=False)
    print(solv_df.to_string(index=False))

    # ── Spread distribution percentiles ----------------------------------
    spread_percentiles = {
        "p50": float(np.percentile(spread_dup.values, 50)),
        "p75": float(np.percentile(spread_dup.values, 75)),
        "p90": float(np.percentile(spread_dup.values, 90)),
        "p95": float(np.percentile(spread_dup.values, 95)),
        "p99": float(np.percentile(spread_dup.values, 99)),
        "max": float(spread_dup.values.max()),
        "mean": float(spread_dup.values.mean()),
    }

    # ── Save JSON summary ------------------------------------------------
    summary = {
        "raw_rows": int(n_raw),
        "rows_after_dropna": int(n_after_dropna),
        "rows_after_canon_filter": int(len(df)),
        "rows_keep_first_occurrence_dedup": int(n_after_first_occurrence),
        "rows_after_greenman_song_protocol": int(n_after_protocol),
        "pct_lost_vs_current": round(pct_lost_vs_current, 2),
        "pct_lost_vs_first_occurrence_dedup": round(pct_lost_vs_first_occ, 2),
        "n_groups_total": n_groups_total,
        "n_groups_singleton": n_singletons,
        "n_groups_duplicate": n_dup_groups,
        "n_groups_averaged_le5nm": n_groups_averaged,
        "n_groups_dropped_gt5nm": n_groups_dropped,
        "records_in_averaged_groups": records_in_averaged,
        "records_in_dropped_groups": records_in_dropped,
        "records_in_singleton_groups": records_singletons,
        "spread_percentiles_nm": spread_percentiles,
        "threshold_nm": THRESHOLD_NM,
        "histogram_bins_lower_inclusive": [float(b) for b in bins[:-1]],
        "histogram_bins_upper_exclusive": [float(b) for b in bins[1:]],
        "histogram_counts": [int(h) for h in hist],
    }
    with open(OUT_DIR / "r1_8_dedup_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OUT] wrote {OUT_DIR/'r1_8_dedup_stats.json'}")

    # ── Short markdown summary -------------------------------------------
    md = []
    md.append(f"# R1-8 Step 0 — Duplicate-Handling Analysis Summary\n")
    md.append(f"_Protocol: Greenman 2022 / Song 2025 ({THRESHOLD_NM} nm threshold)_\n")
    md.append(f"## Headline numbers\n")
    md.append(f"| stage | rows |")
    md.append(f"|---|---|")
    md.append(f"| raw CSV | {n_raw:,} |")
    md.append(f"| after dropna (current training input) | {n_after_dropna:,} |")
    md.append(f"| after hypothetical keep-first-occurrence dedup | {n_after_first_occurrence:,} |")
    md.append(f"| **after Greenman/Song protocol** | **{n_after_protocol:,}** |")
    md.append("")
    md.append(f"**Loss vs current pipeline:** {n_lost_vs_current:,} rows ({pct_lost_vs_current:.2f}%)  ")
    md.append(f"**Loss vs first-occurrence dedup:** {n_lost_vs_first_occ:,} rows ({pct_lost_vs_first_occ:.2f}%)\n")
    md.append(f"## Group statistics\n")
    md.append(f"- total (canon solute, canon solvent) groups: **{n_groups_total:,}**")
    md.append(f"- singletons (n=1): {n_singletons:,}")
    md.append(f"- duplicate groups (n≥2): {n_dup_groups:,}")
    md.append(f"- duplicate groups averaged (spread ≤ 5 nm): {n_groups_averaged:,} (covering {records_in_averaged:,} records)")
    md.append(f"- duplicate groups dropped (spread > 5 nm): {n_groups_dropped:,} (covering {records_in_dropped:,} records)\n")
    md.append(f"## Spread percentiles (duplicate groups only)\n")
    md.append("| percentile | spread (nm) |")
    md.append("|---|---|")
    for k, v in spread_percentiles.items():
        md.append(f"| {k} | {v:.1f} |")
    md.append("")
    def df_to_md(df):
        """Hand-format DataFrame as markdown (avoids the optional `tabulate` dep)."""
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    md.append(f"## Wavelength-stratified breakdown\n")
    md.append(df_to_md(strat_df))
    md.append("\n## Top-10 solvent breakdown\n")
    md.append(df_to_md(solv_df))
    md.append("")
    with open(OUT_DIR / "r1_8_summary.md", "w") as f:
        f.write("\n".join(md))
    print(f"[OUT] wrote {OUT_DIR/'r1_8_summary.md'}")


if __name__ == "__main__":
    main()
