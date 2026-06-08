#!/usr/bin/env python3
"""
Create Table of Contents (TOC) graphic for JCIM submission.
ACS specification: 3.25" x 1.75" at 300 dpi.

Shows the key finding: RF wins on CV benchmarks, DL wins on novel compounds.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(3.25, 1.75))

# Use gridspec for better control
gs = fig.add_gridspec(1, 2, wspace=0.6, left=0.15, right=0.95, top=0.88, bottom=0.22)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# --- Left panel: Cross-validated RMSE (RF wins) ---
models = ["RF", "XGB", "BiGRU", "ChemB."]
rmse = [31.34, 33.70, 34.71, 50.39]
colors = ["#2ca02c", "#2ca02c", "#1f77b4", "#1f77b4"]

bars1 = ax1.barh(models, rmse, color=colors, edgecolor="white", height=0.6)
ax1.set_xlabel("RMSE (nm)", fontsize=5.5, labelpad=2)
ax1.set_title("In-Distribution", fontsize=7, fontweight="bold", pad=4)
ax1.tick_params(labelsize=5.5, pad=1)
ax1.set_xlim(0, 60)
ax1.invert_yaxis()
for bar, val in zip(bars1, rmse):
    ax1.text(val + 0.8, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}", va="center", fontsize=4.5, color="gray")

# --- Right panel: Wetlab MAE (DL wins) ---
models2 = ["RF", "BiGRU", "ChemB."]
mae = [38.5, 28.6, 26.3]
colors2 = ["#2ca02c", "#1f77b4", "#1f77b4"]

bars2 = ax2.barh(models2, mae, color=colors2, edgecolor="white", height=0.55)
ax2.set_xlabel("MAE (nm)", fontsize=5.5, labelpad=2)
ax2.set_title("Novel Compounds", fontsize=7, fontweight="bold", pad=4)
ax2.tick_params(labelsize=5.5, pad=1)
ax2.set_xlim(0, 47)
ax2.invert_yaxis()
for bar, val in zip(bars2, mae):
    ax2.text(val + 0.8, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}", va="center", fontsize=4.5, color="gray")

# Legend at bottom
fp_patch = mpatches.Patch(color="#2ca02c", label="Fingerprint-based")
dl_patch = mpatches.Patch(color="#1f77b4", label="Deep Learning")
fig.legend(handles=[fp_patch, dl_patch], loc="lower center", ncol=2,
           fontsize=5, frameon=False, bbox_to_anchor=(0.55, 0.01))

for ext in ("png", "tiff"):
    fig.savefig(f"results/toc_graphic.{ext}", dpi=300)
plt.close()
print("TOC graphic saved to results/toc_graphic.png and results/toc_graphic.tiff")
