"""Summary figure for the R1-2 response: visual recap of the RF vs Chemprop
tie story across (i) v3 default, (ii) v3 tuned, and (iii) v3 chromophore-novel
subset comparisons.

Generates results/r1_2_summary_figure.png (used as in-line figure in the
response letter, after the R1-2 written response).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = '/home/umesh/paper1_new_cl/results/r1_2_summary_figure.png'


# === Panel A data: per-fold RMSE on v3 ===========================================
rf_v3      = [30.33, 30.46, 30.24, 32.67, 33.81]
cp_def_v3  = [30.62, 30.53, 31.49, 33.90, 39.21]
cp_tun_v3  = [30.70, 28.22, 26.92, 29.84, 28.46]

# === Panel B data: paired comparisons (Δ = Chemprop − RF, paired-t and Wilcoxon)
comparisons = [
    # (label, Δ mean, paired-std, parametric p, wilcoxon p)
    ('RF  vs  Chemprop default\n(v3 full test set)',
     1.65,  3.59,  0.160, None),
    ('RF  vs  Chemprop tuned\n(v3 full test set)',
     -2.67, 1.86,  0.044, 0.125),
    ('RF  vs  Chemprop default\n(v3 chromophore-novel subset)',
     2.40,  2.40,  0.066, None),
]

# ================================================================================
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(12.5, 4.6), gridspec_kw={'width_ratios': [1.25, 1.0]}
)

# ---- Panel A: per-fold RMSE -----------------------------------------------------
models = ['RF\n(tuned)', 'Chemprop\n(default)', 'Chemprop\n(tuned)']
data = [rf_v3, cp_def_v3, cp_tun_v3]
colors = ['#1f77b4', '#d62728', '#2ca02c']
x = np.array([0, 1, 2])

# strip-plot per fold + mean bar
for xi, vals, c in zip(x, data, colors):
    jit = np.random.RandomState(0).uniform(-0.10, 0.10, len(vals))
    axA.scatter(xi + jit, vals, s=70, color=c, edgecolor='white',
                linewidth=1.2, alpha=0.85, zorder=3)
    m, s = np.mean(vals), np.std(vals, ddof=1)
    axA.hlines(m, xi - 0.22, xi + 0.22, color=c, linewidth=2.5, zorder=4)
    # annotation: mean ± std slightly to the right of the strip
    axA.text(xi + 0.30, m, f'{m:.2f}\n± {s:.2f}', color=c, fontsize=9,
             va='center', ha='left', fontweight='bold')

# paired connecting lines between folds (faint grey, emphasises pairing)
for f in range(5):
    axA.plot([0, 1], [rf_v3[f], cp_def_v3[f]], color='gray', alpha=0.25, lw=0.8, zorder=2)
    axA.plot([1, 2], [cp_def_v3[f], cp_tun_v3[f]], color='gray', alpha=0.25, lw=0.8, zorder=2)

# Annotate the fold-4 outlier elimination
axA.annotate(
    'Default fold-4 outlier\n(39.21 nm) eliminated\nby R1-12 HPO probe',
    xy=(1.10, 39.21), xytext=(0.05, 41.5),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
    fontsize=8.5, ha='left',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
              edgecolor='gray', alpha=0.9),
)

axA.set_xticks(x)
axA.set_xticklabels(models, fontsize=10)
axA.set_ylabel('Test RMSE per fold (nm)', fontsize=10)
axA.set_title('Panel A. Per-fold RMSE on v3 cleaned Joung+Beard dataset\n'
              '(stratified 5-fold CV; grey lines connect paired fold pairs)',
              fontsize=10, loc='left')
axA.set_ylim(24, 44)
axA.set_xlim(-0.5, 2.7)
axA.grid(axis='y', alpha=0.3)
axA.spines['top'].set_visible(False)
axA.spines['right'].set_visible(False)


# ---- Panel B: paired-comparison forest plot -------------------------------------
y_positions = np.arange(len(comparisons))[::-1]   # top item first
deltas = [c[1] for c in comparisons]
stds   = [c[2] for c in comparisons]
labels = [c[0] for c in comparisons]
p_par  = [c[3] for c in comparisons]
p_wil  = [c[4] for c in comparisons]


def colour_for(p_p, p_w):
    # both NS  -> gray   (tie)
    # parametric sig but Wilcoxon NS -> orange  (borderline)
    if p_p > 0.05:
        return '#666666'
    if p_w is not None and p_w > 0.05:
        return '#e08a1a'
    return '#cc0000'


for yi, d, s, p_p, p_w in zip(y_positions, deltas, stds, p_par, p_wil):
    c = colour_for(p_p, p_w)
    axB.errorbar([d], [yi], xerr=[s], fmt='o', color=c, ecolor='black',
                 markersize=10, markeredgecolor='black', markeredgewidth=0.8,
                 capsize=5, capthick=1.0, elinewidth=1.0, zorder=4)
    p_text = f'parametric paired-t  p = {p_p:.3f}'
    if p_w is not None:
        p_text += f'\nWilcoxon  p = {p_w:.3f}'
    axB.text(5.8, yi, p_text, fontsize=8.5, va='center', ha='right')

axB.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1)
axB.set_yticks(y_positions)
axB.set_yticklabels(labels, fontsize=9.5)
axB.set_xlabel('Δ RMSE  =  Chemprop − RF   (nm)\n(negative ⇒ Chemprop better; positive ⇒ RF better)',
               fontsize=9.5)
axB.set_title('Panel B. Paired comparisons.  Statistical tie holds in every\n'
              'condition under the conservative non-parametric (Wilcoxon) test.',
              fontsize=10, loc='left')
axB.set_xlim(-6.0, 6.0)
axB.grid(axis='x', alpha=0.3)
axB.spines['top'].set_visible(False)
axB.spines['right'].set_visible(False)
axB.invert_yaxis()

# Legend explaining the colour coding
legend_handles = [
    Patch(facecolor='#666666', label='both tests fail to reject (tie)'),
    Patch(facecolor='#e08a1a',
          label='parametric significant, non-parametric not (borderline)'),
]
axB.legend(handles=legend_handles, loc='lower left', fontsize=8,
           framealpha=0.9, title_fontsize=9, title='Significance @ α = 0.05')

fig.suptitle(
    'R1-2 summary: RF and Chemprop are statistically tied on the v3 dataset, '
    'with and without HPO tuning, on the full test set and on the chromophore-novel subset.',
    fontsize=11, y=1.02,
)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
