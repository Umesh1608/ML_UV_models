# JCIM Revision — Current Status

**Manuscript:** ci-2026-009433, "When Do Simple Models Win? Machine Learning Architectures for UV Absorption Prediction" (title revised — original ended with "and Insights for General Molecular Property Prediction")
**Decision:** Major Revisions, deadline 17-Jun-2026
**See also:** `reviewer_comments.md` for the full reviewer text.

## How to continue this work (read first)

You are picking up a JCIM revision in progress. The user (Umesh) is addressing reviewer concerns one-by-one. For each concern:

1. Read the reviewer's full comment in `reviewer_comments.md`.
2. Discuss the strategy with the user, present the proposed edits, and **wait for their OK** before editing.
3. Apply the change in **both** `benchmark_paper_jcim_revised.tex` (clean) and `benchmark_paper_jcim_revised_marked.tex` (marked copy with `\removed{...}` / `\added{...}`).
4. Append a properly formatted entry to `response_to_reviewers.tex` (reviewer quote, our response, bulleted "Changes to manuscript").
5. Update the TaskList: mark the concern complete, start the next one.
6. **Commit + push** after each concern (per user preference: auto-push after every commit, no `Co-Authored-By` lines).

The originals (`benchmark_paper_jcim.tex`, `supporting_information.tex`) are kept untouched.

## Files

| File | Role |
|------|------|
| `benchmark_paper_jcim.tex` | Original manuscript (DO NOT modify) |
| `supporting_information.tex` | Original SI (DO NOT modify) |
| `benchmark_paper_jcim_revised.tex` | (b) Clean revised manuscript |
| `benchmark_paper_jcim_revised_marked.tex` | (c) Marked manuscript — `\added{}`/`\removed{}` macros defined in preamble |
| `supporting_information_revised.tex` | (d) Clean revised SI |
| `response_to_reviewers.tex` | (a) Point-by-point response document |
| `reviewer_comments.md` | Full reviewer text for reference |
| `revision_status.md` | This file |

## Concerns: completed and pending

### Completed
- **R1-1** Removed "general molecular property prediction" overclaims (title, abstract closing, intro "ideal benchmarking", conclusion "More broadly" paragraph). Reframed locality as testable hypothesis.
- **R1-2** Removed all "RF is optimal" language; reframed as statistical tie with D-MPNN ($p = 0.77$), RF advantage = compute cost.
- **R1-3** Removed "3D methods offering advantages" leftover; corrected "1D representations" to "1D and 2D" to include D-MPNN.
- **R1-4** Rephrased "Architecture-independent solvent encoding" → "Concatenation-based solvent encoding generalizes across architectures"; explicitly note each model's native encoding.

### Pending (priority order)

Next up is **R1-5** (non-local compounds subset analysis). Strategy locked in with user:
- Identify donor-acceptor (D-A, D-A-D) compounds via SMARTS and long-wavelength (λ > 500 nm) compounds.
- Recompute RMSE/MAE/R² per model on each subset using **existing per-fold predictions** (no retraining needed).
- Produce a table + 1-paragraph discussion **in the main text** (not SI).
- Goal: directly test the locality hypothesis and either strengthen it or honestly note where it breaks down.

All other pending concerns are in the TaskList. The biggest wall-clock items are R1-8 (Mayr duplicate protocol → re-run all 5 model families × 5 folds, GPU needed) and R1-12 (D-MPNN HPO sweep).

## Conventions

- **No `Co-Authored-By: Claude` lines in commits.** No AI/Claude disclosure in paper text or commit messages.
- **Auto-push after every commit** with `git push origin main`.
- **Concise progress updates with tables.**
- **Verify data/code before claiming facts.** Memory is point-in-time; check `git log`, file contents, run scripts where needed.
- **Marked copy macros** (`benchmark_paper_jcim_revised_marked.tex` preamble):
  ```latex
  \newcommand{\added}[1]{\textcolor{addedblue}{#1}}
  \newcommand{\removed}[1]{\textcolor{removedred}{\st{#1}}}
  ```

## Verified external facts (do not change without re-verifying source)

- nablaColors (Potapov 2026): 26,369 pairs, best model **UniProp**, RMSE 27.2.
- ChemBERTa: `seyonec/ChemBERTa-zinc-base-v1`, pretrained on **ZINC** (not 77M PubChem).
- Liu 2023: MTBG = BiGRU + GraphSAGE hybrid; we use the BiGRU component only.
- Mamede 2021: Scientific Reports, first author **Florbela**.
- UV-adVISor 2021: authors **Urbina, Batra, Ekins** et al., volume 93.
- Lupo Pasini 2023: authors **Mehta, Yoo, Irle**, page 546.

## Cross-validated v2 results (in current paper text, used in responses)

| Model | RMSE (nm) | MAE | R² |
|-------|-----------|-----|-----|
| RF TUNED | 31.34 ± 1.82 | 15.16 ± 0.44 | 0.914 |
| Chemprop D-MPNN | 31.69 ± 3.18 | 16.84 ± 1.74 | 0.911 |
| XGBoost MSE | 33.70 ± 1.61 | 20.05 ± 0.48 | 0.900 |
| BiGRU TUNED | 34.71 ± 1.40 | 18.09 ± 0.59 | 0.894 |
| BiGRU+Solvent (default) | 36.45 ± 1.12 | 20.70 ± 0.51 | 0.884 |
| ChemBERTa | 50.39 ± 2.30 | 24.31 ± 1.04 | 0.777 |

RF vs Chemprop: $p = 0.77$ (paired test, not significant).
