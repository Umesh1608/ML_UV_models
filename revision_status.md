# JCIM Revision — Current Status

**Manuscript:** ci-2026-009433.R1, "When Do Simple Models Win? Machine Learning Architectures for UV Absorption Prediction"
**Round 2 decision:** revise (editor: Prof. John Kitchin) — deadline **27 July 2026**
**See also:** `reviewer_comments.md` (round-1 text), `jcim_secondrevision_submission/README.md` (upload list).

## Where things stand

**Round 2 is complete and verified ready to upload** (audited 27 July 2026).

All second-round items are addressed and reflected in the compiled PDFs:

| Item | Concern | Status |
|------|---------|--------|
| E1 | Table 5 recovery-% caption inconsistent | ✅ |
| E2 | Regenerate Figures 5, 6, 9 from v3 predictions | ✅ |
| E3 | Abstract vs. Methods on RF hyperparameter tuning | ✅ |
| E4 | Figure 10 ↔ Table 6 consistency (default-BiGRU ensemble) | ✅ |
| R2r-1 | Verbatim reviewer quotes in response document | ✅ |
| R2r-2 | Numbered SI sections S1–S21 + working cross-refs | ✅ |
| R2r-3 | v2/v3 dataset versions defined at first use | ✅ |
| R2r-4 | Inter-revision phrasing removed from captions | ✅ |
| R2r-5 | AI-assistance disclosure updated | ✅ |
| R2r-6 | Dedicated MIT-licensed repo for the manuscript code | ✅ |
| R2r-7 | Figure 2(b) parallel D-MPNN encoders | ✅ |

Round-1 items (R1-1…R1-19, R2-1…R2-10) are all closed; the responses are retained
as Part B of the response document.

## Submission-day audit (27 July 2026)

Verified before upload:

- All five PDFs build cleanly from source with **zero** undefined references or citations.
- Manuscript (57 pp), marked copy (63 pp), and SI (47 pp) reproduce **byte-for-byte**
  from the committed `.tex` — the PDFs are not stale.
- All 21 SI sections are numbered S1–S21, and all 28 hardcoded `Section~S<n>`
  cross-references in the manuscript resolve to the topically correct section.
- External links live: `ML_UV_models_jcim` (200), `ML_UV_models` (200),
  Zenodo DOI 10.5281/zenodo.20600225 (200).
- Headline numbers agree across abstract, Table 2, Table 7, and the response letter.

**One defect found and fixed:** the staged `response_to_reviewers.pdf` had been compiled
without a complete BibTeX pass — 14 citations rendered as `[?]` and the References page
was missing entirely. Rebuilt (45 → 46 pages, all citations resolved). The cover letter
and response letter were also rebuilt so they carry the 27 July submission date.

**Lesson:** always re-scan compiled PDFs for `[?]` (undefined *citation*) as well as `??`
(undefined *reference*) before uploading — the two markers are different, and a grep for
`??` alone will not catch a missing bibliography.

## Files

Working copies live in `jcim_secondrevision_submission/` (the first-revision submission is
preserved unchanged in `jcim_firstrevision_submission/`).

| File | Role |
|------|------|
| `benchmark_paper_jcim_revised.tex` | (b) Clean revised manuscript |
| `benchmark_paper_jcim_revised_marked.tex` | (c) Marked manuscript |
| `supporting_information_revised.tex` | (d) Clean revised SI |
| `response_to_reviewers.tex` | (a) Point-by-point response (Part A round 2, Part B round 1) |
| `cover_letter_revised.tex` | Cover letter |
| `benchmark_paper_jcim.tex`, `supporting_information.tex` | Originals (DO NOT modify) |

Compile with `latexmk -pdf <file>.tex` from **inside** `jcim_secondrevision_submission/`
so `\graphicspath` resolves figures via `../results/`. Only `.tex`/`.bib`/`README` are
tracked in git; the PDFs are build artifacts.

## Marked-copy conventions

Round-1 changes are blue/red; round-2 changes use a second, distinct colour pair so the
two rounds stay separable:

```latex
\newcommand{\added}[1]{{\color{addedblue}#1}}      % round 1
\newcommand{\removed}[1]{{\color{removedred}#1}}   % round 1
\newcommand{\addedii}[1]{{\color{addedgreen}#1}}   % round 2
\newcommand{\removedii}[1]{{\color{removedorange}#1}}
```

Strikethrough via soul's `\st` cannot span paragraph breaks or table cells, so the marked
copy uses colour-only marking.

## Conventions

- **No `Co-Authored-By` lines in commits.** No AI/Claude disclosure in commit messages.
  (The manuscript itself carries a proper AI-assistance disclosure in the Acknowledgement.)
- **Auto-push after every commit** with `git push origin main`.
- **Concise progress updates with tables.**
- **Verify data/code before claiming facts.** Memory is point-in-time; check `git log`,
  file contents, run scripts where needed.

## Verified external facts (do not change without re-verifying source)

- nablaColors (Potapov 2026): 26,369 pairs, best model **UniProp**, RMSE 27.2.
- ChemBERTa: `seyonec/ChemBERTa-zinc-base-v1`, pretrained on **ZINC** (not 77M PubChem).
- Liu 2023: MTBG = BiGRU + GraphSAGE hybrid; we use the BiGRU component only.
- Mamede 2021: Scientific Reports, first author **Florbela**.
- UV-adVISor 2021: authors **Urbina, Batra, Ekins** et al., volume 93.
- Lupo Pasini 2023: authors **Mehta, Yoo, Irle**, page 546.

## v3 cross-validated results (current paper text — Greenman–Song protocol, 18,415 records)

| Model | RMSE (nm) | MAE (nm) | R² |
|-------|-----------|----------|-----|
| RF + Morgan FP | 31.50 ± 1.47 | 15.37 ± 0.22 | 0.914 |
| Chemprop (D-MPNN) | 33.15 ± 3.27 | 18.11 ± 3.02 | 0.904 |
| Chemprop (tuned) | 28.83 ± 1.32 | 13.19 ± 2.16 | 0.928 |
| XGBoost + Morgan FP | 33.92 ± 1.26 | 20.25 ± 0.41 | 0.900 |
| BiGRU (tuned, 3L/256u) | 36.20 ± 1.20 | 18.81 ± 0.58 | 0.886 |
| ChemBERTa (pretrained) | 54.09 ± 3.57 | 26.78 ± 2.35 | 0.744 |

RF vs. default Chemprop: **p = 0.16** (paired test, not significant).
Tuned Chemprop vs. RF: paired *t* p = 0.044 but Wilcoxon p = 0.125 — borderline at n = 5.

Wetlab (16 molecules × 2 solvents): Chemprop MAE 26.2, ChemBERTa 26.3, BiGRU 28.6, RF 38.5.

> Note: the **v2** numbers (RF 31.34, Chemprop 31.69, ChemBERTa 50.39) appear in `CLAUDE.md`
> and older notes. The paper now reports **v3**. Table 2 keeps two v2 rows (default BiGRU,
> MAE-loss XGBoost), flagged with a dagger footnote.
