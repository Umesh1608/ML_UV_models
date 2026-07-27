# ci-2026-009433.R1 — Decision-Letter Compliance Map

One-to-one map from Prof. Kitchin's decision letter to where each item was addressed.
Page numbers refer to the PDFs in `jcim_secondrevision_submission/`.

**Audited 27 July 2026** (submission deadline). 15 of 19 items complete and verified;
4 require an author action in ACS Paragon Plus.

Browsable version: https://claude.ai/code/artifact/6d89e3db-dffd-4213-9068-945c92b483e1

---

## Needs your action (cannot be done from the LaTeX sources)

| Item | What is left |
|------|--------------|
| **ORCID** | Each author validates their own iD on their ACS user profile. Nothing goes in the manuscript. |
| **Funding (form half)** | Manuscript declares no specific grant funding (p. 48). ACS wants the same declaration in the submission form. |
| **English-language editing** | The one item with **no response anywhere** in the package. Needs your factual call — see below. |
| **Cover art** | Optional and opt-in. Nothing prepared; declining costs nothing. |

---

## Editor's four technical issues (Response letter Part A, pp. 2–3)

### E1 — Table 5 caption recovery %
Caption still carried the **93.6% recovery** figure retracted in round 1 (R1-15). The Mamede set
is read from the published SI, not re-queried from Reaxys; RDKit canonicalises 74,783 of 74,784
entries (~100%). Main text was fixed in round 1, caption was missed. Now corrected.

- Clean manuscript **p. 35** (Table 5 caption) · Marked **p. 41**
- Response letter **p. 2** · SI **p. 33** (Section S17, full provenance)

### E2 — Regenerate Figures 5, 6, 9
All three had been carried over from v2 and never regenerated after the v3 (Greenman–Song)
retrain. All now from the same v3 predictions underlying Table 2. The response explicitly
distinguishes **pooled-prediction RMSE** (Figs 5, 6) from **across-fold mean RMSE** (Table 2,
Fig 9) — they agree to within 0.2 nm for every model.

- Clean manuscript **p. 28** (Figs 5, 6), **p. 31** (Fig 9) · Marked **pp. 33, 34, 37**
- Response letter **p. 2** · Scripts `make_parity_plot.py`, `regenerate_figures.py`

### E3 — Abstract vs. Methods on hyperparameter tuning
Abstract was the inaccurate one: RF's tree count and feature fraction came from a
432-configuration grid search. Reworded in three places, preserving the compute-cost contrast
(lightweight CPU grid search, not GPU-based HPO).

- Clean manuscript **p. 1** (abstract), **p. 4** (contributions bullet),
  Model Interpretability and Practical Considerations section
- Marked **p. 4** · Response letter **p. 3**

### E4 — Figure 10 vs Table 6
Real inconsistency: Fig 10 used the **tuned** BiGRU ensemble, Table 6 the **default** one
(old caption Octocrylene +61 nm vs Table 6's +43 nm). Fig 10 regenerated from the default
5-fold ensemble; per-molecule predictions now match Table 6 exactly and ensemble MAE 28.6 nm
matches Table 7. Caption thresholds updated (<22→<20 nm; >28→>25 nm).

- Clean manuscript **p. 34** (Fig 10) vs **p. 38** (Tables 6, 7) · Marked **p. 40** vs **p. 44**
- Response letter **p. 3**

---

## Submission requirements

### TOC graphic — DONE
On the **last page** (p. 57), **no caption** (achemso `tocentry` prints only the standard
heading), artwork 975 × 525 px = **exactly 3.25 × 1.75 in at 300 dpi** (ACS minimum).
Verified as a genuinely embedded 300 ppi image, not a placeholder.

- Source `benchmark_paper_jcim_revised.tex` lines 59–61 · artwork `results/toc_graphic.png`

### Four required files (a–d) — DONE
All four built and verified, plus cover letter. Every document compiles with **zero undefined
references or citations**. Clean manuscript, marked manuscript and SI reproduce **byte-for-byte**
from committed sources — proving none is a stale build. See manifest at the bottom.

### Funding — MANUSCRIPT DONE, FORM PENDING
Acknowledgement declares no specific grant from public, commercial or not-for-profit agencies.
A nil declaration satisfies the requirement. **You must also confirm it in the submission form.**

- Clean manuscript **p. 48** · Marked **p. 55** · Cover letter **p. 2**

### ORCID — YOUR ACTION
Submission-system task only; iDs attach to ACS user profiles, never the manuscript file, so
there is correctly nothing to change in the LaTeX. Each author validates their own.
The cover letter (p. 2) already states these are provided through the system — make it true
before submitting.

### English-language quality — NEEDS YOUR CALL
> "The reviewers indicate that the quality of the written English could be improved."

**No response exists anywhere in the package.** It sits in the standard block of the letter
rather than either reviewer's numbered comments, and neither reviewer raised language in their
own text, so it is plausibly boilerplate — but it was explicitly asked, and one sentence in the
cover letter closes it.

I did not draft that sentence: it is a factual claim about what you did. Two honest options —
(a) state the manuscript was reread and copy-edited for clarity and grammar during this revision,
which the prose-cleanup work supports; or (b) if you used a service or native-speaker colleague,
name them. Do not claim a service that was not used.

- Would go in `cover_letter_revised.tex`, after the Reviewer 1 paragraph, before the file list.

### Cover art — OPTIONAL, NOT SUBMITTED
Invitation, not requirement ("if you have"). Nothing prepared. The TOC graphic would be the
natural starting point but needs re-rendering to cover spec plus an 80-word caption, uploaded
under *Cover Art* / *Cover Art Caption*.

### iThenticate — NO ACTION
Automatic ACS screening. A revision legitimately overlaps its own earlier version.

---

## Reviewer 1's seven suggestions (Response letter Part A, pp. 4–7) — all complete

### 1. Paraphrased reviewer comments
Every comment block in Part B replaced with exact decision-letter text. The requested
double-check found **more than the reviewer spotted**: R2-1, R2-2, R2-4, R2-5, R2-6, R2-7 were
also paraphrased and were likewise restored, and the response says so rather than quietly fixing
them. The wider audit reports the only substantive inaccuracies as the two already surfaced
(the "variance intrinsic to Chemprop" claim, corrected via the R1-12 HPO probe; the 93.6%
Reaxys recovery claim, corrected in R1-15). AI assistance is confirmed directly, cross-referenced
to point 5.

- Response letter **p. 4** (Comment 1) · **p. 8+** (Part B verbatim) · **p. 33** (Reviewer 2)
  · **p. 40** (Summary of Major Changes)

### 2. SI sections not numbered
Now **21 numbered sections, S1–S21**. All 28 `Section S<n>` references in the manuscript were
independently checked against the SI section list — every one resolves to the topically correct
section, no off-by-one drift.

- `supporting_information_revised.tex` line 44: `\renewcommand{\thesection}{S\arabic{section}}`
- Response letter **p. 4**

### 3. v2/v3 definitions
Both halves addressed: defined at first use, and given descriptive names rather than bare version
numbers — v2 = 18,755-row first-occurrence-dedup; v3 = 18,415-row **Greenman–Song** protocol.
The abstract now names the protocol instead of saying "v3".

- Clean manuscript **p. 1** (abstract), **p. 21** (Table 2 dagger footnote flags the two v2 rows)
- SI **p. 16** (Section S10, v2-vs-v3 comparison) · Response letter **p. 5**

### 4. Inter-revision phrasing in captions
Both flagged captions fixed. The Table S24 "missing word" was a **broken citation** rendering as
an empty reference, not a typo — that citation is repaired, so the caption is complete rather than
just reworded. Swept both manuscript and SI: zero remaining revision-relative phrasing.

- SI Table S16 and Table S24 captions · Response letter **p. 5**

### 5. AI-assistance disclosure
Broadened rather than defended. Now covers **code development and debugging, data-analysis
scripting, and editing of the manuscript and response-to-reviewer text** — explicitly covering the
manuscript writing the reviewer inferred from the commit history and CLAUDE.md. Authorship
boundary stated in the same sentence.

- Clean manuscript **p. 48** (Acknowledgement) · Marked **p. 55** · Response letter **p. 6**

### 6. Repo subdirectories + LICENSE
New dedicated repo **ML_UV_models_jcim**, verified live: public, root has four directories
(`data`, `results`, `scripts`, `uvml`) plus README/pyproject/requirements/gitignore, and a
**LICENSE file GitHub detects as MIT**. The loose .tex files and images are out of the root.
Original dev repo stays public for the commit history; the manuscript cites both and says why.

- Clean manuscript **p. 48** (Data and Software Availability) · Response letter **p. 6**
- Verified live: `ML_UV_models_jcim` 200 · `ML_UV_models` 200 · Zenodo 10.5281/zenodo.20600225 200

### 7. Figure 2(b) parallel vs sequential
Panel redrawn with solute and solvent encoders as genuine parallel branches converging on a
concatenation. Caption tightened to match; closing sentence attributes panel (b)'s solvent
handling to the parallel-encoder multicomponent configuration.

- Clean manuscript **p. 14** (Figure 2b) · Marked **p. 15** · Response letter **p. 7**

---

## Caught during today's audit (not requested by anyone)

**The response letter had a broken bibliography.** The staged PDF was compiled without a
complete BibTeX pass: **14 citations rendered as `[?]`** — including Greenman, Song and
Chemprop v2, which several answers depend on — and **the References page was missing entirely**.
Since Reviewer 1's first point was about citation and quoting rigour in exactly this document,
sending it that way would have read badly.

Rebuilt from source: all 14 resolve, the 9-entry References list is restored, 45 → 46 pages.
Old vs new diffed to confirm nothing else changed. Cover letter rebuilt at the same time so both
carry the 27 July date.

**Carry forward:** grepping for `??` finds undefined *references* but not undefined *citations*,
which render as `[?]`. The first scan came back clean and missed this; it surfaced only on
recompiling from source and diffing against the staged file.

---

## Reading the marked copy

Two rounds in two colour pairs, so the editor can isolate what is new:

| Colour | Meaning | Count |
|--------|---------|-------|
| Blue | Round-1 additions | 126 |
| Red | Round-1 deletions | 85 |
| **Green** | **Round-2 additions** | **13** |
| **Orange** | **Round-2 deletions** | **10** |

Colour-only rather than strikethrough, because soul's `\st` cannot span paragraph breaks or
table cells and would have broken several edited captions.

---

## Upload manifest — `jcim_secondrevision_submission/`

| File | ACS designation | Pages | State |
|------|-----------------|-------|-------|
| `When_do_simple_models_win_revised.pdf` | Manuscript | 57 | Verified, unchanged |
| `When_do_simple_models_win_revised_marked.pdf` | Supporting Information for Review Only | 63 | Verified, unchanged |
| `When_do_simple_models_win_response_to_reviewers.pdf` | Response to Reviewers | 46 | **Rebuilt today** |
| `When_do_simple_models_win_supporting_information.pdf` | Supporting Information for Publication | 47 | Verified, unchanged |
| `When_do_simple_models_win_cover_letter.pdf` | Cover Letter | 2 | **Rebuilt today** |
