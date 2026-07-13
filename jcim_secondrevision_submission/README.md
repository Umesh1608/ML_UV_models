# Second Revision Submission — ci-2026-009433.R1 (deadline 27 July 2026)

Files for the **second-revision** submission to ACS Paragon Plus.

## PDFs to upload
- `When_do_simple_models_win_revised.pdf` — (b) clean, unmarked manuscript
- `When_do_simple_models_win_revised_marked.pdf` — (c) marked manuscript (upload as *Supporting Information for Review Only*)
- `When_do_simple_models_win_response_to_reviewers.pdf` — (a) point-by-point response
- `When_do_simple_models_win_supporting_information.pdf` — (d) clean Supporting Information for Publication
- `When_do_simple_models_win_cover_letter.pdf` — cover letter

## LaTeX sources
- `benchmark_paper_jcim_revised.tex`, `benchmark_paper_jcim_revised_marked.tex`
- `supporting_information_revised.tex`
- `response_to_reviewers.tex`
- `cover_letter_revised.tex`
- `Proposal.bib`

Compile with `latexmk -pdf <file>.tex` from inside this folder; figures resolve to
`../results/` via `\graphicspath`.

## What changed vs. the first revision
Addresses the editor's four technical issues (E1 Table 5 caption; E2 regenerate
Figures 5/6/9; E3 abstract/methods tuning wording; E4 Figure 10 ↔ Table 6 consistency)
and Reviewer 1's seven second-round suggestions (verbatim reviewer quotes; numbered SI
sections; v2/v3 definitions; caption cleanup; AI-disclosure update; LICENSE + repo
cleanup; parallel Figure 2(b)). See the response letter (**Part A**) for details.

The first-revision submission is preserved unchanged in `../jcim_firstrevision_submission/`.
