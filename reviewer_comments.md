# JCIM Reviewer Comments — Manuscript ID ci-2026-009433

**Decision:** Major Revisions
**Deadline:** 17-Jun-2026
**Editor:** Prof. John Kitchin
**Date received:** 20-May-2026

## Manuscript title (original)
"When Do Simple Models Win? Machine Learning Architectures for UV Absorption Prediction and Insights for General Molecular Property Prediction"

Authors: Arampath, U.; Pero, B.; Stewart, D.; Demirjian, D.

---

## Required deliverables on resubmission

(a) Point-by-point response to the reviewers → `response_to_reviewers.tex`
(b) Clean, unmarked revised manuscript → `benchmark_paper_jcim_revised.tex`
(c) Marked copy with tracked/highlighted changes → `benchmark_paper_jcim_revised_marked.tex` (uploaded as "Supporting Information for Review")
(d) Clean, unmarked Supporting Information for Publication → `supporting_information_revised.tex`

Plus: cover image (optional, 80-word caption), ORCIDs for all authors, funding sources confirmed.

---

## Reviewer 1

### R1-1 (overgeneralization — title/abstract/conclusion)
The authors make some claims about the conclusions from this work being extensible to molecular property prediction in general (e.g. in the title, pg. 1 line 35 in abstract, pg. 39 line 34 in conclusion). While this work seems to give evidence to the authors' hypothesis regarding the locality of target properties, I think these statements go too far given that the work does not validate this hypothesis in any domains other than optical properties. The work frames UV absorption as "an ideal benchmarking target" (pg. 3 line 25), and while the stated reasons do indeed make UV absorption a particularly interesting benchmarking task, I'm not sure that it's appropriate to say that it's "ideal" in the sense that conclusions that we draw from benchmarking on UV absorption can necessarily be applied more generally in molecular property prediction.

### R1-2 ("RF is optimal" not statistically supported)
The claim that RF is "optimal" (pg 1 line 51 in abstract) is hurt by the fact that the difference between RF and Chemprop is not statistically significant.

### R1-3 (3D methods typo)
Pg. 3 line 39: "3D methods offering advantages primarily for structurally diverse datasets (see Cross-Dataset Benchmarking)" — Is this a typo referring to 3D methods? This work doesn't use 3D methods, as discussed in the limitations and future work sections.

### R1-4 ("Architecture-independent solvent encoding" mischaracterization)
Pg. 4 line 33: I'm not sure if it's appropriate to say "Architecture-independent solvent encoding" when some architectures use Morgan fingerprints to represent solvents and others learn the representation. If I understand correctly, the only thing that's architecture independent is that the solvent representation is simply concatenated with the solute representation rather than using some more complicated process to combine them.

### R1-5 (non-local compounds subset analysis)
The idea that absorption is a locally determined property is right to a first approximation, but in some cases (e.g. longer-wavelength dyes, donor-acceptor or donor-acceptor-donor dyes, etc.) global effects, charge transfer, and 3D geometry become more important. It would be insightful if the authors could identify any compounds in their dataset where non-local effects would be expected to be important, and see if these results lend more support to their claims about locality and architecture choice.

### R1-6 (cite Mayr 2022 for prior solvent encoding work)
Page 8 line 41: "but systematic evaluation of solvent encoding strategies for UV absorption prediction has been limited." — Prior work benchmarked 4 different solvent encoding strategies for UVVis absorption (https://doi.org/10.1039/d1sc05677h)

### R1-7 (Deep4Chem overlap with Joung; why these primary datasets)
The Deep4Chem dataset isn't a great "external dataset" given that it substantially overlaps with the Joung dataset that makes up part of the primary dataset. Why were the Joung and Beard datasets chosen to make the primary dataset as opposed to the many other UV absorption datasets that exist (see Table 3 here: https://doi.org/10.1039/D5DD00402K)?

### R1-8 (duplicate handling: Mayr 2022 protocol)
Why was the first occurrence retained when duplicates were identified? Prior work (https://doi.org/10.1039/d1sc05677h and https://doi.org/10.1039/D5DD00402K) has taken a more robust approach by dropping all entries with large discrepancies (> 5 nm) while taking the average for small discrepancies (< 5 nm).

### R1-9 (64/16/20 split rationale + chromophore-solvent data leakage)
Why was 64/16/20 chosen as the split fractions? These numbers are a bit unusual — did these simply fall out of the solvent stratification process? Does this splitting strategy address the data leakage concerns raised by prior work (https://doi.org/10.1039/d1sc05677h) for the same chromophore appearing in both train and test sets in different solvents?

### R1-10 (move architecture equations to SI)
The detailed description of the architectures on pages 11-19, especially the equations, are probably not necessary to include in the main body of the paper. These may be better placed in the supporting information since the architectures are well-documented in other work, and the core messages of the paper do not depend on any of these mathematical details.

### R1-11 ("We adapt" → "We use" for Chemprop)
pg. 18 line 20: I think "We adapt" should be "We use" since this capability is already present in Chemprop out of the box and has been used for solute-solvent prediction for optical and other properties in many prior works.

### R1-12 (D-MPNN default hyperparameters fairness)
Why are all D-MPNN hyperparameters kept at their published defaults (pg 18 line 26)? Is this a fair comparison if not all architectures have been optimized?

### R1-13 (wetlab OOD: scaffold/dye-family similarity)
I find it surprising that ChemBERTa performs substantially better (32.2 nm vs. 50.39 nm) on the "out-of-distribution" experimental validation vs. the in distribution testing. To me, this raises concerns about whether these compounds are actually "out-of-distribution" for ChemBERTa. Additionally, I don't think the criterion used here are sufficient for truly testing out-of-distribution performance ("Of these 16 molecules, 15 are entirely absent from the training dataset, providing a genuine out-of-distribution test. One molecule (Ferulic Acid) appears in the training data with a different solvent (water)." - pg. 32 line 10). Prior work (https://doi.org/10.1039/d1sc05677h) demonstrated that splitting based on scaffold is a more rigorous test than simply on chromophore identity. If the training set contains other compounds from the same dye families or sharing the same scaffolds as the compounds that are experimentally validated, then this is not really "out-of-distribution".

### R1-14 (solvent-effect metrics on multi-solvent chromophores only)
In the analysis on the effect of solvent information, should these metrics take into account all chromophores, or only those for which measurements are present in multiple solvents?

### R1-15 (Mamede/Reaxys reproducibility)
For the reconstruction of the Mamede dataset, will this dataset be provided along with the code in this publication, or is that not permitted by the Reaxys license? If the data can't be provided directly, I think a more detailed description of the query process may be necessary so that others can reproduce this part of the work.

### R1-16 (ChemBERTa 600 nm ceiling in parity plot)
The behavior of ChemBERTa shown in the parity plot in Figure 5 is strange for high wavelengths. This seems quite unusual to have what appears to be a "ceiling" where many compounds that have experimental wavelengths above 600 are all predicted to be about 600nm. This seems to be the reason behind the very high RMSE for ChemBERTa due to outliers while the MAE remains at a more expected level. Do the authors have an explanation for this behavior?

### R1-17 (lipophilicity miscategorized as optical property)
Pg 40 line 42: lipophilicity is incorrectly listed as an optical property

### R1-18 (code public during peer review)
Code should be available during peer review, not just at the time of publication. This is critical for assessing the reproducibility of the work.

### R1-19 (cite Chemprop v2 paper)
Per the guidelines from the Chemprop GitHub, work that uses chemprop v2 should cite the paper about v2 (https://doi.org/10.1021/acs.jcim.5c02332) in addition to the original 2019 paper that describes the theory behind chemprop.

---

## Reviewer 2

### R2-1 (experimental λ_max uncertainty)
Could the authors provide more information about the accuracy of the experimental λ_max values? In particular, it would be useful to know the expected measurement uncertainty and, where comparable measurements exist across sources, the inter-laboratory or inter-dataset variability. This would help contextualize the reported model errors.

### R2-2 (Joung+Beard dataset overview figure)
Could the authors provide a more detailed overview of the Joung+Beard dataset? A figure showing the distribution of λ_max values and the distribution of solvents would be helpful. In addition, a small set of representative solute structures would make the chemical diversity of the dataset easier to assess.

### R2-3 (MoleculeNet inclusion plans)
Are there plans to include this benchmark, or a cleaned version of it, in MoleculeNet or a similar standardized benchmark suite? This would make future comparisons with additional models easier and more reproducible.

### R2-4 (solvatochromism: per-solute solvent shifts)
The manuscript shows that including solvent information improves performance, but I would appreciate a deeper analysis of solvatochromism. For example, are there solutes in the dataset with experimental measurements in multiple solvents, and if so, how large are the observed solvent-induced shifts?

### R2-5 (polarity-grouped solvent variability)
Even when repeated experimental measurements across solvents are unavailable, the authors could analyze model-predicted solvent sensitivity more directly. For instance, solvents could be grouped by polarity or related physicochemical properties, and the variability of predictions across solvent groups could be reported.

### R2-6 (D-MPNN missing from Figure 2)
Figure 2 appears to compare RF, BiGRU, and ChemBERTa, but the D-MPNN/GNN model seems to be missing. Since the D-MPNN is one of the central models in the study and performs very well, it would be helpful to include it in the architecture comparison figure or explain why it is omitted.

### R2-7 (D-MPNN 636K params vs dataset size)
The D-MPNN has 636k trainable parameters. Could the authors comment on whether this model size is appropriate relative to the size of the training dataset? A short discussion of overfitting controls, such as early stopping, normalization, and default Chemprop regularization, would help justify the model capacity.

### R2-8 (LLM/foundation-model extensions)
Could this work be extended to modern large language models or general-purpose molecular foundation models? For example, it would be interesting to discuss whether GPT-like or Claude/Opus-style models could contribute through SMILES featurization, chemical reasoning, data curation, or as part of an agentic workflow, even if they are not directly used as regression models.

### R2-9 (pretrained molecular GNN foundation models)
Are there pretrained or foundational graph neural network models for molecules that could be adapted to this task by adding a new readout head for UV absorption prediction? This might provide a useful comparison to ChemBERTa as the pretrained sequence-model baseline. This point is also related to point 7.

### R2-10 (Figure 5 axis limits)
In Figure 5, the lower limits of the x- and y-axes appear to extend to approximately -200 nm. Since the physically relevant and observed λ_max values are positive and much closer to the 200--900 nm range, would it be more informative to set the axis limits closer to the data range?
