# R1-13 Step 0: Wetlab OOD Audit

- **Wetlab molecules:** 16
- **Training records (v3):** 18,415
- **Training unique solutes:** 7,846
- **Training unique generic Murcko scaffolds:** 3,013
- **Training unique specific Murcko scaffolds:** 4,054

## Nearest-Train Tanimoto Similarity Bins
- Near-duplicate (>0.85): **1**
- High similar (0.7--0.85): **0**
- Some similar (0.5--0.7): **10**
- Dissimilar (<=0.5): **5**
- Tanimoto: mean=0.575, median=0.581, min=0.350, max=1.000

## Bemis-Murcko Scaffold Match
- Exact SMILES duplicate of a train molecule: **1** / 16
- Generic scaffold present in train: **15** / 16
- Specific scaffold present in train: **15** / 16

## Per-Molecule Table

| Molecule | lambda_EtOH | lambda_MeOH | Exact dup | Gen-scaf in train | Spec-scaf in train | Nearest Tan | Nearest train SMILES | Nearest train lambda_max |
|---|---|---|---|---|---|---|---|---|
| Amiloxate | 309 | 298 | N | Y | Y | 0.435 | `CCOC(=O)/C=C/c1ccc(N(C)C)cc1` | 362.5 |
| Avobenzone | 358 | 358 | N | N | N | 0.379 | `COc1ccc(OC)cc1` | 286.2 |
| Caffeic Acid | 325 | 325 | N | Y | Y | 0.636 | `COc1cc(/C=C/C(=O)O)ccc1O` | 287.0 |
| Cinnamic Acid | 275 | 276 | N | Y | Y | 0.571 | `CN(C)c1ccc(/C=C/C(=O)O)cc1` | 355.64 |
| Coumaric Acid | 310 | 309 | N | Y | Y | 0.586 | `CN(C)c1ccc(/C=C/C(=O)O)cc1` | 355.64 |
| Dioxybenzone | 283 | 283 | N | Y | Y | 0.618 | `COc1ccc(C(=O)[O-])c(O)c1` | 295.0 |
| Ferulic Acid | 324 | 324 | Y | Y | Y | 1.000 | `COc1cc(/C=C/C(=O)O)ccc1O` | 287.0 |
| Homosalate | 308 | 308 | N | Y | Y | 0.350 | `O=C(O)c1ccccc1O` | 295.0 |
| Octinoxate | 309 | 308 | N | Y | Y | 0.667 | `CCCCC(CC)COC(=O)/C=C/c1c(OC)ccc(OC)c1OC` | 322.5 |
| Octisalate | 308 | 308 | N | Y | Y | 0.491 | `CCCCC(CC)COC(=O)/C=C/c1ccccc1OC` | 316.0 |
| Octocrylene | 301 | 302 | N | Y | Y | 0.458 | `CCCCC(CC)COC(=O)/C=C/c1ccccc1OC` | 316.0 |
| Oxybenzone | 287 | 287 | N | Y | Y | 0.600 | `COc1ccc(C(=O)[O-])c(O)c1` | 295.0 |
| PABA | 291 | 290 | N | Y | Y | 0.591 | `CC(=O)c1ccc(N)cc1` | 329.4 |
| Padimate O | 310 | 310 | N | Y | Y | 0.575 | `CCOC(=O)c1ccc(N(C)C)cc1` | 303.71 |
| Sinapic Acid | 324 | 324 | N | Y | Y | 0.676 | `COc1cc(/C=C/C(=O)O)ccc1O` | 287.0 |
| Sulisobenzone | 287 | 286 | N | Y | Y | 0.564 | `COc1ccc(C(=O)[O-])c(O)c1` | 295.0 |
