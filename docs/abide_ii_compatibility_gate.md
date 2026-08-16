# ABIDE-II compatibility gate

Status: **failed on 14 August 2026; ABIDE-II model evaluation is not authorized.**

## Question

ABIDE-II would strengthen the paper only as a prospective external evaluation.
That requires a representation close enough to the frozen ABIDE-I C-PAC
`filt_noglobal` AAL-116 ROI time series to make the comparison interpretable.
ABIDE-II cannot be used as a second development set or added simply to increase
the sample count.

## Reproducible inventory

The score-blind inventory is defined in
`configs/abide_ii_inventory_v1.json` and produced by:

```powershell
python scripts/inventory_abide_ii.py --refresh
```

The command downloads only official release metadata and quality files, hashes
each source, lists the relevant public S3 prefixes, and writes
`reproducibility/abide_ii_gate_inventory.json`. It does not fit a model, create
predictions, or read model scores. Four unit tests cover identifier parsing and
the fail-closed decision rule.

## What the inventory found

- The official main composite phenotype contains 1,114 unique participants
  from 19 sites: 521 ASD and 593 control records.
- The official functional quality file covers 1,043 of those 1,114 participant
  identifiers. It does not contain main-cohort records for SU_2 or U_MIA_1 and
  is missing one IP_1 participant.
- The main ABIDE-II identifiers have no numeric overlap with the ABIDE-I
  phenotype used here.
- The separate longitudinal table contains 38 participants represented at two
  time points. All 38 identifiers occur in ABIDE-I, so that collection is
  frozen out of any independent external cohort.
- The public PCP prefix used by Study 1 contains 1,102 ABIDE-I AAL derivatives
  and zero matching main ABIDE-II identifiers.
- The tested public prefixes `data/Projects/ABIDEII` and
  `data/Projects/ABIDE_II` contain no objects.
- The official LLE area exposes 21 non-mask ABIDE-II site directories. Its
  documentation describes mixed C-PAC and SPM-preprocessed volumes, not the
  frozen C-PAC filtered, no-GSR AAL-116 ROI derivative.

The official ABIDE-II release page provides the composite phenotype, quality
metrics, scan protocols, and site-specific raw archives. The ABIDE-I PCP page
documents the C-PAC strategy and AAL derivative used in Study 1. Neither source
documents a complete matching ABIDE-II ROI-time-series release.

## Gate decision

The gate failed because provenance and phenotype coverage alone do not establish
representation or preprocessing compatibility. Extracting AAL time series from
the LLE volumes would change the pipeline and mix preprocessing families across
sites. Reprocessing the raw scans could be a separate project, but it would
require a versioned C-PAC workflow, raw-data access, subject-level registration
and nuisance-regression checks, and a new pre-specified QC contract. It cannot be
silently treated as the same derivative.

No ten-participant time-series smoke test or AWS evaluation was launched because
the required derivative was absent. This is a gate failure, not a negative
ABIDE-II predictive result.

## Conditions for reopening the gate

The gate may be rerun if either of the following becomes available:

1. an official full ABIDE-II C-PAC `filt_noglobal` AAL-116 ROI-time-series
   derivative with documented file-to-phenotype mapping; or
2. a separately frozen raw-data preprocessing study that reproduces the Study 1
   nuisance, filtering, registration, atlas, censoring, and temporal contracts.

If the gate later passes, the main Study 1 result remains primary. ABIDE-II must
be evaluated once with a minimal frozen set consisting of the connectome elastic
net, GCN, and learned BuNN. No ABIDE-II label may select preprocessing,
exclusions, hyperparameters, epochs, checkpoints, or model variants.

## Official sources checked

- [ABIDE-II release page](https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html)
- [ABIDE-II phenotype directory](https://fcon_1000.projects.nitrc.org/indi/abide2/release/phenotypic_data/)
- [ABIDE-II phenotypic data legend](https://fcon_1000.projects.nitrc.org/indi/abide/ABIDEII_Data_Legend.pdf)
- [ABIDE Preprocessed downloads](https://preprocessed-connectomes-project.org/abide/download.html)
- [ABIDE LLE repository documentation](https://fcon_1000.projects.nitrc.org/indi/abide/LLE_home.html)
