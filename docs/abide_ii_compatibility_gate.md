# ABIDE-II compatibility gate

Status: **not passed; no ABIDE-II model evaluation is authorized.**

## Purpose

ABIDE-II can strengthen the paper only if it supports a prospective external
evaluation that is close enough to the frozen ABIDE-I representation. It must
not become a second development set or a loosely matched dataset added for a
larger sample count.

## What the official sources currently establish

- The official ABIDE-II page distributes phenotypic files, quality metrics,
  site-specific raw imaging archives, and acquisition information.
- The main Preprocessed Connectomes Project ABIDE download page documents the
  ABIDE-I public S3 derivatives, including C-PAC `filt_noglobal` AAL ROI time
  series. It does not document a matching full ABIDE-II derivative collection.
- A separate LLE repository lists several ABIDE-II sites preprocessed with
  C-PAC defaults, but this is not yet evidence that the data match the frozen
  C-PAC strategy, filtering, nuisance regression, spatial normalization, AAL
  extraction, and quality-control contract used in Study 1.

The current evidence therefore does not justify launching an ABIDE-II test.

## Pass criteria

All criteria must be answered before labels or model scores are inspected:

1. **Provenance:** every file comes from an official ABIDE-II or documented
   preprocessing repository and maps unambiguously to the official phenotype
   record.
2. **Representation:** every accepted participant has a 116-column AAL ROI
   time series with a documented atlas and orientation matching Study 1.
3. **Preprocessing:** filtering, global-signal treatment, nuisance regression,
   registration, temporal censoring, and temporal units are documented well
   enough to classify differences from the ABIDE-I C-PAC `filt_noglobal`
   derivative.
4. **Quality control:** diagnosis, site, scan identity, usable length, finite
   values, and ROI variance can be checked without outcome-based exclusions.
5. **Independence:** any participant or longitudinal scan already represented
   in ABIDE-I is excluded or handled under a frozen overlap rule.
6. **Evaluation:** Study 1 code, weights or refitting rule, feature order,
   densities, metrics, and uncertainty procedure are frozen before ABIDE-II
   labels are evaluated.
7. **No adaptation on test labels:** ABIDE-II diagnosis labels cannot choose
   preprocessing, exclusions, hyperparameters, epochs, checkpoints, or model
   variants.

## Feasibility sequence

1. Download only the official phenotype legend, composite phenotype table,
   quality metrics, and site inventory.
2. Build a file-level inventory without diagnosis-based filtering.
3. Select ten scans across several sites using identifiers only.
4. Produce AAL-116 time series for the sample under one documented pipeline,
   or locate an official derivative that already supplies them.
5. Run only technical checks: shape, finite values, ROI variance, scan length,
   orientation, and connectome construction.
6. Record every mismatch against the frozen ABIDE-I contract.
7. Pass the gate only if the mismatches support a scientifically interpretable
   external evaluation. Otherwise stop and report incompatibility.

## Evaluation contract if the gate passes

- Keep Study 1 as the primary result.
- Treat ABIDE-II as one prospective external transportability study.
- Evaluate a minimal frozen set: connectome elastic net, GCN, and learned BuNN.
- Report ordinary and balanced accuracy, AUROC, sensitivity, specificity,
  sample coverage, and site-level results.
- Do not retune to recover a favorable result.
- Label material preprocessing differences and avoid calling the result an
  exact replication.

## Current decision

No compute job should start yet. The next authorized ABIDE-II action is the
metadata and derivative-availability inventory. Model training or score
generation remains blocked by this gate.

