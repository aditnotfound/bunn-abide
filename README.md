# BuNN ABIDE-I operator audit

This repository contains a reproducible comparison of identity propagation,
GCN aggregation, trivial-bundle diffusion, learned-local capacity, and learned
Bundle Neural Network (BuNN) transport on ABIDE-I functional connectomes.

## Study status

The frozen ABIDE-I study and its paper are complete. The analysis retained 754
technically eligible participants from 18 sites and used nested held-out-site
validation. Classical baselines, the five-operator neural experiment,
confirmatory analysis, and the secondary robustness audit all passed their
pre-specified integrity checks before interpretation.

The primary analysis found no detected predictive or representation-preservation
advantage for learned BuNN transport under the specified pipeline. This is a
conditional computational result. It is not evidence about biological bundle
geometry, clinical diagnosis, or BuNN performance outside this setting.

A score-blind ABIDE-II metadata and derivative inventory was completed on
14 August 2026. The external-evaluation gate failed because no complete official
C-PAC `filt_noglobal` AAL-116 ROI-time-series derivative was found for the main
ABIDE-II cohort. No ABIDE-II model was fitted or scored.

The current PDFs are available at `output/pdf/manuscript.pdf` and
`output/pdf/supplement.pdf`.

## Repository layout

- `configs/`: frozen dataset, split, model, analysis, paper, and release contracts.
- `docs/`: protocol, decisions, experiment records, results, and release notes.
- `src/`: shared neural data, operator, model, and training implementation.
- `scripts/`: data preparation, training, auditing, analysis, and paper builders.
- `tests/`: unit, leakage-prevention, integrity, and release-safety tests.
- `paper/`: LaTeX manuscript, supplement, references, and generated inputs.
- `reproducibility/`: frozen result snapshot and evidence inventory.
- `output/pdf/`: visually checked manuscript and supplement.

Raw ABIDE files, participant-linked run artifacts, model checkpoints, cloud
configuration, and credentials are intentionally excluded from version control.

## Reproduce the paper

Python 3.14 was used for the frozen experiment. Install the recorded direct
dependencies in an isolated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.
Then validate the frozen manuscript inputs and run the test suite:

```powershell
python scripts/build_manuscript_inputs.py
python -m pytest -q
```

The manuscript and supplement require a LaTeX distribution with BibTeX. See
`paper/README.md` for the compilation sequence and `docs/release.md` for the
full verification and release boundary.

## Data

The project uses the public ABIDE-I Preprocessed Connectomes Project C-PAC
`filt_noglobal/rois_aal` derivative. The repository records the source URL,
eligibility rules, checksums, site counts, and technical exclusions. It does
not redistribute participant time series or participant-linked predictions.

## Citation and license

Citation metadata is provided in `CITATION.cff`. The repository currently uses
an all-rights-reserved notice. Select an open-source license deliberately before
publishing a reusable code release.
