# Bundle Neural Networks on ABIDE-I

Held-out-site comparison of identity propagation, GCN aggregation, trivial-bundle
diffusion, learned-local capacity, and learned Bundle Neural Network (BuNN)
transport on ABIDE-I functional connectomes.

The analysis used 754 technically eligible participants from 18 sites. Learned
BuNN transport had no detected predictive or representation-preservation
advantage under this pipeline. That is a conditional computational result. It
is not a claim about biological bundle geometry, clinical diagnosis, or BuNN
outside this setting.

## Papers

- IEEE-style manuscript: [`output/pdf/manuscript.pdf`](output/pdf/manuscript.pdf)
- Supplement: [`output/pdf/supplement.pdf`](output/pdf/supplement.pdf)
- Longer evidence report: [`output/pdf/yhsa-submission/Comp-183-Research Report.pdf`](output/pdf/yhsa-submission/Comp-183-Research%20Report.pdf)

## Setup

Python 3.14 was used for the frozen experiment. CPU-only wheels are enough for
the integrity tests. Use the matching CUDA PyTorch wheel only if you repeat
model fitting on a GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

On Linux or macOS, activate with `source .venv/bin/activate`. You can also
install the pinned wheels from `requirements.txt`.

Then rebuild the manuscript inputs and run the tests:

```powershell
python scripts/build_manuscript_inputs.py
python -m pytest -q
```

Compile `paper/manuscript.tex` and `paper/supplement.tex` with a LaTeX engine
that supports BibTeX. See `paper/README.md`.

## Layout

- `src/`: operators, models, and training used by the study
- `scripts/`: data prep, training, audits, analysis, and paper builders
- `tests/`: leakage, integrity, and release-safety checks
- `configs/`: frozen dataset, split, model, and analysis contracts
- `paper/`: manuscript, supplement, and generated tables/figures
- `reproducibility/`: result snapshot and evidence hashes
- `docs/`: protocol, claim ledger, and release boundary
- `output/pdf/`: compiled papers

Raw ABIDE files, participant-linked run artifacts, checkpoints, and cloud
credentials are not in this repository.

## Data

The study uses the public ABIDE-I Preprocessed Connectomes Project C-PAC
`filt_noglobal/rois_aal` derivative. Source URLs, eligibility rules, checksums,
and site counts are recorded here. Time series and participant-linked
predictions are not redistributed.

A score-blind ABIDE-II inventory found no complete official C-PAC
`filt_noglobal` AAL-116 ROI-time-series derivative for the main ABIDE-II
cohort, so no ABIDE-II model was fitted.

## Citation and license

See `CITATION.cff`. The repository uses an all-rights-reserved notice. Choose
an open-source license deliberately before treating this as a reusable library.
The public ABIDE data and third-party packages remain under their own terms.
