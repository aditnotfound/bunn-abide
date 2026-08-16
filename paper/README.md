# Reproducible manuscript package

This directory contains a reproducible paper scaffold compiled from the frozen
Step 13.1 evidence bundle. Scientific values in the manuscript and supplement
are generated; do not edit files under `generated/tex/` or `generated/numbers.tex`
by hand.

## Build

From the repository root:

```powershell
python scripts/build_manuscript_inputs.py
python -m unittest tests.test_build_manuscript_inputs
```

Then compile `paper/manuscript.tex` and `paper/supplement.tex` with a LaTeX
engine that supports BibTeX. Run LaTeX, BibTeX, and LaTeX twice more so all
citations and cross-references resolve. Final checked PDFs are written to
`output/pdf/`.

## Provenance boundary

- `configs/manuscript_v1.json` binds the frozen evidence inputs by SHA-256.
- `paper/generated/manuscript_input_manifest.json` records every generated
  manuscript input.
- `docs/step13_claim_ledger.md` defines allowed and prohibited claims.
- The prose and figures have passed the project editorial and visual audits.
  The owner must still verify the author name and affiliation before any
  external publication.

## Release package

`python scripts/build_release_package.py` creates the public-safe deterministic
archive after the manuscript and tests pass. The archive excludes participant
data, run directories, credentials, checkpoints, and local cloud settings. See
`docs/release.md` for the precise boundary.
