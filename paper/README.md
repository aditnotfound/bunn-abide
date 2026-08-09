# Step 13.2 manuscript package

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
- The manuscript is a scientific scaffold that the author must read, verify,
  and revise in their own voice before external submission.
