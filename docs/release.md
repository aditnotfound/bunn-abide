# Reproducibility and release boundary

## Completion state

The ABIDE-I experiment, confirmatory analysis, robustness audit, manuscript,
and supplement are complete. The release package reproduces the frozen paper
from aggregate, hash-bound evidence. It does not repeat model fitting.

## Included in the public-safe package

- source code and tests;
- frozen protocol and configuration files;
- aggregate result snapshot and evidence hashes;
- generated paper tables and figures;
- manuscript and supplement source;
- checked PDFs; and
- project, decision, experiment, and AI-use records.

The private artifact manifest remains included because the manuscript contract
binds its digest. It records filenames, byte counts, and SHA-256 hashes, but it
does not contain the underlying participant-linked files.

## Deliberately excluded

- raw or processed participant data;
- participant identifiers, predictions, and fold-level run directories;
- the sealed 1 GB neural training archive;
- model checkpoints and MLflow data;
- SSH keys, cloud credentials, `.env` files, and `.run-control` settings;
- local absolute paths, IP addresses, AWS account identifiers, and notification
  destinations.

These exclusions preserve privacy and keep the release small. A researcher who
wants to repeat training must retrieve ABIDE-I from the recorded public source
and rebuild the local data artifacts using the documented scripts.

## Deterministic verification

Run the following commands from the repository root:

```powershell
python scripts/build_manuscript_inputs.py
python -m pytest -q
python scripts/build_release_package.py
```

The release builder scans every included text file for private paths,
credentials, IP addresses, notification addresses, and AWS resource identifiers.
It then writes a deterministic ZIP archive and a SHA-256 manifest under
`output/release/`.

Compile `paper/manuscript.tex` and `paper/supplement.tex` after regenerating
their inputs. A final audit must render every PDF page and check citations,
references, figures, tables, headers, footers, and page boundaries.

## Owner sign-off before external publication

The repository can be archived locally without further scientific work. Before
an external release, the owner must verify the author name and affiliation,
choose whether to replace the all-rights-reserved notice with an open-source
license, and decide whether a venue-specific template or anonymized version is
needed. These are ownership and publication decisions, not unresolved analysis.
