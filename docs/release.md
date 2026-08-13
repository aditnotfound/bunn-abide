# Reproducibility and release boundary

## Completion state

The ABIDE-I experiment, confirmatory analysis, robustness audit, E1 checkpoint
intervention study, E2 synthetic mechanism study, manuscript, supplement, and
final evidence report are complete. The release package reproduces the papers
from aggregate, hash-bound evidence. It does not repeat model fitting.

## Included in the public-safe package

- source code and tests;
- frozen protocol and configuration files;
- aggregate result snapshot and evidence hashes;
- generated paper tables and figures;
- manuscript and supplement source;
- the separate final evidence-report source and PDF;
- E1, E2, and final-report manifests;
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

With the private analysis archives present, the complete suite runs 75 tests.
In a public clone, 72 tests run and three private evidence-regeneration tests
are explicitly skipped. The public suite still verifies the frozen result
snapshot, manuscript inputs, release contents, privacy scan, and deterministic
archive. Regenerating the Step 13.1 paper assets from raw accepted analysis
outputs requires the excluded private evidence package.

The release builder scans every included text file for private paths,
credentials, IP addresses, notification addresses, and AWS resource identifiers.
It refuses to package modified tracked files, then writes a deterministic ZIP
archive and a SHA-256 manifest under `output/release/`. Untracked files are not
eligible for inclusion.

Compile `paper/manuscript.tex` and `paper/supplement.tex` after regenerating
their inputs. A final audit must render every PDF page and check citations,
references, figures, tables, headers, footers, and page boundaries.

## Owner sign-off before external publication

The repository can be archived locally without further scientific work. Before
an external release, the owner must verify the author name and affiliation,
choose whether to replace the all-rights-reserved notice with an open-source
license, and decide whether a venue-specific template or anonymized version is
needed. These are ownership and publication decisions, not unresolved analysis.

## Final clean-checkout audit

Commit `3bdf2ff` was cloned into a separate directory on 2026-08-10. The
hash-bound manuscript inputs rebuilt successfully. The public suite completed
with 72 passes and the three documented private-evidence skips. The private
repository completed all 75 tests. The dependency lock resolved under Python
3.14 without installation changes.

Bundled Tectonic compiled the 8-page manuscript with BibTeX and the 3-page
supplement. Extracted text contained no unresolved citation, reference,
placeholder, or question-mark markers. All 11 pages were rendered and visually
checked for clipping, overlap, broken tables, figure legibility, headers,
footers, and page order. No layout defect was found.
