# Step 5: ABIDE-I Manifest Construction

Step 5 turns the public ABIDE-I PCP metadata into a frozen participant manifest.
This manifest is the dataset contract used by all later baselines, graph models,
fold generation, and result tables.

## Frozen derivative path

- Dataset: ABIDE-I via the Preprocessed Connectomes Project.
- Pipeline: `cpac`.
- Strategy: `filt_noglobal`.
- ROI derivative: `rois_aal`.
- Time-series URL pattern:
  `https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_aal/<FILE_ID>_rois_aal.1D`
- Phenotypic/QC source:
  `https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv`

## What this step must produce

- `data/raw/Phenotypic_V1_0b_preprocessed1.csv`
- `data/processed/abide_i_manifest_cpac_filt_noglobal_rois_aal.csv`
- `data/processed/abide_i_manifest_summary.json`
- A recorded manifest SHA-256 hash.
- A site-by-diagnosis count table.
- A clear QC rule decision.

The `data/` outputs are intentionally not committed. The script and summary
hash let the same manifest be regenerated.

## Primary command

```bash
python scripts/prepare_abide_manifest.py
```

Default choices:

- Requires `FILE_ID` to be present and not `no_filename`.
- Requires at least one ASD and one control participant per site, because a
  held-out-site balanced-accuracy score is otherwise undefined.
- The final primary command below selects the QC rule explicitly; do not rely
  on the script's provisional default.
- Does not HEAD-check every ROI time-series URL unless requested.

## URL smoke check

Before downloading all time series, run a quick derivative availability check:

```bash
python scripts/prepare_abide_manifest.py --check-urls --max-url-checks 25
```

For the final manifest audit, run:

```bash
python scripts/prepare_abide_manifest.py --check-urls
```

If URL checks remove rows, record the missing count and inspect examples before
continuing.

## QC rule comparison

The PCP table contains multiple QC columns. Do not silently switch rules. Compare
candidate manifests before freezing the primary rule:

```bash
python scripts/prepare_abide_manifest.py --qc-rule sub_in_smp --summary-output data/processed/summary_sub_in_smp.json
python scripts/prepare_abide_manifest.py --qc-rule rater1_ok --summary-output data/processed/summary_rater1_ok.json
python scripts/prepare_abide_manifest.py --qc-rule func2_ok --summary-output data/processed/summary_func2_ok.json
python scripts/prepare_abide_manifest.py --qc-rule rater1_and_func2_ok --summary-output data/processed/summary_rater1_and_func2_ok.json
```

The chosen QC rule must be added to `docs/decision_log.md` before model
training begins.

## Observed QC comparison and primary decision

On 2026-08-03, using the current PCP phenotype table and requiring an available
`FILE_ID`, the candidate populations were:

| QC rule | Subjects | Sites | ASD | Control |
| --- | ---: | ---: | ---: | ---: |
| `sub_in_smp` | 709 | 17 | 341 | 368 |
| `rater1_ok` | 997 | 20 | 476 | 521 |
| `func2_ok` | 786 | 20 | 386 | 400 |
| `rater1_and_func2_ok` | 776 | 20 | 379 | 397 |

The primary manifest uses `rater1_and_func2_ok`: it requires both manual
overall QC and functional QC while retaining the full set of nominal sites.
Two of those sites (`LEUVEN_2` and `SBL`) have only one diagnosis in the
selected population, so the manifest excludes them under
`--min-site-class-size 1`. This produces 769 subjects across 18 evaluable
held-out sites. It is a site-eligibility rule, not a post-result exclusion.

## Final primary command

```bash
python scripts/prepare_abide_manifest.py --qc-rule rater1_and_func2_ok --check-urls
```

## Step 5 completion criteria

Step 5 is complete only when:

- The manifest script runs on the AWS instance and locally.
- The manifest hash is recorded.
- The site count and ASD/control counts are reviewed.
- The QC rule is frozen in `docs/decision_log.md`.
- At least one ROI time-series file has been downloaded and parsed in Step 6.

## AWS dataset installation

The manifest and downloader are committed, while the downloaded participant
files remain ignored by Git. On the AWS instance, copy these two scripts and
the primary manifest, then run:

```bash
python3 scripts/download_abide_timeseries.py --workers 8
```

The downloader is resumable: re-running the same command skips non-empty
files and writes `data/processed/abide_i_timeseries_download_summary.json`.
Do a small `--limit 5` smoke test first, then run the full 769-file download.

### Installation record

On 2026-08-03, the full primary manifest was installed on the designated AWS
instance at `~/bunn-abide/data/` using the committed downloader. The summary
was 769 requested files, 764 newly downloaded, 5 resumed from the smoke test,
and 0 failures (about 169 MiB on disk). A subsequent validation found 769
files matching the manifest, no missing or extra files, and no malformed time
series. Each numeric row has 116 AAL values; usable scan lengths range from 78
to 296 time points. The GPU driver was provisioned separately and is now
verified; see `docs/aws_environment.md`.
