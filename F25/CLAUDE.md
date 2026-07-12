This is a research project investigating the "gay voice" phenomenon within Gen Z males.

See `README.md` for the full folder layout, run order, and method notes. Quick map:

- **Raw data** (immutable): `data/raw/` — `Master Spreadsheet.csv` (metadata + true
  Kinsey, canonical speaker names), `ratings.csv` (listener ratings from the Neon
  platform), `features.csv` (88 eGeMAPSv02 features).
- **Processed data** (regenerable): `data/processed/` — built by `src/build_dataset.py`.
- **Code**: `src/` (`common.py` paths, `build_dataset.py` first, then numbered analysis
  scripts). All paths resolve relative to the script, so cwd doesn't matter.
- **Outputs**: `outputs/figures/`, `outputs/tables/`.
- **Archive**: `archive/` holds the superseded exploratory pipeline (old
  `run_classifiers.py` config-driven experiments, `classifier_logs/`,
  `feature_rankings/`). Reference only — not the active path.

Use **`py -3.13`** (has pandas/scipy/sklearn/matplotlib/seaborn/diptest). The default
`python` (3.14) lacks the scientific stack.

## Kinsey Scale
Modified scale: 1 = complete attraction to women, 2 = preference to women, 3 = even bi,
4 = preference to men, 5 = complete men. Ratings use the same 1–5 scale
(1 = sounds straight ... 5 = sounds gay).

## Three data layers (same 50 speakers)
Acoustics (features.csv) · Actual orientation (Kinsey + self-ID) · Perceived gayness
(listener ratings). Central questions: acoustics→actual, acoustics→perceived,
perceived-vs-actual accuracy.

## Data-integrity gotcha
Speaker names are canonicalized to the Master Spreadsheet. The SAME three speakers were
originally misspelled in two independent places and both were corrected to the Master
canonical spelling (Chandan→Chandani, Steven→Samuel, Gallway→Galway):
  1. `data/raw/ratings.csv` (listener export) — guarded by `build_dataset.py check_join()`.
  2. The **audio filenames** in `clean_wavs/`, and hence `mfa_corpus/` + `mfa_textgrids/`
     folder names — which feed `file_id` in the segmental pipeline. These were renamed
     to canonical; `04_segmental.py` / `05_segmental_explore.py` now warn loudly if any
     segmental `file_id` fails to match Master (this bug silently ran the segmental
     analyses on n=47 before it was caught). No MFA re-run was needed — TextGrid content
     is filename-independent.
Join everything through `data/processed/crosswalk.csv`.
