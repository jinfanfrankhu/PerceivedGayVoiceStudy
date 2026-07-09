"""Shared paths and small helpers for the F25 analysis pipeline.

Every path is resolved relative to this file, so scripts work from any cwd.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # F25/
RAW = ROOT / 'data' / 'raw'                             # immutable source data
PROC = ROOT / 'data' / 'processed'                      # generated datasets
OUTPUTS = ROOT / 'outputs'
FIG = OUTPUTS / 'figures'
TABLES = OUTPUTS / 'tables'

# Raw source files
MASTER_CSV = RAW / 'Master Spreadsheet.csv'
RATINGS_CSV = RAW / 'ratings.csv'
FEATURES_CSV = RAW / 'features.csv'

# Processed outputs
CROSSWALK_CSV = PROC / 'crosswalk.csv'
SPEAKERS_CSV = PROC / 'speakers.csv'
RATINGS_CLEAN_CSV = PROC / 'ratings_clean.csv'


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
