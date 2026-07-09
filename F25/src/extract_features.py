from multiprocessing import Pool
import opensmile
import pandas as pd
import os
import argparse
from pathlib import Path

# Resolve paths relative to this file so the script works from any cwd.
ROOT = Path(__file__).resolve().parent.parent   # F25/
WAV_DIR = ROOT / 'clean_wavs'
OUT_CSV = ROOT / 'data' / 'raw' / 'features.csv'

VERBOSITY = 0

def init_worker(verbosity):
    global VERBOSITY
    VERBOSITY = verbosity

def extract_features(filename):
    if not filename.endswith('.wav'):
        return None
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    parts = filename.replace('.wav', '').split('_')
    initials = ''.join([p[0] for p in parts])
    filepath = str(WAV_DIR / filename)

    if VERBOSITY > 0:
        print(f'Processing {initials}...')

    features = smile.process_file(filepath)
    features.insert(0, 'ID', initials)
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable progress updates')
    args = parser.parse_args()
    verbosity = 1 if args.verbose else 0

    files = os.listdir(WAV_DIR)
    # Use 12 cores since my machine has 16
    with Pool(processes=12, initializer=init_worker, initargs=(verbosity,)) as pool:
        results = pool.map(extract_features, files)

    results = [r for r in results if r is not None]
    feature_matrix = pd.concat(results, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(OUT_CSV, index=False)
