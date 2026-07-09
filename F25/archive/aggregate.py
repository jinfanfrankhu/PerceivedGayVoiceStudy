"""Rebuild results_all.csv from all experiments/*/results/*/results.csv files."""
import glob
import os
import pandas as pd

pattern = os.path.join('experiments', '*', 'results', '*', 'results.csv')
paths = sorted(glob.glob(pattern))

if not paths:
    print('No result files found in experiments/*/results/*/')
else:
    frames = [pd.read_csv(p) for p in paths]
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv('results_all.csv', index=False)
    print(f'Aggregated {len(paths)} result file(s) → results_all.csv ({len(all_results)} rows)')
