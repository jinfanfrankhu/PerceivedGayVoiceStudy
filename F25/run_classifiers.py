import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import argparse
import os
from datetime import datetime

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--note', type=str, default='', help='Short note describing this run')
args = parser.parse_args()

# --- Output setup ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('classifier_logs', exist_ok=True)
out_csv = f'classifier_logs/results_{timestamp}.csv'

def log(msg='', f=None):
    print(msg)
    if f:
        f.write(msg + '\n')

# --- Load and merge ---
print('Loading data...')
features = pd.read_csv('features.csv')
meta = pd.read_csv('Master Spreadsheet.csv')
meta = meta.rename(columns={'Initials': 'ID'})
df = features.merge(meta[['ID', 'Kinsey Scale (1-5)', 'Self-Described Sexual Orientation']], on='ID')
n_speakers = df.shape[0]
n_features = len([c for c in features.columns if c != 'ID'])
print(f'  Merged: {n_speakers} speakers, {n_features} features')

feature_cols = [c for c in features.columns if c != 'ID']
X = df[feature_cols].values

# --- Label definitions ---
print('\nBuilding label vectors...')

binary_labels = df['Self-Described Sexual Orientation'].str.lower().str.strip()
y_binary = (binary_labels == 'straight').astype(int).values
print(f'  Binary  — straight: {y_binary.sum()}, non-straight: {(y_binary==0).sum()}')

def three_class(row):
    label = str(row['Self-Described Sexual Orientation']).lower().strip()
    if label == 'straight': return 0
    elif label == 'gay': return 2
    else: return 1

y_three = df.apply(three_class, axis=1).values
counts = pd.Series(y_three).value_counts().sort_index()
print(f'  3-class — straight: {counts[0]}, bi: {counts[1]}, gay: {counts[2]}')

y_ordinal = df['Kinsey Scale (1-5)'].values
print(f'  Ordinal — range {y_ordinal.min()}–{y_ordinal.max()}, mean {y_ordinal.mean():.2f}')

# --- Single LOOCV fold (parallelized) ---
def run_fold(train_idx, test_idx, X, y, k, C):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    selector = SelectKBest(f_classif, k=k)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    clf = SVC(kernel='linear', C=C, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf.predict(X_test)[0], y_test[0]

# --- LOOCV runner ---
def run_loocv(X, y, label, k, C, f, csv_rows, n_jobs=12):
    log(f'\n{"="*50}', f)
    log(f'Running: {label}', f)
    log(f'  Speakers: {len(y)} | Classes: {np.unique(y).tolist()} | Folds: {len(y)}', f)
    log(f'  Parallelizing across {n_jobs} cores...', f)

    loo = LeaveOneOut()
    splits = list(loo.split(X))

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(train, test, X, y, k, C)
        for train, test in tqdm(splits, desc=f'  Folds', unit='fold')
    )
    elapsed = time.time() - t0

    preds, trues = zip(*results)
    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds)

    log(f'\n  Finished in {elapsed:.1f}s', f)
    log(f'  Accuracy: {acc:.3f}  ({int(acc*len(y))}/{len(y)} correct)', f)
    log(f'  Confusion matrix:\n{cm}', f)

    csv_rows.append({
        'timestamp':        timestamp,
        'note':             args.note,
        'k':                k,
        'C':                C,
        'model':            label,
        'accuracy':         round(acc, 4),
        'n_correct':        int(acc * len(y)),
        'n_total':          len(y),
        'classes':          str(np.unique(y).tolist()),
        'confusion_matrix': str(cm.tolist()),
        'elapsed_s':        round(elapsed, 2),
    })

    return acc

# --- Main sweep ---
csv_rows = []

chance_binary  = 50
chance_three   = round(max(pd.Series(y_three).value_counts()) / len(y_three) * 100)
chance_ordinal = 20

for k in [10, 15, 20]:
    for C in [0.1, 0.3, 0.5, 0.7, 1.0]:
        out_txt = f'classifier_logs/results_{timestamp}_k{k}_C{C}.txt'
        with open(out_txt, 'w') as f:
            log(f'Run: {timestamp}', f)
            log(f'Note: {args.note}' if args.note else 'Note: (none)', f)
            log(f'k={k}  C={C}', f)
            log('', f)
            log(f'Chance baselines: binary={chance_binary}%, 3-class={chance_three}%, ordinal={chance_ordinal}%', f)
            log(f'\n{"="*50}', f)
            log(f'Starting classification — 3 models, LOOCV  (k={k}, C={C})', f)

            run_loocv(X, y_binary,  'Binary: straight vs. non-straight', k, C, f, csv_rows)
            run_loocv(X, y_three,   '3-class: straight / bi / gay',      k, C, f, csv_rows)
            run_loocv(X, y_ordinal, 'Ordinal 1-5: attraction scale',     k, C, f, csv_rows)

            log('\nDone.', f)
        print(f'  -> wrote {out_txt}')

pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
print(f'\nAll results written to {out_csv}')
