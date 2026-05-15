import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import argparse
from datetime import datetime

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--note', type=str, default='', help='Short note describing this run')
args = parser.parse_args()

# --- Output setup ---
import os
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('classifier_logs', exist_ok=True)
out_txt  = f'classifier_logs/results_{timestamp}.txt'
out_csv  = f'classifier_logs/results_{timestamp}.csv'

_log_file = open(out_txt, 'w')

def log(msg=''):
    print(msg)
    _log_file.write(msg + '\n')

log(f'Run: {timestamp}')
log(f'Note: {args.note}' if args.note else 'Note: (none)')
log('')

# --- Load and merge ---
log('Loading data...')
features = pd.read_csv('features.csv')
meta = pd.read_csv('Master Spreadsheet.csv')
meta = meta.rename(columns={'Initials': 'ID'})
df = features.merge(meta[['ID', 'Kinsey Scale (1-5)', 'Self-Described Sexual Orientation']], on='ID')
n_speakers = df.shape[0]
n_features = len([c for c in features.columns if c != 'ID'])
log(f'  Merged: {n_speakers} speakers, {n_features} features')

feature_cols = [c for c in features.columns if c != 'ID']
X = df[feature_cols].values

# --- Label definitions ---
log('\nBuilding label vectors...')

binary_labels = df['Self-Described Sexual Orientation'].str.lower().str.strip()
y_binary = (binary_labels == 'straight').astype(int).values
log(f'  Binary  — straight: {y_binary.sum()}, non-straight: {(y_binary==0).sum()}')

def three_class(row):
    label = str(row['Self-Described Sexual Orientation']).lower().strip()
    if label == 'straight': return 0
    elif label == 'gay': return 2
    else: return 1

y_three = df.apply(three_class, axis=1).values
counts = pd.Series(y_three).value_counts().sort_index()
log(f'  3-class — straight: {counts[0]}, bi: {counts[1]}, gay: {counts[2]}')

y_ordinal = df['Kinsey Scale (1-5)'].values
log(f'  Ordinal — range {y_ordinal.min()}–{y_ordinal.max()}, mean {y_ordinal.mean():.2f}')

# --- Single LOOCV fold (parallelized) ---
def run_fold(train_idx, test_idx, X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    pca = PCA(n_components=10)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    clf = SVC(kernel='linear', C=0.1, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf.predict(X_test)[0], y_test[0]

# --- LOOCV runner ---
csv_rows = []

def run_loocv(X, y, label, n_jobs=12):
    log(f'\n{"="*50}')
    log(f'Running: {label}')
    log(f'  Speakers: {len(y)} | Classes: {np.unique(y).tolist()} | Folds: {len(y)}')
    log(f'  Parallelizing across {n_jobs} cores...')

    loo = LeaveOneOut()
    splits = list(loo.split(X))

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(train, test, X, y)
        for train, test in tqdm(splits, desc=f'  Folds', unit='fold')
    )
    elapsed = time.time() - t0

    preds, trues = zip(*results)
    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds)

    log(f'\n  Finished in {elapsed:.1f}s')
    log(f'  Accuracy: {acc:.3f}  ({int(acc*len(y))}/{len(y)} correct)')
    log(f'  Confusion matrix:\n{cm}')

    csv_rows.append({
        'timestamp': timestamp,
        'note':      args.note,
        'model':     label,
        'accuracy':  round(acc, 4),
        'n_correct': int(acc * len(y)),
        'n_total':   len(y),
        'classes':   str(np.unique(y).tolist()),
        'confusion_matrix': str(cm.tolist()),
        'elapsed_s': round(elapsed, 2),
    })

    return acc

log('\n' + '='*50)
log('Starting classification — 3 models, LOOCV')
log(f'Chance baselines: binary=50%, 3-class={round(max(pd.Series(y_three).value_counts())/len(y_three)*100)}%, ordinal=20%')

run_loocv(X, y_binary,  'Binary: straight vs. non-straight')
run_loocv(X, y_three,   '3-class: straight / bi / gay')
run_loocv(X, y_ordinal, 'Ordinal 1-5: attraction scale')

log('\nDone.')
_log_file.close()

pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
print(f'\nResults written to {out_txt} and {out_csv}')
