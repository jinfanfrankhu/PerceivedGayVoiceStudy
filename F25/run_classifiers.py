import sys
import yaml
import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
import time
import argparse
import os
from datetime import datetime


REGRESSION_MODELS = {'svr', 'ridge', 'ordinal_ridge'}


class Logger:
    def __init__(self):
        self.lines = []

    def __call__(self, msg=''):
        print(msg)
        self.lines.append(str(msg))

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines) + '\n')


def load_data():
    features = pd.read_csv('features.csv')
    meta = pd.read_csv('Master Spreadsheet.csv')
    meta = meta.rename(columns={'Initials': 'ID'})
    df = features.merge(
        meta[['ID', 'Kinsey Scale (1-5)', 'Self-Described Sexual Orientation']], on='ID'
    )
    feature_cols = [c for c in features.columns if c != 'ID']
    X = df[feature_cols].values
    return df, X, feature_cols


def build_labels(df, label_names):
    labels = {}
    orientations = df['Self-Described Sexual Orientation'].str.lower().str.strip()

    if 'binary' in label_names:
        y = (orientations == 'straight').astype(int).values
        labels['binary'] = (y, 0.50)

    if 'three_class' in label_names:
        def _map(row):
            v = str(row['Self-Described Sexual Orientation']).lower().strip()
            return 0 if v == 'straight' else (2 if v == 'gay' else 1)
        y = df.apply(_map, axis=1).values
        chance = round(pd.Series(y).value_counts().max() / len(y), 2)
        labels['three_class'] = (y, chance)

    if 'ordinal' in label_names:
        y = df['Kinsey Scale (1-5)'].values
        labels['ordinal'] = (y, 0.20)

    return labels


def build_model(model_cfg, params):
    t = model_cfg['type']
    if t == 'svc':
        return SVC(
            kernel=model_cfg.get('kernel', 'rbf'),
            C=params.get('C', model_cfg.get('C', 1.0)),
            class_weight=model_cfg.get('class_weight') or None,
        )
    elif t == 'random_forest':
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', model_cfg.get('n_estimators', 100)),
            class_weight=model_cfg.get('class_weight') or None,
            random_state=model_cfg.get('random_state'),
        )
    elif t == 'svr':
        return SVR(
            kernel=model_cfg.get('kernel', 'linear'),
            C=params.get('C', model_cfg.get('C', 1.0)),
            epsilon=params.get('epsilon', model_cfg.get('epsilon', 0.1)),
        )
    elif t == 'ridge':
        return Ridge(alpha=params.get('alpha', model_cfg.get('alpha', 1.0)))
    elif t == 'ordinal_ridge':
        try:
            from mord import OrdinalRidge
        except ImportError:
            raise ImportError('mord required for ordinal_ridge. Install with: pip install mord')
        return OrdinalRidge(alpha=params.get('alpha', model_cfg.get('alpha', 1.0)))
    else:
        raise ValueError(f'Unknown model type: {t}')


def apply_feature_selection(X_train, X_test, y_train, feat_cfg, params):
    method = feat_cfg.get('method', 'none')
    if method == 'kbest':
        k = params.get('k', feat_cfg.get('k', 10))
        sel = SelectKBest(f_classif, k=k)
        X_train = sel.fit_transform(X_train, y_train)
        X_test = sel.transform(X_test)
    elif method == 'pca':
        n = params.get('n_components', feat_cfg.get('n_components', 10))
        pca = PCA(n_components=n)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    return X_train, X_test


def run_fold(train_idx, test_idx, X, y, model_cfg, feat_cfg, params):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_test = apply_feature_selection(X_train, X_test, y_train, feat_cfg, params)

    model = build_model(model_cfg, params)
    model.fit(X_train, y_train)
    return model.predict(X_test)[0], y_test[0]


def run_loocv(X, y, model_cfg, feat_cfg, params, n_jobs=12):
    splits = list(LeaveOneOut().split(X))
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(train, test, X, y, model_cfg, feat_cfg, params)
        for train, test in tqdm(splits, desc='  LOOCV', unit='fold', leave=False, file=sys.stderr)
    )
    elapsed = time.time() - t0
    preds, trues = zip(*results)
    return np.array(preds, dtype=float), np.array(trues, dtype=float), elapsed


def clf_metrics(preds, trues):
    pi, ti = preds.astype(int), trues.astype(int)
    return accuracy_score(ti, pi), confusion_matrix(ti, pi)


def reg_metrics(preds, trues):
    mae  = mean_absolute_error(trues, preds)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    r, p = spearmanr(trues, preds)
    r = float(r) if not np.isnan(r) else 0.0
    p = float(p) if not np.isnan(p) else 1.0
    within1 = float(np.mean(np.abs(preds - trues) <= 1.0))
    return mae, rmse, r, p, within1


def build_param_grid(model_cfg, feat_cfg):
    sweep = {}
    for k, v in model_cfg.items():
        if isinstance(v, list):
            sweep[k] = v
    for k, v in feat_cfg.items():
        if isinstance(v, list):
            sweep[k] = v
    if not sweep:
        return [{}]
    keys, values = zip(*sweep.items())
    return [dict(zip(keys, combo)) for combo in product(*values)]


LABEL_DISPLAY = {
    'binary':      'Binary (straight vs. non-straight)',
    'three_class': '3-class (straight / bi / gay)',
    'ordinal':     'Ordinal (1-5 attraction scale)',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to experiment config YAML')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    exp_name    = cfg['name']
    note        = cfg.get('note', '')
    model_cfg   = cfg.get('model', {})
    feat_cfg    = cfg.get('feature_selection', {'method': 'none'})
    label_names = cfg.get('labels', ['binary', 'three_class', 'ordinal'])
    regression  = model_cfg['type'] in REGRESSION_MODELS

    exp_dir   = os.path.dirname(os.path.abspath(args.config))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = os.path.join(exp_dir, 'results', timestamp)
    os.makedirs(run_dir, exist_ok=True)

    log = Logger()
    df, X, feature_cols = load_data()
    labels = build_labels(df, label_names)
    param_grid = build_param_grid(model_cfg, feat_cfg)

    # ── Header ──────────────────────────────────────────────────────────────
    log('=' * 56)
    log(f'Experiment : {exp_name}')
    log(f'Note       : {note or "(none)"}')
    log(f'Timestamp  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'Dataset    : {len(df)} speakers, {len(feature_cols)} features')
    log(f'Model      : {model_cfg["type"]}  ({"regression" if regression else "classification"})')
    log(f'Feat sel   : {feat_cfg.get("method", "none")}')
    log(f'Labels     : {", ".join(label_names)}')
    log(f'Param combos: {len(param_grid)}')
    log('=' * 56)

    # ── Class distributions ─────────────────────────────────────────────────
    log()
    log('Class distributions:')
    for lname, (y, chance) in labels.items():
        if lname == 'binary':
            log(f'  Binary      straight={y.sum()}  non-straight={(y==0).sum()}  (chance {chance:.0%})')
        elif lname == 'three_class':
            c = pd.Series(y).value_counts().sort_index()
            log(f'  3-class     straight={c.get(0,0)}  bi={c.get(1,0)}  gay={c.get(2,0)}  (chance {chance:.0%})')
        elif lname == 'ordinal':
            log(f'  Ordinal     range {y.min()}–{y.max()}, mean {y.mean():.2f}  (chance {chance:.0%})')

    # ── Sweep ───────────────────────────────────────────────────────────────
    csv_rows = []
    best = {lname: (None, -np.inf) for lname in labels}

    for params in param_grid:
        param_str = '  '.join(f'{k}={v}' for k, v in params.items()) if params else 'fixed'
        pad = '─' * max(0, 50 - len(param_str))
        log()
        log(f'── {param_str} {pad}')

        for lname, (y, chance) in labels.items():
            preds, trues, elapsed = run_loocv(X, y, model_cfg, feat_cfg, params)
            display = LABEL_DISPLAY[lname]

            row = {
                'experiment':  exp_name,
                'timestamp':   timestamp,
                'note':        note,
                'label':       lname,
                'model_type':  model_cfg['type'],
                'feat_method': feat_cfg.get('method', 'none'),
                'n_total':     len(y),
                'elapsed_s':   round(elapsed, 2),
            }

            if regression:
                mae, rmse, r, p, within1 = reg_metrics(preds, trues)
                log(f'  {display:<38}  MAE={mae:.3f}  RMSE={rmse:.3f}  Spearman r={r:.3f}  within-1={within1:.0%}  {elapsed:.1f}s')

                score = r
                row.update({
                    'mae':         round(mae, 4),
                    'rmse':        round(rmse, 4),
                    'spearman_r':  round(r, 4),
                    'spearman_p':  round(p, 4),
                    'within1_acc': round(within1, 4),
                })
            else:
                acc, cm = clf_metrics(preds, trues)
                n_correct = int(round(acc * len(y)))
                arrow     = '↑' if acc > chance + 0.005 else ('↓' if acc < chance - 0.005 else '~')
                cm_str    = str(cm.tolist())
                log(f'  {display:<38}  {acc:.3f}  ({n_correct}/{len(y)})  {arrow} chance  {elapsed:.1f}s')
                log(f'    cm: {cm_str}')

                score = acc
                row.update({
                    'accuracy':        round(acc, 4),
                    'n_correct':       n_correct,
                    'chance':          chance,
                    'above_chance':    acc > chance,
                    'confusion_matrix': cm_str,
                })

            if score > best[lname][1]:
                best[lname] = (params, score)

            for k, v in model_cfg.items():
                if k != 'type' and not isinstance(v, list):
                    row[f'model_{k}'] = v
            for k, v in feat_cfg.items():
                if k != 'method' and not isinstance(v, list):
                    row[f'feat_{k}'] = v
            row.update(params)
            csv_rows.append(row)

    # ── Summary ─────────────────────────────────────────────────────────────
    log()
    log('─' * 56)
    log('Best per label:')
    for lname, (params, score) in best.items():
        param_str  = '  '.join(f'{k}={v}' for k, v in params.items()) if params else 'n/a'
        metric_lbl = 'spearman_r' if regression else 'acc'
        log(f'  {lname:<12}  {metric_lbl}={score:.3f}  [{param_str}]')

    # ── Write outputs ────────────────────────────────────────────────────────
    csv_path = os.path.join(run_dir, 'results.csv')
    log_path = os.path.join(run_dir, 'run.log')

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    log.save(log_path)

    print(f'\nResults : {csv_path}')
    print(f'Log     : {log_path}')


if __name__ == '__main__':
    main()
