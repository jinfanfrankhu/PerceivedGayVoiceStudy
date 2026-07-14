"""07c_elasticnet.py  --  Elastic Net member of the prediction family (robustness twin of 07).

Same honest question and the SAME harness as 07_ridge.py -- LOOCV out-of-fold prediction
vs a 1000-run permutation null, both targets, the same three feature-set ladder, the same
Spearman headline -- so the two are read side by side. The ONLY thing that changes is the
estimator: Ridge's L2 shrink-everything is swapped for Elastic Net's L1+L2 blend, which can
ZERO out useless features (sparse) while still SHARING credit across correlated ones (the
grouping effect Lasso lacks). If the study spine (perceived predictable, actual not) is real,
it should not care which of these linear models we use -- that invariance is the point.

Why Elastic Net specifically as the robustness check:
  * Ridge keeps all p features with small weights -> great when everything is a bit useful,
    but noisy eGeMAPS columns are never actually turned off.
  * Lasso (07b) turns most off but splits votes among correlated cues and is unstable at n=50.
  * Elastic Net is the middle ground: sparse like Lasso, but co-selects correlated clusters
    like Ridge. If perceived survives here too, the signal is not a Ridge artefact.

Both alpha AND l1_ratio are chosen by inner 5-fold CV INSIDE each training fold (never touches
the held-out speaker), and the permutation null re-runs that whole selection -> the model-
selection optimism is nulled out exactly as in 07.

Speed: ElasticNetCV is ~10-30x heavier per fit than RidgeCV's analytic inner-LOO, so the
1000-permutation null is parallelised across cores with joblib (identical results, just faster).

Expected result, same as 07: PERCEIVED beats its permutation null; ACTUAL does not.

Outputs (shared prediction/ folder for cross-model comparison with ridge_*):
  outputs/tables/enet_summary.csv            one row per (target x feature set)
  outputs/figures/prediction/enet_null.png   observed CV rho vs permutation null
"""
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

from common import SPEAKERS_CSV, FEATURES_CSV, PROC, FIG, TABLES, ensure_dirs

PRED_DIR = FIG / 'prediction'
KINSEY = 'Kinsey Scale (1-5)'
TARGETS = {'perceived': 'perceived_mean', 'actual': KINSEY}

N_PERM = 1000
SEED = 0
# a-priori inner-CV grid: l1_ratio spans balanced (.5) to near-Lasso (.99). A coarse grid
# is deliberate -- Elastic Net is insensitive to fine l1_ratio spacing, and CD on the
# correlated eGeMAPS block is the runtime bottleneck. alphas auto (n_alphas below).
L1_RATIOS = [0.5, 0.9, 0.99]
N_ALPHAS = 30
MAX_ITER = 5000                            # ample; 20000 just paid a non-convergence tax
N_JOBS = -1                                # parallelise the permutation null across cores

F0R = 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2'
F0S = 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm'
F0M = 'F0semitoneFrom27.5Hz_sma3nz_amean'
HNR = 'HNRdBACF_sma3nz_amean'

# PRIMARY: the 13 pre-registered confirmatory features (mirrors 04/06/07).
SEGMENTAL_CONF = ['S_cog', 'S_skew', 'S_dur', 'vowel_space_area', 'front_f2',
                  'diph_dynamism', 'diph_duration', F0R, F0S, F0M, 'v_cpps', HNR, 'v_h1h2']


def build_derived(df, tokens):
    """diph_dynamism (mean z of trajlen) + diph_duration -- identical to 04/06/07."""
    tl = ['AY_trajlen', 'EY_trajlen', 'OW_trajlen', 'AW_trajlen']
    z = df[tl].apply(lambda c: (c - c.mean()) / c.std(ddof=1))
    df['diph_dynamism'] = z.mean(axis=1)
    good = tokens[(~tokens['dropped']) & (tokens['base'].isin(['AY', 'EY', 'OW', 'AW']))]
    df['diph_duration'] = df['file_id'].map(good.groupby('file_id')['dur'].mean())
    return df


def make_model():
    """The single a-priori pipeline. Refit from scratch inside every LOOCV fold.
    ElasticNetCV chooses alpha AND l1_ratio by inner 5-fold CV -> no held-out leak."""
    return make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        ElasticNetCV(l1_ratio=L1_RATIOS, n_alphas=N_ALPHAS, cv=5,
                     max_iter=MAX_ITER, random_state=SEED, n_jobs=1))


def loocv_oof(X, y):
    """Out-of-fold predictions: for each speaker, train on the other 49 and predict it."""
    n = len(y)
    oof = np.empty(n)
    idx = np.arange(n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for i in range(n):
            tr = idx != i
            oof[i] = make_model().fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    return oof


def cv_score(X, y):
    """Cross-validated Spearman between out-of-fold predictions and truth."""
    oof = loocv_oof(X, y)
    rho = spearmanr(oof, y).statistic
    ss = ((y - oof) ** 2).sum()
    r2 = 1 - ss / ((y - y.mean()) ** 2).sum()          # out-of-fold R^2 (can be < 0)
    mae = np.abs(y - oof).mean()
    return rho, r2, mae


def _perm_stat(X, y, seed):
    """One permutation: shuffle labels, re-run the whole LOOCV, return null Spearman."""
    yp = np.random.default_rng(seed).permutation(y)
    return spearmanr(loocv_oof(X, yp), yp).statistic


def permutation_null(X, y):
    """Re-run the whole LOOCV on shuffled labels N_PERM times (parallel) -> null rhos."""
    seeds = np.random.SeedSequence(SEED).spawn(N_PERM)   # independent, reproducible streams
    nulls = Parallel(n_jobs=N_JOBS)(
        delayed(_perm_stat)(X, y, s) for s in seeds)
    return np.asarray(nulls)


def evaluate(name, X, y):
    rho, r2, mae = cv_score(X, y)
    nulls = permutation_null(X, y)
    # one-sided (we only care about predicting BETTER than chance); +1 smoothing
    p = (1 + int((nulls >= rho).sum())) / (N_PERM + 1)
    return {'cv_spearman': rho, 'cv_r2': r2, 'cv_mae': mae, 'perm_p': p,
            'null_mean': float(nulls.mean()), 'null_p95': float(np.percentile(nulls, 95)),
            'n_features': X.shape[1], 'n': len(y), '_nulls': nulls}


def figure(rows, path):
    ensure_dirs(PRED_DIR)
    order = ['segmental-conf', 'eGeMAPS-88', 'combined']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, tgt in zip(axes, ['perceived', 'actual']):
        sub = {r['feature_set']: r for r in rows if r['target'] == tgt}
        y = np.arange(len(order))
        for yi, fs in zip(y, order):
            r = sub[fs]
            nl = r['_nulls']
            lo, hi = np.percentile(nl, [5, 95])
            ax.plot([lo, hi], [yi, yi], color='#c9c9c9', lw=9, solid_capstyle='butt',
                    zorder=1, label='permutation null (5-95%)' if yi == 0 else None)
            ax.plot([nl.mean(), nl.mean()], [yi - 0.25, yi + 0.25], color='#9a9a9a', lw=1.5,
                    zorder=2)
            beats = r['perm_p'] < 0.05
            rho = r['cv_spearman']
            # rho == -1 is the LOO mean-artifact: Elastic Net zeroed every coefficient
            # (no signal), so oof_i = (SUM y - y_i)/(n-1), strictly decreasing in y_i.
            collapsed = rho <= -0.999
            ax.scatter([rho], [yi], s=110, zorder=3,
                       color='#3a6ea5' if beats else '#c1553b',
                       edgecolor='k', linewidth=0.6,
                       label='observed CV rho' if yi == 0 else None)
            txt = (f"rho={rho:.2f}  p={r['perm_p']:.3f}" + ('  *' if beats else ''))
            if collapsed:
                txt += '  (collapsed to intercept: no signal)'
            ax.annotate(txt, (rho, yi), xytext=(6, 8), textcoords='offset points',
                        fontsize=8, color='#28425f' if beats else '#7a2f1f')
        ax.axvline(0, color='#bbb', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlabel('cross-validated Spearman  (out-of-fold pred vs truth)')
        ax.set_title(f'{tgt.upper()}  '
                     + ('(perceived_mean)' if tgt == 'perceived' else '(true Kinsey)'),
                     fontsize=11)
        ax.set_xlim(-1.08, 0.9)          # wide enough to show a fully-collapsed rho=-1
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Robustness twin: LOOCV Elastic Net vs permutation null (n=50)\n'
                 'blue = beats null (p<.05), red = indistinguishable from chance',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ensure_dirs(TABLES, PRED_DIR)
    print('07c_elasticnet.py  (robustness twin: honest LOOCV Elastic Net + permutation null)')
    sp = pd.read_csv(SPEAKERS_CSV)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv')
    tokens = pd.read_csv(PROC / 'segmental_tokens.csv')
    egemaps = [c for c in pd.read_csv(FEATURES_CSV).columns if c != 'ID']

    df = sp.merge(seg, on='file_id', how='left', suffixes=('', '_seg'))
    df = build_derived(df, tokens)

    seg_all = [c for c in seg.columns if c != 'file_id'] + ['diph_dynamism', 'diph_duration']
    feature_sets = {
        'segmental-conf': SEGMENTAL_CONF,                       # PRIMARY (a-priori 13)
        'eGeMAPS-88': egemaps,                                  # off-the-shelf baseline
        'combined': list(dict.fromkeys(egemaps + seg_all)),     # richest, caveat applies
    }

    rows = []
    for tgt_name, tgt_col in TARGETS.items():
        d = df.dropna(subset=[tgt_col])
        y = d[tgt_col].to_numpy(float)
        for fs_name, cols in feature_sets.items():
            X = d[cols].to_numpy(float)
            res = evaluate(fs_name, X, y)
            res.update({'target': tgt_name, 'feature_set': fs_name})
            rows.append(res)
            print(f"  {tgt_name:10} {fs_name:14} k={res['n_features']:3d}  "
                  f"CV rho={res['cv_spearman']:+.3f}  R2={res['cv_r2']:+.2f}  "
                  f"perm p={res['perm_p']:.3f}  (null mean {res['null_mean']:+.2f}, "
                  f"95% {res['null_p95']:+.2f})")

    figure(rows, PRED_DIR / 'enet_null.png')
    # persist the null arrays so the figure can be re-plotted without the (~6h) recompute
    np.savez_compressed(TABLES / 'enet_nulls.npz',
                        **{f"{r['target']}__{r['feature_set']}": r['_nulls'] for r in rows})
    out = pd.DataFrame([{k: v for k, v in r.items() if k != '_nulls'} for r in rows])
    out = out[['target', 'feature_set', 'n_features', 'n', 'cv_spearman', 'cv_r2',
               'cv_mae', 'perm_p', 'null_mean', 'null_p95']]
    out.to_csv(TABLES / 'enet_summary.csv', index=False)
    print('  -> tables/enet_summary.csv + figures/prediction/enet_null.png')
    print('done.')


if __name__ == '__main__':
    main()
