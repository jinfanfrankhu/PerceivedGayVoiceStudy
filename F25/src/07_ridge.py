"""07_ridge.py  --  the FINAL step: honest multivariate prediction (Ridge regression).

Correlations (03-06) asked "does feature X relate to the target, one at a time?".
This asks the different, multivariate question: "can we predict the target from ALL
the acoustics TOGETHER?" Combinations can carry signal no single feature shows.

The danger at n=50 with dozens of features is that ANY model looks good on data it has
seen (this is what sank the archived eGeMAPS ML runs -- memorization, not learning). So
the whole design exists to NOT fool ourselves:

  * LEAVE-ONE-OUT CV: train on 49 speakers, predict the held-out 1, repeat 50x. A speaker
    is never scored by a model that trained on it.
  * PERMUTATION NULL (the crucial part): shuffle the labels and re-run the ENTIRE LOOCV
    pipeline, 1000x. This is what "predicting from noise" looks like given our exact n and
    feature count. The real score only counts if it beats this null. Directly answers
    "could I have gotten this by chance?".
  * ONE A-PRIORI PIPELINE, no sweep: median-impute -> standardize -> RidgeCV. Ridge (L2)
    is the safe default for p possibly > n; its alpha is chosen by efficient inner LOO
    INSIDE each training fold (never touching the held-out speaker), and the permutation
    null re-runs that too, so the alpha-selection optimism is also nulled out.

Design decisions (settled with user):
  * Both targets are REGRESSION (continuous), evaluated by cross-validated Spearman rho
    between out-of-fold predictions and truth -- no arbitrary Kinsey/perceived cutpoints,
    and perceived-vs-actual stay apples-to-apples. Spearman (not R^2) is the headline: it
    is rank-based (robust to Kinsey bimodality) and communicable as "how well ordered".
  * Feature-set LADDER, all three run; PRIMARY = segmental-confirmatory (the pre-registered
    13, theory-chosen before results -> cleanest). eGeMAPS-88 is the off-the-shelf baseline
    (03 showed it has no single surviving feature -> expected to struggle). Combined is
    richest but includes exploratory-selected columns -> read with the double-dip caveat.

Expected result, stated in advance: PERCEIVED beats its permutation null; ACTUAL does not.

This is the Ridge (L2) member of the prediction family; Elastic Net / PLS / PCR live in
sibling scripts and share the outputs/figures/prediction/ folder for cross-model comparison.

Outputs:
  outputs/tables/ridge_summary.csv            one row per (target x feature set)
  outputs/figures/prediction/ridge_null.png   observed CV rho vs permutation null
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from common import SPEAKERS_CSV, FEATURES_CSV, PROC, FIG, TABLES, ensure_dirs

PRED_DIR = FIG / 'prediction'
KINSEY = 'Kinsey Scale (1-5)'
TARGETS = {'perceived': 'perceived_mean', 'actual': KINSEY}

N_PERM = 1000
SEED = 0
ALPHAS = np.logspace(-1, 3, 12)          # a-priori ridge grid, tuned by inner LOO per fold

F0R = 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2'
F0S = 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm'
F0M = 'F0semitoneFrom27.5Hz_sma3nz_amean'
HNR = 'HNRdBACF_sma3nz_amean'

# PRIMARY: the 13 pre-registered confirmatory features (mirrors 04/06). Derived
# diph_dynamism / diph_duration are built the same way as 06.
SEGMENTAL_CONF = ['S_cog', 'S_skew', 'S_dur', 'vowel_space_area', 'front_f2',
                  'diph_dynamism', 'diph_duration', F0R, F0S, F0M, 'v_cpps', HNR, 'v_h1h2']


def build_derived(df, tokens):
    """diph_dynamism (mean z of trajlen) + diph_duration -- identical to 04/06."""
    tl = ['AY_trajlen', 'EY_trajlen', 'OW_trajlen', 'AW_trajlen']
    z = df[tl].apply(lambda c: (c - c.mean()) / c.std(ddof=1))
    df['diph_dynamism'] = z.mean(axis=1)
    good = tokens[(~tokens['dropped']) & (tokens['base'].isin(['AY', 'EY', 'OW', 'AW']))]
    df['diph_duration'] = df['file_id'].map(good.groupby('file_id')['dur'].mean())
    return df


def make_model():
    """The single a-priori pipeline. Refit from scratch inside every LOOCV fold."""
    return make_pipeline(SimpleImputer(strategy='median'),
                         StandardScaler(),
                         RidgeCV(alphas=ALPHAS))       # inner-LOO alpha, no held-out leak


def loocv_oof(X, y):
    """Out-of-fold predictions: for each speaker, train on the other 49 and predict it."""
    n = len(y)
    oof = np.empty(n)
    idx = np.arange(n)
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


def permutation_null(X, y, rng):
    """Re-run the whole LOOCV on shuffled labels N_PERM times -> null Spearman rhos."""
    nulls = np.empty(N_PERM)
    for k in range(N_PERM):
        yp = rng.permutation(y)
        nulls[k] = spearmanr(loocv_oof(X, yp), yp).statistic
    return nulls


def evaluate(name, X, y, rng):
    rho, r2, mae = cv_score(X, y)
    nulls = permutation_null(X, y, rng)
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
            # null spread as a grey band (5th-95th pct) with the median tick
            lo, hi = np.percentile(nl, [5, 95])
            ax.plot([lo, hi], [yi, yi], color='#c9c9c9', lw=9, solid_capstyle='butt',
                    zorder=1, label='permutation null (5-95%)' if yi == 0 else None)
            ax.plot([nl.mean(), nl.mean()], [yi - 0.25, yi + 0.25], color='#9a9a9a', lw=1.5,
                    zorder=2)
            beats = r['perm_p'] < 0.05
            ax.scatter([r['cv_spearman']], [yi], s=110, zorder=3,
                       color='#3a6ea5' if beats else '#c1553b',
                       edgecolor='k', linewidth=0.6,
                       label='observed CV rho' if yi == 0 else None)
            ax.annotate(f"rho={r['cv_spearman']:.2f}  p={r['perm_p']:.3f}"
                        + ('  *' if beats else ''),
                        (r['cv_spearman'], yi), xytext=(6, 8),
                        textcoords='offset points', fontsize=8,
                        color='#28425f' if beats else '#7a2f1f')
        ax.axvline(0, color='#bbb', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlabel('cross-validated Spearman  (out-of-fold pred vs truth)')
        ax.set_title(f'{tgt.upper()}  '
                     + ('(perceived_mean)' if tgt == 'perceived' else '(true Kinsey)'),
                     fontsize=11)
        ax.set_xlim(-0.5, 0.9)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Honest multivariate prediction: LOOCV Ridge vs permutation null (n=50)\n'
                 'blue = beats null (p<.05), red = indistinguishable from chance',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ensure_dirs(TABLES, PRED_DIR)
    print('07_ridge.py  (final: honest LOOCV Ridge + permutation null)')
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
            rng = np.random.default_rng(SEED)          # same seed -> comparable nulls
            res = evaluate(fs_name, X, y, rng)
            res.update({'target': tgt_name, 'feature_set': fs_name})
            rows.append(res)
            print(f"  {tgt_name:10} {fs_name:14} k={res['n_features']:3d}  "
                  f"CV rho={res['cv_spearman']:+.3f}  R2={res['cv_r2']:+.2f}  "
                  f"perm p={res['perm_p']:.3f}  (null mean {res['null_mean']:+.2f}, "
                  f"95% {res['null_p95']:+.2f})")

    figure(rows, PRED_DIR / 'ridge_null.png')
    out = pd.DataFrame([{k: v for k, v in r.items() if k != '_nulls'} for r in rows])
    out = out[['target', 'feature_set', 'n_features', 'n', 'cv_spearman', 'cv_r2',
               'cv_mae', 'perm_p', 'null_mean', 'null_p95']]
    out.to_csv(TABLES / 'ridge_summary.csv', index=False)
    print('  -> tables/ridge_summary.csv + figures/prediction/ridge_null.png')
    print('done.')


if __name__ == '__main__':
    main()
