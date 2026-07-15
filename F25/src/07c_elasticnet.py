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

The rho=-1 intercept artefact (why some cells are reported as undefined):
  When the inner CV picks an alpha large enough to zero EVERY coefficient, the fold has no
  features left and predicts only its training intercept -- which in LOOCV is the leave-one-out
  mean (SUM y - y_i)/(n-1), a strictly DECREASING function of y_i. Spearman of that against y
  is exactly -1 for ANY data. It means "the model found nothing", not "the model predicts
  backwards", so it is recorded as undefined (NaN) rather than as a strong negative result.

  This makes the metric NEGATIVELY BIASED under H0, not merely noisy: on shuffled labels the
  same collapse fires ~15-55% of the time (see null_collapse_frac), which is why null_mean sits
  near -0.6 instead of 0. That bias is a property of LOOCV+Spearman, NOT a defect in the null --
  and calibrating against it is exactly what a permutation test is for, so every draw is kept
  when computing p (see summarise). Read null_mean/null_p95 as "what this metric does when there
  is nothing to find", never as evidence of inverse prediction. The cost is POWER, not validity:
  a metric that throws away a third of its draws on a degenerate value is a blunt instrument.
  If that ever bites, the fix is a less pathological metric (K-fold rather than LOO) -- which is
  a full recompute and a change to 07's shared harness, so it is deliberately not done here.

Outputs (shared prediction/ folder for cross-model comparison with ridge_*):
  outputs/tables/enet_summary.csv            one row per (target x feature set)
  outputs/figures/prediction/enet_null.png   observed CV rho vs permutation null
  outputs/tables/enet_nulls.npz              raw null draws (see --from-cache)

Run with --from-cache to re-derive the summary and figure from the persisted null draws
without paying the ~6h recompute.
"""
import argparse
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
# A fully-collapsed fit scores exactly -1 (see the artefact note in the module docstring);
# a near-total collapse -- most folds zeroed, a stray fold not -- lands a hair above it and is
# just as meaningless, so the test is a tolerance rather than equality. Genuine inverse
# prediction across n=50 never reaches this.
COLLAPSE_RHO = -0.999
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


def is_collapsed(rho):
    """True when rho is the intercept artefact rather than a real (inverse) fit."""
    return bool(np.isfinite(rho) and rho <= COLLAPSE_RHO)


def collapse_note(r):
    """Why a cell has no rho. The fold fraction is unavailable when re-deriving from a
    summary written before it was recorded, so it is only named when known."""
    if np.isfinite(r['zeroed_folds']):
        return f"Elastic Net zeroed every coefficient in {r['zeroed_folds']:.0%} of folds"
    return 'Elastic Net zeroed every coefficient'


def loocv_oof(X, y):
    """Out-of-fold predictions + the fraction of folds that zeroed every coefficient.

    The zeroed-fold fraction is the mechanism behind the rho=-1 artefact, reported as a
    diagnostic so a collapsed cell can be read as "no features survived" rather than a mystery.
    """
    n = len(y)
    oof = np.empty(n)
    idx = np.arange(n)
    zeroed = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for i in range(n):
            tr = idx != i
            m = make_model().fit(X[tr], y[tr])
            if not np.any(m[-1].coef_):                    # ElasticNetCV step of the pipeline
                zeroed += 1
            oof[i] = m.predict(X[i:i + 1])[0]
    return oof, zeroed / n


def cv_score(X, y):
    """Cross-validated Spearman between out-of-fold predictions and truth."""
    oof, zeroed_frac = loocv_oof(X, y)
    rho = spearmanr(oof, y).statistic
    ss = ((y - oof) ** 2).sum()
    r2 = 1 - ss / ((y - y.mean()) ** 2).sum()          # out-of-fold R^2 (can be < 0)
    mae = np.abs(y - oof).mean()
    return rho, r2, mae, zeroed_frac


def _perm_stat(X, y, seed):
    """One permutation: shuffle labels, re-run the whole LOOCV, return null Spearman."""
    yp = np.random.default_rng(seed).permutation(y)
    oof, _ = loocv_oof(X, yp)
    return spearmanr(oof, yp).statistic


def permutation_null(X, y):
    """Re-run the whole LOOCV on shuffled labels N_PERM times (parallel) -> null rhos."""
    seeds = np.random.SeedSequence(SEED).spawn(N_PERM)   # independent, reproducible streams
    nulls = Parallel(n_jobs=N_JOBS)(
        delayed(_perm_stat)(X, y, s) for s in seeds)
    return np.asarray(nulls)


def summarise(rho_raw, r2, mae, zeroed_frac, nulls, n_features, n):
    """Fold one (target x feature set) cell down to its reported row.

    The p-value deliberately uses EVERY null draw, collapsed ones included. It is tempting to
    drop them as degenerate, but they are honest H0 behaviour: under the null the observed fit
    collapses just as often as a shuffled one does, so observed and null draws stay exchangeable
    and the unfiltered p is exact. Screening the null by value would break that exchangeability
    and buy nothing -- collapsed draws never exceed a positive observed rho anyway.

    What the artefact does corrupt is INTERPRETATION, so that is what gets screened: a collapsed
    observation is reported as undefined rather than as a strong negative result.
    """
    collapsed = is_collapsed(rho_raw)
    # one-sided (we only care about predicting BETTER than chance); +1 smoothing
    p = (1 + int((nulls >= rho_raw).sum())) / (len(nulls) + 1)
    return {'cv_spearman': np.nan if collapsed else rho_raw, 'cv_spearman_raw': rho_raw,
            'collapsed': collapsed, 'cv_r2': r2, 'cv_mae': mae, 'zeroed_folds': zeroed_frac,
            'perm_p': p, 'null_mean': float(nulls.mean()),
            'null_p95': float(np.percentile(nulls, 95)),
            'null_collapse_frac': float(np.mean([is_collapsed(v) for v in nulls])),
            'n_features': n_features, 'n': n, '_nulls': nulls}


def evaluate(name, X, y):
    rho_raw, r2, mae, zeroed_frac = cv_score(X, y)
    nulls = permutation_null(X, y)
    return summarise(rho_raw, r2, mae, zeroed_frac, nulls, X.shape[1], len(y))


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
            # every draw, collapsed included -- this band is the metric's true H0 behaviour
            lo, hi = np.percentile(nl, [5, 95])
            ax.plot([lo, hi], [yi, yi], color='#c9c9c9', lw=9, solid_capstyle='butt',
                    zorder=1, label='permutation null (5-95%)' if yi == 0 else None)
            ax.plot([nl.mean()] * 2, [yi - 0.25, yi + 0.25], color='#9a9a9a', lw=1.5,
                    zorder=2)
            if r['collapsed']:
                # No point to plot: the fit zeroed every coefficient, so rho is undefined
                # rather than strongly negative (see the artefact note in the docstring).
                ax.annotate(f'no signal: {collapse_note(r)}',
                            (0, yi), xytext=(6, 0), textcoords='offset points',
                            fontsize=8, color='#7a7a7a', style='italic', va='center')
                continue
            beats = r['perm_p'] < 0.05
            rho = r['cv_spearman']
            ax.scatter([rho], [yi], s=110, zorder=3,
                       color='#3a6ea5' if beats else '#c1553b',
                       edgecolor='k', linewidth=0.6,
                       label='observed CV rho' if yi == 0 else None)
            ax.annotate(f"rho={rho:.2f}  p={r['perm_p']:.3f}" + ('  *' if beats else ''),
                        (rho, yi), xytext=(6, 8), textcoords='offset points',
                        fontsize=8, color='#28425f' if beats else '#7a2f1f')
        ax.axvline(0, color='#bbb', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlabel('cross-validated Spearman  (out-of-fold pred vs truth)')
        ax.set_title(f'{tgt.upper()}  '
                     + ('(perceived_mean)' if tgt == 'perceived' else '(true Kinsey)'),
                     fontsize=11)
        ax.set_xlim(-1.08, 0.9)          # wide enough for the null's collapsed tail at rho=-1
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Robustness twin: LOOCV Elastic Net vs permutation null (n=50)\n'
                 'blue = beats null (p<.05), red = indistinguishable from chance, '
                 'grey italic = no signal (rho undefined)\n'
                 'null sits below 0 because LOOCV+Spearman is negatively biased when there is '
                 'nothing to find -- the test calibrates against that',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def from_cache():
    """Re-derive the rows from persisted null draws + the last summary -- no refit.

    The null draws are raw data (a rho is a rho); only the screening and reporting around them
    changed, so the ~6h of fits does not need repeating. cv_spearman_raw is kept in the summary
    precisely so this stays idempotent -- reading back a NaN'd cv_spearman would lose the
    artefact value needed to re-detect the collapse.
    """
    nulls = np.load(TABLES / 'enet_nulls.npz')
    prev = pd.read_csv(TABLES / 'enet_summary.csv')
    raw_col = 'cv_spearman_raw' if 'cv_spearman_raw' in prev.columns else 'cv_spearman'
    rows = []
    for _, r in prev.iterrows():
        res = summarise(float(r[raw_col]), float(r['cv_r2']), float(r['cv_mae']),
                        float(r['zeroed_folds']) if 'zeroed_folds' in prev.columns else np.nan,
                        nulls[f"{r['target']}__{r['feature_set']}"],
                        int(r['n_features']), int(r['n']))
        res.update({'target': r['target'], 'feature_set': r['feature_set']})
        rows.append(res)
    return rows


def compute():
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
    return rows


def report(r):
    """One console line per cell; collapsed cells have no rho or p to print."""
    head = f"  {r['target']:10} {r['feature_set']:14} k={r['n_features']:3d}  "
    tail = (f"perm p={r['perm_p']:.3f}  (null mean {r['null_mean']:+.2f}, "
            f"95% {r['null_p95']:+.2f}, {r['null_collapse_frac']:.0%} of draws collapsed)")
    if r['collapsed']:
        return head + f"NO SIGNAL ({collapse_note(r)}; rho undefined)  {tail}"
    return head + (f"CV rho={r['cv_spearman']:+.3f}  R2={r['cv_r2']:+.2f}  {tail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--from-cache', action='store_true',
                    help='re-derive summary + figure from tables/enet_nulls.npz '
                         '(no ~6h recompute)')
    args = ap.parse_args()

    ensure_dirs(TABLES, PRED_DIR)
    print('07c_elasticnet.py  (robustness twin: honest LOOCV Elastic Net + permutation null)')
    rows = from_cache() if args.from_cache else compute()
    if args.from_cache:
        print('  (re-derived from cached null draws; no refit)')
    for r in rows:
        print(report(r))

    figure(rows, PRED_DIR / 'enet_null.png')
    if not args.from_cache:
        # persist the null arrays so the figure can be re-plotted without the (~6h) recompute
        np.savez_compressed(TABLES / 'enet_nulls.npz',
                            **{f"{r['target']}__{r['feature_set']}": r['_nulls'] for r in rows})
    out = pd.DataFrame([{k: v for k, v in r.items() if k != '_nulls'} for r in rows])
    out = out[['target', 'feature_set', 'n_features', 'n', 'cv_spearman', 'cv_spearman_raw',
               'collapsed', 'zeroed_folds', 'cv_r2', 'cv_mae', 'perm_p', 'null_mean',
               'null_p95', 'null_collapse_frac']]
    out.to_csv(TABLES / 'enet_summary.csv', index=False)
    print('  -> tables/enet_summary.csv + figures/prediction/enet_null.png')
    print('done.')


if __name__ == '__main__':
    main()
