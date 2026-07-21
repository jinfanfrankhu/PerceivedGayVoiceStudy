"""07b_lasso_selection.py  --  Lasso STABILITY SELECTION as a feature-importance diagnostic.

NOT a predictor and NOT an inference. A descriptive datapoint only: across many resamples,
how OFTEN does each acoustic feature get picked (non-zero coefficient) by a sparse linear
model? Features chosen in a high fraction of resamples are "stable" -> candidate important
cues; features rarely chosen are candidate unimportant. (Meinshausen & Buhlmann 2010,
"Stability Selection".)

WHY this and not a p-value: Lasso is a strong SELECTOR but an unstable PREDICTOR at n=50,
so we exploit the selection and ignore the prediction. No significance is claimed.

THE ONE CAVEAT TO READ EVERY BAR THROUGH -- vote splitting. When two features are
correlated (e.g. S_cog & S_peak both index sibilant fronting), Lasso picks ONE per run, so
each looks ~half as important as the construct really is. => interpret at the FAMILY /
cluster level, not feature-by-feature. To make the splitting visible we run the identical
selection with ELASTIC NET (grouping effect co-selects correlated clusters): where a
Lasso bar is low but its Elastic-Net twin is high, votes were being split.

Method (per pool x target):
  * impute (median) + standardize on the full 50 once (selection stability, not a perf claim).
  * pick alpha ONCE via LassoCV / ElasticNetCV on the full data (a typical sparsity level).
  * 1000 bootstrap resamples; refit plain Lasso/ElasticNet at that alpha; record non-zeros.
  * selection frequency = fraction of the 1000 runs each feature survived.

Run on TWO pools (combined-169 = everything competes; segmental-only = phonetic cues only)
and BOTH targets. Expected contrast = the study spine again: PERCEIVED yields a few clearly
stable cues; ACTUAL yields nothing stable (different noise features win each run).

Outputs (shared prediction/handcrafted/ folder; diagnostic, so clearly named):
  outputs/tables/lasso_selection_{combined,segmental}.csv
  outputs/figures/prediction/handcrafted/lasso_selection_{combined,segmental}.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV, Lasso, ElasticNet

from common import SPEAKERS_CSV, FEATURES_CSV, PROC, FIG, TABLES, ensure_dirs

PRED_DIR = FIG / 'prediction' / 'handcrafted'
KINSEY = 'Kinsey Scale (1-5)'
TARGETS = {'perceived': 'perceived_mean', 'actual': KINSEY}

N_BOOT = 1000
SEED = 0
L1_RATIO = 0.5            # Elastic Net blend (0=Ridge, 1=Lasso); 0.5 = even
TOP_N = 25               # bars shown per panel
STABLE = 0.50            # selection-frequency line for "stable"

FAM_COLOR = {'sibilant': '#c1553b', 'fricative': '#e0a15a', 'vowel': '#3a6ea5',
             'diphthong': '#5aa5a0', 'voice-quality': '#8a6bbf', 'eGeMAPS': '#9a9a9a'}


def family(f):
    """Coarse phonetic family for colouring / cluster-level reading."""
    if f.startswith(('S_', 'SH_', 'Z_')):
        return 'sibilant'
    if f.startswith(('F_', 'V_', 'DH_', 'HH_')):
        return 'fricative'
    if f.startswith('diph_') or (('trajlen' in f or 'rate' in f)
                                 and f[:2] in ('AY', 'EY', 'OW', 'AW')):
        return 'diphthong'
    if f in ('vowel_space_area', 'front_f2') or f.endswith(('_z1', '_z2')):
        return 'vowel'
    if f in ('v_cpps', 'v_h1h2'):
        return 'voice-quality'
    return 'eGeMAPS'


def build_derived(df, tokens):
    """diph_dynamism + diph_duration -- identical to 04/06/07."""
    tl = ['AY_trajlen', 'EY_trajlen', 'OW_trajlen', 'AW_trajlen']
    z = df[tl].apply(lambda c: (c - c.mean()) / c.std(ddof=1))
    df['diph_dynamism'] = z.mean(axis=1)
    good = tokens[(~tokens['dropped']) & (tokens['base'].isin(['AY', 'EY', 'OW', 'AW']))]
    df['diph_duration'] = df['file_id'].map(good.groupby('file_id')['dur'].mean())
    return df


# ----------------------------------------------------------------------------
# stability selection for one (X, y) with a given sparse estimator
# ----------------------------------------------------------------------------
def _prep(X):
    """Median-impute + standardize on the full sample once (see docstring rationale)."""
    Xi = SimpleImputer(strategy='median').fit_transform(X)
    return StandardScaler().fit_transform(Xi)


def stability(Xz, y, kind, rng):
    """Return per-feature selection frequency over N_BOOT bootstraps at a fixed alpha."""
    n = len(y)
    if kind == 'lasso':
        alpha = LassoCV(cv=5, random_state=SEED, max_iter=20000).fit(Xz, y).alpha_
        est = Lasso(alpha=alpha, max_iter=20000)
    else:
        cvm = ElasticNetCV(l1_ratio=L1_RATIO, cv=5, random_state=SEED, max_iter=20000)
        alpha = cvm.fit(Xz, y).alpha_
        est = ElasticNet(alpha=alpha, l1_ratio=L1_RATIO, max_iter=20000)
    hits = np.zeros(Xz.shape[1])
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        est.fit(Xz[idx], y[idx])
        hits += np.abs(est.coef_) > 1e-8
    return hits / N_BOOT, float(alpha)


def run_pool(df, cols, pool_name):
    feats = list(cols)
    rows = {f: {'feature': f, 'family': family(f)} for f in feats}
    alphas = {}
    for tgt, col in TARGETS.items():
        d = df.dropna(subset=[col])
        Xz = _prep(d[feats].to_numpy(float))
        y = d[col].to_numpy(float)
        for kind in ('lasso', 'enet'):
            rng = np.random.default_rng(SEED)          # same draws -> comparable
            freq, alpha = stability(Xz, y, kind, rng)
            alphas[(tgt, kind)] = alpha
            for f, v in zip(feats, freq):
                rows[f][f'{kind}_{tgt}'] = v
    tab = pd.DataFrame(rows.values())
    tab = tab.sort_values('lasso_perceived', ascending=False).reset_index(drop=True)
    return tab, alphas


# ----------------------------------------------------------------------------
def figure(tab, pool_name, path):
    ensure_dirs(PRED_DIR)
    # NOT sharey: each panel is sorted by its OWN target, so the two y-axes carry
    # different feature orderings and must label independently.
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=False)
    for ax, tgt in zip(axes, ['perceived', 'actual']):
        top = tab.sort_values(f'lasso_{tgt}', ascending=False).head(TOP_N).iloc[::-1]
        yy = np.arange(len(top))
        colors = [FAM_COLOR[f] for f in top['family']]
        ax.barh(yy, top[f'lasso_{tgt}'], color=colors, height=0.7, zorder=2,
                label='Lasso selection freq')
        # Elastic Net overlay -> where it sits ABOVE the bar tip, Lasso was splitting votes
        ax.scatter(top[f'enet_{tgt}'], yy, marker='D', s=26, facecolor='white',
                   edgecolor='k', linewidth=0.9, zorder=3, label='Elastic Net (same run)')
        ax.axvline(STABLE, color='#777', ls='--', lw=1, zorder=1)
        ax.set_yticks(yy)
        ax.set_yticklabels(top['feature'], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel('selection frequency over 1000 bootstraps')
        ax.set_title(f'{tgt.upper()}  ({"perceived_mean" if tgt=="perceived" else "true Kinsey"})',
                     fontsize=11)
    # shared family legend
    handles = [plt.Line2D([0], [0], marker='s', ls='', color=c, label=k)
               for k, c in FAM_COLOR.items()]
    handles.append(plt.Line2D([0], [0], marker='D', ls='', mfc='white', mec='k',
                              label='Elastic Net'))
    axes[1].legend(handles=handles, fontsize=7, loc='lower right', title='feature family')
    fig.suptitle(f'Lasso stability selection ({pool_name}) — DIAGNOSTIC, not prediction\n'
                 'high bar = frequently chosen; dashed = 0.50; Elastic-Net diamond above a '
                 'short bar = Lasso was splitting correlated votes', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _report(tab, pool_name, alphas):
    print(f'  [{pool_name}] alphas '
          + '  '.join(f'{k[0][:4]}/{k[1]}={v:.3g}' for k, v in alphas.items()))
    for tgt in ('perceived', 'actual'):
        stable = tab[tab[f'lasso_{tgt}'] >= STABLE].sort_values(f'lasso_{tgt}',
                                                                ascending=False)
        names = ', '.join(f"{r.feature}({r[f'lasso_{tgt}']:.2f})"
                          for _, r in stable.head(8).iterrows()) or '(none)'
        print(f'    {tgt:10} stable(>= {STABLE:.2f}): {len(stable):2d}  ->  {names}')


def main():
    ensure_dirs(TABLES, PRED_DIR)
    print('07b_lasso_selection.py  (stability-selection diagnostic; Lasso + Elastic Net)')
    sp = pd.read_csv(SPEAKERS_CSV)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv')
    tokens = pd.read_csv(PROC / 'segmental_tokens.csv')
    egemaps = [c for c in pd.read_csv(FEATURES_CSV).columns if c != 'ID']

    df = sp.merge(seg, on='file_id', how='left', suffixes=('', '_seg'))
    df = build_derived(df, tokens)
    seg_all = [c for c in seg.columns if c != 'file_id'] + ['diph_dynamism', 'diph_duration']

    pools = {
        'combined': list(dict.fromkeys(egemaps + seg_all)),   # everything competes
        'segmental': seg_all,                                  # phonetic cues only
    }
    for pool_name, cols in pools.items():
        tab, alphas = run_pool(df, cols, pool_name)
        keep = ['feature', 'family', 'lasso_perceived', 'enet_perceived',
                'lasso_actual', 'enet_actual']
        tab[keep].to_csv(TABLES / f'lasso_selection_{pool_name}.csv', index=False)
        figure(tab, pool_name, PRED_DIR / f'lasso_selection_{pool_name}.png')
        _report(tab, pool_name, alphas)
    print('  -> tables/lasso_selection_*.csv + figures/prediction/handcrafted/lasso_selection_*.png')
    print('done.')


if __name__ == '__main__':
    main()
