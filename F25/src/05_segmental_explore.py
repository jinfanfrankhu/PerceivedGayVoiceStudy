"""05_segmental_explore.py  --  EXPLORATORY sweep of ALL segmental features.

NOT confirmatory. The pre-registered directional test is 04_segmental.py; this is the
hypothesis-GENERATING companion (decision A5). It correlates every segmental feature
(all vowels' Lobanov F1/F2, all fricatives' spectral moments, diphthong dynamics,
breathiness) two-tailed against perceived_mean and true_kinsey, with BH-FDR within each
target's family.

Purpose: (a) see which literature cues fail here (challenge by non-replication), and
(b) surface any non-canonical feature that unexpectedly tracks the targets (challenge by
discovery). Every hit here is a CANDIDATE for a future study / classifier input, never a
confirmed result on this n=50 sample. Effect size + bootstrap CI lead (decision A3).

Outputs:
  outputs/tables/segmental_explore.csv           feature x perceived/actual rho/p/q
  outputs/figures/segmental/explore_volcano.png  rho vs -log10(q), both targets
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, false_discovery_control
from adjustText import adjust_text

from common import SPEAKERS_CSV, PROC, FIG, TABLES, ensure_dirs

SEG_FIG = FIG / 'segmental'
KINSEY = 'Kinsey Scale (1-5)'
Q = 0.10
N_BOOT = 2000
SEED = 0
MIN_N = 30          # need at least this many non-null speakers to test a feature


def _fast_spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return (ra * rb).sum() / d if d else np.nan


def corr_ci(x, y, rng):
    rho, p = spearmanr(x, y)
    n = len(x)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = np.array([_fast_spearman(x[i], y[i]) for i in idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return rho, p, lo, hi


def sweep(df, feat_cols, tcol, rng):
    rows = []
    for c in feat_cols:
        d = df[[c, tcol]].dropna()
        if len(d) < MIN_N or d[c].std() == 0:
            rows.append({'feature': c, 'rho': np.nan, 'p': np.nan,
                         'ci_lo': np.nan, 'ci_hi': np.nan, 'n': len(d)})
            continue
        rho, p, lo, hi = corr_ci(d[c].to_numpy(float), d[tcol].to_numpy(float), rng)
        rows.append({'feature': c, 'rho': rho, 'p': p, 'ci_lo': lo,
                     'ci_hi': hi, 'n': len(d)})
    out = pd.DataFrame(rows)
    ok = out['p'].notna()
    out['q'] = np.nan
    out.loc[ok, 'q'] = false_discovery_control(out.loc[ok, 'p'].to_numpy(), method='bh')
    out['hit'] = out['q'] <= Q
    return out


def volcano(perc, act, path):
    ensure_dirs(SEG_FIG)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for ax, tab, title in [(axes[0], perc, 'vs PERCEIVED'),
                           (axes[1], act, 'vs ACTUAL (Kinsey)')]:
        d = tab.dropna(subset=['rho', 'q']).copy()
        d['nlq'] = -np.log10(d['q'].clip(lower=1e-6))
        hit = d['hit']
        ax.scatter(d.loc[~hit, 'rho'], d.loc[~hit, 'nlq'], s=28,
                   color='#9fb2c4', alpha=0.8)
        ax.scatter(d.loc[hit, 'rho'], d.loc[hit, 'nlq'], s=55, color='#c1553b')
        ax.axhline(-np.log10(Q), color='#888', ls='--', lw=1)
        ax.text(ax.get_xlim()[0], -np.log10(Q), f' q={Q}', fontsize=8,
                color='#555', va='bottom')
        ax.axvline(0, color='#ccc', lw=0.8)
        # label EVERY BH-significant (red) feature, plus top-5 near-misses for context
        to_label = pd.concat([
            d[d['hit']],
            d.reindex(d['q'].sort_values().index).head(5),
        ]).drop_duplicates('feature')
        texts = [ax.text(r['rho'], r['nlq'], r['feature'], fontsize=7,
                         color='#c1553b' if r['hit'] else '#666',
                         fontweight='bold' if r['hit'] else 'normal')
                 for _, r in to_label.iterrows()]
        adjust_text(texts, ax=ax, expand=(1.3, 1.6),
                    arrowprops=dict(arrowstyle='-', color='#bbb', lw=0.5))
        ax.set_xlabel('Spearman rho')
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel('-log10(BH q)')
    fig.suptitle('Exploratory segmental sweep (red = BH-significant at q<=%.2f)\n'
                 'hypothesis-generating only -- candidates for replication/classifier'
                 % Q, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ensure_dirs(TABLES, SEG_FIG)
    print('05_segmental_explore.py  (EXPLORATORY -- not confirmatory)')
    sp = pd.read_csv(SPEAKERS_CSV)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv')
    unmatched = sorted(set(seg['file_id']) - set(sp['file_id']))
    if unmatched:
        print(f'  WARNING: {len(unmatched)} segmental speakers do not match Master '
              f'(silently dropped from n): {unmatched}')
    df = sp.merge(seg, on='file_id', how='left')
    feat_cols = [c for c in seg.columns if c != 'file_id']
    print(f'  {len(feat_cols)} segmental features x 2 targets')

    rng = np.random.default_rng(SEED)
    perc = sweep(df, feat_cols, 'perceived_mean', rng)
    rng = np.random.default_rng(SEED)
    act = sweep(df, feat_cols, KINSEY, rng)

    merged = perc.add_suffix('_perceived').rename(
        columns={'feature_perceived': 'feature'}).merge(
        act.add_suffix('_actual').rename(columns={'feature_actual': 'feature'}),
        on='feature')
    merged = merged.reindex(
        merged['q_perceived'].fillna(1).sort_values().index).reset_index(drop=True)
    merged.to_csv(TABLES / 'segmental_explore.csv', index=False)
    volcano(perc, act, SEG_FIG / 'explore_volcano.png')

    def top(tab, tname):
        h = tab[tab['hit']].reindex(tab[tab['hit']]['q'].sort_values().index)
        print(f'  {tname}: {len(h)} features survive BH (q<=%.2f)' % Q)
        for _, r in h.iterrows():
            print(f'    {r.feature:22} rho={r.rho:+.3f}  q={r.q:.3f}  n={int(r.n)}')
    top(perc, 'PERCEIVED')
    top(act, 'ACTUAL')
    print('  -> segmental_explore.csv + explore_volcano.png')
    print('done.')


if __name__ == '__main__':
    main()
