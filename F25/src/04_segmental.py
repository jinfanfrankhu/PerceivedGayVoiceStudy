"""04_segmental.py  --  CONFIRMATORY test of the pre-registered segmental hypotheses.

Pre-registration in docs/DECISIONS.md section D. Directions were committed from the
lit review BEFORE testing, so every test is ONE-TAILED in the predicted direction —
the power payoff for 02's null (few tests, directional). We lead with effect size
(rho + bootstrap CI); one-tailed p and BH-FDR q (within each target's family) support.

Two targets, corrected separately:
  perceived_mean  -- what listeners respond to  (perception literature)
  true_kinsey     -- what gay men actually produce (production literature)
Group A features predict the SAME sign for both; Group B predict OPPOSITE signs
(pitch/breathiness stereotype vs Holmes 2024 production reality) -- the divergence
is the paper's core claim.

Outputs:
  outputs/tables/segmental_confirmatory.csv          full 50, side-by-side
  outputs/tables/segmental_confirmatory_sens.csv     transmen-excluded re-run
  outputs/figures/segmental/confirmatory_forest.png  rho + CI, both targets
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, false_discovery_control

from common import SPEAKERS_CSV, PROC, FIG, TABLES, ensure_dirs

SEG_FIG = FIG / 'segmental'
KINSEY = 'Kinsey Scale (1-5)'
Q = 0.10
N_BOOT = 2000
SEED = 0

F0R = 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2'
F0S = 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm'
F0M = 'F0semitoneFrom27.5Hz_sma3nz_amean'
HNR = 'HNRdBACF_sma3nz_amean'

# pre-registered family: (label, column, group, sign_perceived, sign_actual)
FEATURES = [
    ('S_cog',              'S_cog',            'A', +1, +1),
    ('S_skew',             'S_skew',           'A', -1, -1),
    ('S_dur',              'S_dur',            'A', +1, +1),
    ('vowel_space_area',   'vowel_space_area', 'A', +1, +1),
    ('front_f2',           'front_f2',         'A', +1, +1),
    ('diph_dynamism',      'diph_dynamism',    'A', +1, +1),
    ('diph_duration',      'diph_duration',    'A', +1, +1),
    ('F0_range',           F0R,                'B', +1, -1),
    ('F0_stddev',          F0S,                'B', +1, -1),
    ('F0_mean',            F0M,                'B', +1, -1),
    ('v_cpps',             'v_cpps',           'B', -1, +1),
    ('HNR',                HNR,                'B', -1, +1),
    ('v_h1h2',             'v_h1h2',           'B', +1, -1),
]
TARGETS = [('perceived', 'perceived_mean', 3), ('actual', KINSEY, 4)]


def _fast_spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return (ra * rb).sum() / d if d else np.nan


def one_tailed(x, y, sign, rng):
    """Spearman rho, one-tailed p in predicted `sign` direction, bootstrap 95% CI."""
    rho, p_two = spearmanr(x, y)
    # one-tailed toward the predicted sign
    p_one = p_two / 2 if np.sign(rho) == sign else 1 - p_two / 2
    n = len(x)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = np.array([_fast_spearman(x[i], y[i]) for i in idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return rho, p_one, lo, hi


def build_derived(df, tokens):
    """diphthong dynamism (mean z-scored trajlen) + diphthong duration."""
    tl = ['AY_trajlen', 'EY_trajlen', 'OW_trajlen', 'AW_trajlen']
    z = df[tl].apply(lambda c: (c - c.mean()) / c.std(ddof=1))
    df['diph_dynamism'] = z.mean(axis=1)
    good = tokens[(~tokens['dropped']) & (tokens['base'].isin(['AY', 'EY', 'OW', 'AW']))]
    dur = good.groupby('file_id')['dur'].mean()
    df['diph_duration'] = df['file_id'].map(dur)
    return df


def run(df, tag):
    rng = np.random.default_rng(SEED)
    rows = []
    for label, col, grp, s_perc, s_act in FEATURES:
        r = {'feature': label, 'group': grp}
        for tname, tcol, sign in [('perceived', 'perceived_mean', s_perc),
                                  ('actual', KINSEY, s_act)]:
            d = df[[col, tcol]].dropna()
            x = d[col].to_numpy(float); y = d[tcol].to_numpy(float)
            rho, p1, lo, hi = one_tailed(x, y, sign, rng)
            r[f'pred_{tname}'] = '+' if sign > 0 else '-'
            r[f'rho_{tname}'] = rho
            r[f'ci_lo_{tname}'] = lo
            r[f'ci_hi_{tname}'] = hi
            r[f'p_{tname}'] = p1
            r[f'n_{tname}'] = len(d)
        rows.append(r)
    out = pd.DataFrame(rows)
    # BH-FDR within each target's family separately
    for tname in ('perceived', 'actual'):
        out[f'q_{tname}'] = false_discovery_control(out[f'p_{tname}'].to_numpy(),
                                                    method='bh')
        out[f'confirmed_{tname}'] = out[f'q_{tname}'] <= Q
    return out


def forest(tab, path):
    ensure_dirs(SEG_FIG)
    order = tab.iloc[::-1].reset_index(drop=True)   # top feature at top
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    for ax, tname, title in [(axes[0], 'perceived', 'vs PERCEIVED (listeners)'),
                             (axes[1], 'actual', 'vs ACTUAL (true Kinsey)')]:
        for yi, (_, row) in zip(y, order.iterrows()):
            rho = row[f'rho_{tname}']
            lo, hi = row[f'ci_lo_{tname}'], row[f'ci_hi_{tname}']
            conf = row[f'confirmed_{tname}']
            col = '#c1553b' if row['group'] == 'B' else '#3a6ea5'
            ax.plot([lo, hi], [yi, yi], color=col, lw=2, alpha=0.7, zorder=1)
            ax.scatter([rho], [yi], s=70, color=col if conf else 'white',
                       edgecolor=col, linewidth=1.8, zorder=2)
            # predicted-direction arrow marker at left margin
            ax.annotate(row[f'pred_{tname}'], (ax.get_xlim()[0], yi),
                        fontsize=8, color=col, va='center')
        ax.axvline(0, color='#999', lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Spearman rho (one-tailed test, filled = BH-confirmed q<=%.2f)' % Q)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f'{f}  [{g}]' for f, g in
                             zip(order['feature'], order['group'])], fontsize=9)
    fig.suptitle('Confirmatory segmental hypotheses  (blue=Group A agree, '
                 'red=Group B diverge)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _summary(tab, label):
    cp = int(tab['confirmed_perceived'].sum())
    ca = int(tab['confirmed_actual'].sum())
    div = tab[(tab.group == 'B') & tab.confirmed_perceived & ~tab.confirmed_actual]
    print(f'  {label}: confirmed  perceived={cp}/13  actual={ca}/13')
    print(f'    Group-B divergence (perceived-only): '
          f'{list(div["feature"]) if len(div) else "none"}')


def main():
    ensure_dirs(TABLES, SEG_FIG)
    print('04_segmental.py')
    sp = pd.read_csv(SPEAKERS_CSV)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv')
    tokens = pd.read_csv(PROC / 'segmental_tokens.csv')
    unmatched = sorted(set(seg['file_id']) - set(sp['file_id']))
    if unmatched:
        print(f'  WARNING: {len(unmatched)} segmental speakers do not match Master '
              f'(silently dropped from n): {unmatched}')
    df = sp.merge(seg, on='file_id', how='left')
    df = build_derived(df, tokens)

    full = run(df, 'full')
    full.to_csv(TABLES / 'segmental_confirmatory.csv', index=False)
    forest(full, SEG_FIG / 'confirmatory_forest.png')
    print('  [1] full 50 -> segmental_confirmatory.csv + confirmatory_forest.png')
    _summary(full, 'full 50')

    excl = run(df[~df['is_transman']], 'excl')
    excl.to_csv(TABLES / 'segmental_confirmatory_sens.csv', index=False)
    print(f'  [2] sensitivity (n={int((~df.is_transman).sum())}) '
          f'-> segmental_confirmatory_sens.csv')
    _summary(excl, 'excl transmen')
    print('done.')


if __name__ == '__main__':
    main()
