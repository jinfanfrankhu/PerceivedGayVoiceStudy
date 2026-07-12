"""06_divergence.py  --  the RIGOROUS "acoustics track perceived more than actual" test.

Everything up to here shows divergence the weak way: a feature is BH-significant vs
PERCEIVED but not vs ACTUAL. That is the two-verdict fallacy (a significant and a
non-significant correlation are NOT proven to differ). 06 tests the DIFFERENCE in the
two correlations directly.

For each feature we have two Spearman correlations measured on the SAME 50 speakers:
    rho_perceived = corr(feature, perceived_mean)
    rho_actual    = corr(feature, true Kinsey)
They SHARE the feature variable -> they are *dependent* (overlapping) correlations, so
their difference has a smaller standard error than two independent correlations would.

We test  Delta = rho_perceived - rho_actual  two ways:
  * PRIMARY -- bootstrap: resample the 50 speakers with replacement, recompute BOTH
    Spearman correlations on each resample, and take the percentile CI + two-sided p of
    Delta. Distribution-free, so it inherits none of the normality assumptions that made
    us drop Pearson (Kinsey is bimodal). This is the headline.
  * CROSS-CHECK -- Williams's (1959) / Steiger (1980) test for two dependent correlations
    with a variable in common. It is a NORMAL-THEORY test and we feed it Spearman
    coefficients, so it is only APPROXIMATE here -- reported alongside, never as the
    verdict.

Family = the 13 pre-registered confirmatory features (BH-FDR within that family). The
exploratory volcano hits are shown too but flagged POST-HOC (no FDR claim).

CAVEAT baked into every output: `perceived_mean` is an average of ~69 listener ratings
(low measurement error) while `Kinsey` is a single coarse self-report (higher error).
Classical measurement error ATTENUATES a correlation, so part of any Delta>0 could be
"perceived is measured more reliably", not "acoustics genuinely relate to perception
more". We cannot fully disattenuate (Kinsey has no repeat measure). So read a positive
Delta as *consistent with* perception-biased acoustics, not proof of it.

Outputs:
  outputs/tables/segmental_divergence.csv          per-feature Delta, CI, p, q, Williams
  outputs/figures/segmental/divergence_test.png    Delta forest + rho-perceived vs rho-actual
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, false_discovery_control

from common import SPEAKERS_CSV, PROC, FIG, TABLES, ensure_dirs

SEG_FIG = FIG / 'segmental'
KINSEY = 'Kinsey Scale (1-5)'
Q = 0.10
N_BOOT = 5000
SEED = 0

F0R = 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2'
F0S = 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm'
F0M = 'F0semitoneFrom27.5Hz_sma3nz_amean'
HNR = 'HNRdBACF_sma3nz_amean'

# 13 pre-registered confirmatory features (mirrors 04_segmental.py): (label, column, group)
CONFIRMATORY = [
    ('S_cog',            'S_cog',            'A'),
    ('S_skew',           'S_skew',           'A'),
    ('S_dur',            'S_dur',            'A'),
    ('vowel_space_area', 'vowel_space_area', 'A'),
    ('front_f2',         'front_f2',         'A'),
    ('diph_dynamism',    'diph_dynamism',    'A'),
    ('diph_duration',    'diph_duration',    'A'),
    ('F0_range',         F0R,                'B'),
    ('F0_stddev',        F0S,                'B'),
    ('F0_mean',          F0M,                'B'),
    ('v_cpps',           'v_cpps',           'B'),
    ('HNR',              HNR,                'B'),
    ('v_h1h2',           'v_h1h2',           'B'),
]

# strongest exploratory hits from 05 (post-hoc; shown for context, NOT in the FDR family)
EXPLORATORY = [
    ('AY_z2',           'AY_z2'),
    ('S_cog_singleton', 'S_cog_singleton'),
    ('S_peak',          'S_peak'),
    ('Z_cog',           'Z_cog'),
    ('Z_peak',          'Z_peak'),
    ('AE_z2',           'AE_z2'),
    ('AO_z1',           'AO_z1'),
    ('AW_z1',           'AW_z1'),
    ('EY_z2',           'EY_z2'),
]

GROUP_COLOR = {'A': '#3a6ea5', 'B': '#c1553b', 'explore': '#8a8a8a'}


def _spear(a, b):
    """Spearman via Pearson-on-ranks; ranks recomputed each call (needed in bootstrap)."""
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return (ra * rb).sum() / d if d else np.nan


def _rank_rows(X):
    """Average ranks (scipy 'average' method) along axis=1, for every row at once.

    For value x_i: rank = (#strictly-less) + (#equal + 1)/2  -- the mean of the ordinal
    ranks its tie-group would occupy. Identical to rankdata() per row, but vectorized
    over the whole (N_BOOT, n) bootstrap matrix, replacing 5000 Python rankdata calls."""
    less = (X[:, :, None] > X[:, None, :]).sum(axis=2)        # (B, n) strictly-less count
    equal = (X[:, :, None] == X[:, None, :]).sum(axis=2)      # (B, n) equal count (incl self)
    return less + (equal + 1) / 2.0


def _pearson_rows(A, B):
    """Row-wise Pearson correlation of two (B, n) rank matrices -> (B,) Spearman rhos."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def williams_t(r_jk, r_jh, r_kh, n):
    """Williams (1959) test that two dependent correlations sharing variable j are equal.
    j = feature, k = perceived, h = actual. NORMAL-THEORY -> approximate on Spearman."""
    R = 1 - r_jk**2 - r_jh**2 - r_kh**2 + 2 * r_jk * r_jh * r_kh
    rbar = (r_jk + r_jh) / 2
    num = (r_jk - r_jh) * np.sqrt((n - 1) * (1 + r_kh))
    den = np.sqrt(2 * ((n - 1) / (n - 3)) * R + rbar**2 * (1 - r_kh)**3)
    t = num / den if den else np.nan
    p = 2 * stats.t.sf(abs(t), n - 3) if np.isfinite(t) else np.nan
    return t, p


def divergence_one(feat, perc, act, rng):
    """Delta = rho_perceived - rho_actual, with bootstrap CI/p and Williams cross-check.
    feat/perc/act are already aligned + NaN-free numpy arrays."""
    n = len(feat)
    rho_p = _spear(feat, perc)
    rho_a = _spear(feat, act)
    r_kh = _spear(perc, act)                     # perceived <-> actual on this subset
    delta = rho_p - rho_a
    abs_delta = abs(rho_p) - abs(rho_a)

    idx = rng.integers(0, n, size=(N_BOOT, n))
    rF = _rank_rows(feat[idx])                    # rank once per resample, reused for both
    bp = _pearson_rows(rF, _rank_rows(perc[idx]))
    ba = _pearson_rows(rF, _rank_rows(act[idx]))
    bd = bp - ba
    m = np.isfinite(bd)
    bd = bd[m]
    lo, hi = np.percentile(bd, [2.5, 97.5])
    boot_p = min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))
    abd = np.abs(bp[m]) - np.abs(ba[m])
    alo, ahi = np.percentile(abd, [2.5, 97.5])

    wt, wp = williams_t(rho_p, rho_a, r_kh, n)
    return {
        'rho_perceived': rho_p, 'rho_actual': rho_a, 'r_perc_actual': r_kh,
        'delta': delta, 'ci_lo': lo, 'ci_hi': hi, 'boot_p': boot_p,
        'abs_delta': abs_delta, 'abs_ci_lo': alo, 'abs_ci_hi': ahi,
        'williams_t': wt, 'williams_p': wp, 'n': n,
    }


def build_derived(df, tokens):
    """diph_dynamism (mean z of trajlen) + diph_duration -- identical to 04_segmental.py."""
    tl = ['AY_trajlen', 'EY_trajlen', 'OW_trajlen', 'AW_trajlen']
    z = df[tl].apply(lambda c: (c - c.mean()) / c.std(ddof=1))
    df['diph_dynamism'] = z.mean(axis=1)
    good = tokens[(~tokens['dropped']) & (tokens['base'].isin(['AY', 'EY', 'OW', 'AW']))]
    dur = good.groupby('file_id')['dur'].mean()
    df['diph_duration'] = df['file_id'].map(dur)
    return df


def run(df, rng):
    rows = []
    specs = [(lab, col, grp, 'confirmatory') for lab, col, grp in CONFIRMATORY] + \
            [(lab, col, 'explore', 'exploratory') for lab, col in EXPLORATORY]
    for lab, col, grp, fam in specs:
        d = df[[col, 'perceived_mean', KINSEY]].dropna()
        res = divergence_one(d[col].to_numpy(float),
                             d['perceived_mean'].to_numpy(float),
                             d[KINSEY].to_numpy(float), rng)
        res.update({'feature': lab, 'group': grp, 'family': fam})
        rows.append(res)
    out = pd.DataFrame(rows)
    # BH-FDR on the bootstrap p, corrected SEPARATELY within each family. The confirmatory
    # q is the real claim; the exploratory q is post-hoc (these features were picked BY
    # their perceived correlation) and shown for curiosity only.
    out['q_bh'] = np.nan
    for fam in ('confirmatory', 'exploratory'):
        m = out['family'] == fam
        out.loc[m, 'q_bh'] = false_discovery_control(
            out.loc[m, 'boot_p'].to_numpy(), method='bh')
    # only the confirmatory family can formally "diverge"
    out['diverges'] = (out['family'] == 'confirmatory') & (out['q_bh'] <= Q)
    return out


def figure(tab, path):
    ensure_dirs(SEG_FIG)
    conf = tab[tab.family == 'confirmatory'].sort_values('delta').reset_index(drop=True)
    expl = tab[tab.family == 'exploratory'].sort_values('delta').reset_index(drop=True)
    order = pd.concat([conf, expl], ignore_index=True)
    y = np.arange(len(order))
    sep = len(conf) - 0.5                          # line between the two families

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9),
                                   gridspec_kw={'width_ratios': [1.15, 1]})

    # --- Panel A: Delta forest (rho_perceived - rho_actual), bootstrap 95% CI ---
    for yi, (_, r) in zip(y, order.iterrows()):
        c = GROUP_COLOR[r['group']]
        ax1.plot([r['ci_lo'], r['ci_hi']], [yi, yi], color=c, lw=2, alpha=0.7, zorder=1)
        filled = bool(r['diverges'])
        ax1.scatter([r['delta']], [yi], s=70, color=c if filled else 'white',
                    edgecolor=c, linewidth=1.8, zorder=2)
    ax1.axvline(0, color='#999', lw=1)
    ax1.axhline(sep, color='#ccc', ls=':', lw=1)
    ax1.text(ax1.get_xlim()[1], sep + 0.3,
             'exploratory (post-hoc) above · confirmatory-13 below ',
             fontsize=7, color='#888', ha='right', va='bottom')
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r['feature']} [{r['group']}]" for _, r in order.iterrows()],
                        fontsize=8)
    ax1.set_xlabel('Δ = ρ(feature, perceived) − ρ(feature, actual)      '
                   '(>0 → more perceived-associated)')
    ax1.set_title('Difference-of-correlations, bootstrap 95% CI\n'
                  '(filled = CI excludes 0 AND survives BH-FDR among the 13)', fontsize=10)

    # --- Panel B: rho_perceived vs rho_actual, diagonal = equal association ---
    lim = float(np.nanmax(np.abs(np.concatenate(
        [tab['rho_perceived'], tab['rho_actual']]))) * 1.2)
    ax2.plot([-lim, lim], [-lim, lim], 'k:', alpha=0.5, label='equal (ρ_perc = ρ_act)')
    ax2.axhline(0, color='#ddd', lw=0.8); ax2.axvline(0, color='#ddd', lw=0.8)
    for _, r in tab.iterrows():
        c = GROUP_COLOR[r['group']]
        filled = bool(r['diverges'])
        ax2.scatter([r['rho_actual']], [r['rho_perceived']], s=55,
                    color=c if filled else 'white', edgecolor=c, linewidth=1.6, zorder=3)
        ax2.annotate(r['feature'], (r['rho_actual'], r['rho_perceived']),
                     fontsize=6.5, color='#555', xytext=(4, 2),
                     textcoords='offset points')
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.set_xlabel('ρ  vs  ACTUAL (true Kinsey)')
    ax2.set_ylabel('ρ  vs  PERCEIVED (crowd mean)')
    ax2.set_title('Above the diagonal → tracks perception more\n'
                  '(blue = Group A, red = Group B, grey = exploratory)', fontsize=10)
    ax2.legend(fontsize=8, loc='lower right')

    fig.suptitle('Perceived-vs-actual divergence test  (n=50)   —   caveat: perceived '
                 'is a 69-rating mean (low noise) vs single Kinsey self-report; part of '
                 'Δ>0 may be reliability, not signal', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ensure_dirs(TABLES, SEG_FIG)
    print('06_divergence.py  (rigorous perceived-vs-actual difference test)')
    sp = pd.read_csv(SPEAKERS_CSV)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv')
    tokens = pd.read_csv(PROC / 'segmental_tokens.csv')
    unmatched = sorted(set(seg['file_id']) - set(sp['file_id']))
    if unmatched:
        print(f'  WARNING: {len(unmatched)} segmental speakers unmatched to Master: {unmatched}')
    df = sp.merge(seg, on='file_id', how='left')
    df = build_derived(df, tokens)

    rng = np.random.default_rng(SEED)
    tab = run(df, rng)
    cols = ['feature', 'group', 'family', 'rho_perceived', 'rho_actual', 'r_perc_actual',
            'delta', 'ci_lo', 'ci_hi', 'boot_p', 'q_bh', 'abs_delta', 'abs_ci_lo',
            'abs_ci_hi', 'williams_t', 'williams_p', 'diverges', 'n']
    tab[cols].to_csv(TABLES / 'segmental_divergence.csv', index=False)
    figure(tab, SEG_FIG / 'divergence_test.png')

    hdr = f'  {"feature":16}{"delta":>7}{"95% CI":>18}{"boot p":>9}{"q":>7}{"Williams p":>12}'

    def _block(sub):
        for _, r in sub.iterrows():
            star = ' *' if r['diverges'] else ''
            print(f"  {r.feature:16}{r.delta:+7.2f}  [{r.ci_lo:+.2f},{r.ci_hi:+.2f}]"
                  f"{r.boot_p:9.3f}{r.q_bh:7.2f}{r.williams_p:12.3f}{star}")

    conf = tab[tab.family == 'confirmatory'].sort_values('delta', ascending=False)
    ndiv = int(tab['diverges'].sum())
    print(f'  confirmatory-13: {ndiv} feature(s) diverge (delta CI excludes 0 & BH q<=%.2f)' % Q)
    print(hdr)
    _block(conf)

    expl = tab[tab.family == 'exploratory'].sort_values('delta', ascending=False)
    print('  exploratory (post-hoc; BH corrected within this family only, no confirmatory '
          'claim):')
    print(hdr)
    _block(expl)

    print('  -> tables/segmental_divergence.csv + figures/segmental/divergence_test.png')
    print('done.')


if __name__ == '__main__':
    main()
