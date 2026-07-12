"""03_inference.py  --  feature <-> target inference.  Run after build_dataset.py.

For each of the 88 acoustic features, how strongly does it relate to
  (a) ACTUAL orientation  (true Kinsey, ordinal 1-5)   and
  (b) PERCEIVED gayness    (crowd's perceived_mean)?

Design decisions (see F25/README.md and the analysis-plan memo):

  * Spearman rank correlation is primary (Kinsey is ordinal; ~half the features
    fail Shapiro-Wilk normality). Pearson is reported alongside, never as the
    headline.
  * We lead with the EFFECT SIZE (rho + a bootstrap 95% CI). p / q are support.
    At n=50 a non-significant ACTUAL feature means "couldn't confirm", NOT "no
    relationship" -- so the CI matters more than the star.
  * Multiple comparisons: Benjamini-Hochberg FDR at q=0.10, applied SEPARATELY
    within each target's family of 88 tests (actual and perceived are corrected
    in isolation -- the divergence between them is the whole point, so they must
    not be pooled).
  * Divergence here is DESCRIPTIVE ONLY: we flag "relates to perceived but not
    actual". We do NOT claim actual and perceived formally differ -- comparing
    two significance verdicts is a fallacy. The rigorous difference-of-
    correlations test is a deferred TODO.
  * Sensitivity: the identical code path is re-run excluding the 2 transmen
    (is_transman), whose Kinsey scores are the ones most worth stress-testing.

Artifacts:
  outputs/tables/feature_inference.csv              full 50, side-by-side
  outputs/tables/feature_inference_sensitivity.csv  full vs transmen-excluded
  outputs/figures/inference/divergence.png          full 50
  outputs/figures/inference/divergence_sensitivity.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata, false_discovery_control

from common import SPEAKERS_CSV, FEATURES_CSV, FIG, TABLES, ensure_dirs

INF_DIR = FIG / 'inference'

KINSEY = 'Kinsey Scale (1-5)'
TARGETS = {'actual': KINSEY, 'perceived': 'perceived_mean'}

Q = 0.10          # BH-FDR false-discovery rate we tolerate
N_BOOT = 2000     # bootstrap resamples for the rho confidence interval
SEED = 0


# ----------------------------------------------------------------------------
# effect size + uncertainty for a single feature/target pair
# ----------------------------------------------------------------------------
def _fast_spearman(a, b):
    """Spearman via Pearson-on-ranks (fast enough to bootstrap 2000x x 88)."""
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return (ra * rb).sum() / denom if denom else np.nan


def correlate_one(x, y, rng):
    """Return Spearman rho, its 95% bootstrap CI, p, and Pearson r/p."""
    rho, p = spearmanr(x, y)
    r, pr = pearsonr(x, y)
    n = len(x)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = np.array([_fast_spearman(x[i], y[i]) for i in idx])
    boot = boot[np.isfinite(boot)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return rho, ci_lo, ci_hi, p, r, pr


def correlate_target(df, feats, target_col, rng):
    """One row per feature: rho / CI / p (Spearman) + Pearson r/p, vs target."""
    rows = []
    for c in feats:
        d = df[[c, target_col]].dropna()
        x = d[c].to_numpy(float)
        y = d[target_col].to_numpy(float)
        if len(d) < 5 or x.std() == 0:
            rows.append({'feature': c, 'rho': np.nan, 'ci_lo': np.nan,
                         'ci_hi': np.nan, 'p': np.nan, 'pearson_r': np.nan,
                         'pearson_p': np.nan, 'n': len(d)})
            continue
        rho, lo, hi, p, r, pr = correlate_one(x, y, rng)
        rows.append({'feature': c, 'rho': rho, 'ci_lo': lo, 'ci_hi': hi,
                     'p': p, 'pearson_r': r, 'pearson_p': pr, 'n': len(d)})
    out = pd.DataFrame(rows)
    # BH-FDR across this target's 88 tests, in isolation.
    ok = out['p'].notna()
    out['q'] = np.nan
    out.loc[ok, 'q'] = false_discovery_control(out.loc[ok, 'p'].to_numpy(),
                                               method='bh')
    out['flagged'] = out['q'] <= Q
    return out


# ----------------------------------------------------------------------------
# side-by-side table: actual vs perceived
# ----------------------------------------------------------------------------
def build_table(speakers, feats, rng):
    parts = {}
    for name, col in TARGETS.items():
        t = correlate_target(speakers, feats, col, rng)
        parts[name] = t.set_index('feature').add_suffix(f'_{name}')
    table = pd.concat(parts.values(), axis=1).reset_index()
    # descriptive divergence: relates to perceived (BH-flagged) but not actual.
    table['divergent'] = table['flagged_perceived'] & ~table['flagged_actual']
    table = table.sort_values('rho_perceived', key=lambda s: s.abs(),
                              ascending=False).reset_index(drop=True)
    return table


# ----------------------------------------------------------------------------
# divergence figure
# ----------------------------------------------------------------------------
def divergence_figure(table, path, title):
    ensure_dirs(INF_DIR)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                   gridspec_kw={'width_ratios': [1, 1.15]})

    # --- Panel A: rho_actual vs rho_perceived, every feature ---
    ra, rp = table['rho_actual'], table['rho_perceived']
    div = table['divergent']
    ax1.axhline(0, color='#bbbbbb', lw=0.8)
    ax1.axvline(0, color='#bbbbbb', lw=0.8)
    lim = np.nanmax(np.abs(np.concatenate([ra, rp]))) * 1.15
    ax1.plot([-lim, lim], [-lim, lim], 'k:', alpha=0.5, label='equal (y = x)')
    ax1.scatter(ra[~div], rp[~div], s=28, color='#9fb2c4', alpha=0.8,
                label='other features')
    # only draw the divergent-feature series (and its legend entry) if any exist,
    # else the legend advertises a category with no points in it.
    if div.any():
        ax1.scatter(ra[div], rp[div], s=48, color='#c1553b',
                    label='perceived-only (BH)')
        for _, row in table[div].iterrows():
            ax1.annotate(row['feature'][:24],
                         (row['rho_actual'], row['rho_perceived']),
                         fontsize=6, xytext=(4, 2), textcoords='offset points')
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel('Spearman rho  vs  ACTUAL (true Kinsey)')
    ax1.set_ylabel('Spearman rho  vs  PERCEIVED (crowd mean)')
    ax1.set_title('Each feature: actual vs perceived association\n'
                  '(points far from the diagonal diverge)', fontsize=10)
    ax1.legend(fontsize=8, loc='lower right')

    # --- Panel B: top features by |rho_perceived|, paired bars ---
    top = table.head(15).iloc[::-1]
    ypos = np.arange(len(top))
    ax2.barh(ypos + 0.2, top['rho_perceived'], height=0.38,
             color='#3a6ea5', label='vs perceived')
    ax2.barh(ypos - 0.2, top['rho_actual'], height=0.38,
             color='#b8875b', label='vs actual')
    # mark BH-significant bars with a star at the bar tip
    for y, (_, row) in zip(ypos, top.iterrows()):
        if row['flagged_perceived']:
            ax2.text(row['rho_perceived'], y + 0.2, ' *', va='center',
                     fontsize=11, color='#28425f')
        if row['flagged_actual']:
            ax2.text(row['rho_actual'], y - 0.2, ' *', va='center',
                     fontsize=11, color='#6e4e2e')
    ax2.axvline(0, color='k', lw=0.8)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f[:32] for f in top['feature']], fontsize=7)
    ax2.set_xlabel('Spearman rho')
    ax2.set_title('15 features most associated with perceived gayness\n'
                  '(* = BH-FDR significant at q=%.2f)' % Q, fontsize=10)
    ax2.legend(fontsize=8, loc='lower right')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
def _summary(table, label):
    na, npv = int(table['flagged_actual'].sum()), int(table['flagged_perceived'].sum())
    nd = int(table['divergent'].sum())
    best_a = table.loc[table['q_actual'].idxmin()] if table['q_actual'].notna().any() else None
    print(f'  {label}: BH-flagged  actual={na}  perceived={npv}  '
          f'perceived-only(divergent)={nd}')
    if best_a is not None:
        print(f'    strongest actual feature: {best_a["feature"][:34]} '
              f'(rho={best_a["rho_actual"]:.2f}, q={best_a["q_actual"]:.3f})')


def main():
    ensure_dirs(FIG, TABLES, INF_DIR)
    print('03_inference.py')
    speakers = pd.read_csv(SPEAKERS_CSV)
    feats = [c for c in pd.read_csv(FEATURES_CSV).columns if c != 'ID']

    # --- full 50 ---
    rng = np.random.default_rng(SEED)
    full = build_table(speakers, feats, rng)
    full.to_csv(TABLES / 'feature_inference.csv', index=False)
    divergence_figure(full, INF_DIR / 'divergence.png',
                      'Feature associations: actual vs perceived  (all 50 speakers)')
    print(f'  [1] full 50 -> feature_inference.csv  + divergence.png')
    _summary(full, 'full 50')

    # --- sensitivity: identical path, transmen excluded ---
    excl_df = speakers[~speakers['is_transman']].copy()
    rng = np.random.default_rng(SEED)          # same seed -> comparable CIs
    excl = build_table(excl_df, feats, rng)
    divergence_figure(excl, INF_DIR / 'divergence_sensitivity.png',
                      'Feature associations  (2 transmen excluded, n=%d)' % len(excl_df))

    # comparison table: full vs excluded, focused on the actual-Kinsey shift
    keep = ['feature', 'rho_actual', 'q_actual', 'rho_perceived', 'q_perceived']
    comp = full[keep].merge(excl[keep], on='feature',
                            suffixes=('_full', '_excl'))
    comp['d_rho_actual'] = comp['rho_actual_excl'] - comp['rho_actual_full']
    comp['d_rho_perceived'] = comp['rho_perceived_excl'] - comp['rho_perceived_full']
    comp = comp.reindex(comp['d_rho_actual'].abs().sort_values(ascending=False).index)
    comp.to_csv(TABLES / 'feature_inference_sensitivity.csv', index=False)
    print(f'  [2] sensitivity (n={len(excl_df)}) -> '
          f'feature_inference_sensitivity.csv  + divergence_sensitivity.png')
    _summary(excl, f'excl transmen (n={len(excl_df)})')
    biggest = comp.iloc[0]
    print(f'    largest actual-rho shift from dropping transmen: '
          f'{biggest["feature"][:30]} '
          f'({biggest["rho_actual_full"]:.2f} -> {biggest["rho_actual_excl"]:.2f})')
    print('done.')


if __name__ == '__main__':
    main()
