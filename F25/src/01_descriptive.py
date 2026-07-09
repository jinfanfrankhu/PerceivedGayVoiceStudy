"""01_descriptive.py  --  descriptive figures + tables.  Run after build_dataset.py.

Produces four groups of artifacts:

  1. outputs/figures/rating_histograms/   50 per-speaker rating distributions
     (+ an overview grid), true Kinsey and perceived mean marked, dip-test flag.
  2. outputs/figures/listener_demographics/   who the 289 raters are.
  3. outputs/figures/diagnostics/perceived_vs_actual.png  (+ tables/)
     the crowd's perceived mean vs the speaker's true Kinsey.
  4. outputs/figures/diagnostics/feature_*   Q-Q / histograms + a Shapiro-Wilk
     normality table, to settle Pearson-vs-Spearman empirically.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, shapiro, probplot

from common import (SPEAKERS_CSV, RATINGS_CLEAN_CSV, FEATURES_CSV,
                    FIG, TABLES, ensure_dirs)

HIST_DIR = FIG / 'rating_histograms'
DEMO_DIR = FIG / 'listener_demographics'
DIAG_DIR = FIG / 'diagnostics'

KINSEY = 'Kinsey Scale (1-5)'


def short(pseudonym):
    return pseudonym.replace('_', ' ')


# ----------------------------------------------------------------------------
# 1. Per-speaker rating histograms
# ----------------------------------------------------------------------------
def rating_histograms(speakers, ratings):
    ensure_dirs(HIST_DIR)
    order = speakers.sort_values('perceived_mean')['file_id'].tolist()

    # --- overview grid (10 rows x 5 cols) ---
    fig, axes = plt.subplots(10, 5, figsize=(18, 26))
    for ax, fid in zip(axes.flat, order):
        row = speakers[speakers.file_id == fid].iloc[0]
        counts = [row[f'rated_{k}'] for k in range(1, 6)]
        bars = ax.bar(range(1, 6), counts, color='#8899aa', width=0.8)
        # true Kinsey (dashed red) and perceived mean (dashed blue)
        ax.axvline(row[KINSEY], color='crimson', ls='--', lw=1.5)
        ax.axvline(row['perceived_mean'], color='steelblue', ls='--', lw=1.5)
        flag = '  *split*' if row['is_split'] else ''
        ax.set_title(f'{short(fid)[:22]}\nK={int(row[KINSEY])} '
                     f'perc={row["perceived_mean"]:.1f}{flag}', fontsize=7.5)
        ax.set_xticks(range(1, 6))
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(order):]:
        ax.axis('off')
    fig.suptitle('Per-speaker rating distributions (sorted by perceived mean)\n'
                 'red -- = true Kinsey,  blue -- = perceived mean', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(HIST_DIR / '_overview_grid.png', dpi=110)
    plt.close(fig)

    # --- one file per speaker ---
    for fid in order:
        row = speakers[speakers.file_id == fid].iloc[0]
        counts = [row[f'rated_{k}'] for k in range(1, 6)]
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.bar(range(1, 6), counts, color='#8899aa', width=0.8)
        ax.axvline(row[KINSEY], color='crimson', ls='--', lw=2,
                   label=f'true Kinsey = {int(row[KINSEY])}')
        ax.axvline(row['perceived_mean'], color='steelblue', ls='--', lw=2,
                   label=f'perceived mean = {row["perceived_mean"]:.2f}')
        verdict = 'listeners SPLIT' if row['is_split'] else 'single-sided'
        ax.set_title(f'{short(fid)}  (n={int(row["n_ratings"])}, '
                     f'SD={row["perceived_sd"]:.2f})\n'
                     f'{verdict}  (straight-camp {row["frac_low"]:.0%}, '
                     f'gay-camp {row["frac_high"]:.0%})', fontsize=9)
        ax.set_xlabel('rating (1 = sounds straight ... 5 = sounds gay)')
        ax.set_ylabel('# listeners')
        ax.set_xticks(range(1, 6))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(HIST_DIR / f'{fid}.png', dpi=110)
        plt.close(fig)
    print(f'  [1] rating histograms -> {HIST_DIR}  (50 + overview grid)')


# ----------------------------------------------------------------------------
# 2. Listener demographics  (one row per listener session)
# ----------------------------------------------------------------------------
def listener_demographics(ratings):
    ensure_dirs(DEMO_DIR)
    per_listener = ratings.groupby('session_id').agg(
        birth_year=('birth_year', 'first'),
        native_eng=('native_eng', 'first'),
        is_prolific=('is_prolific', 'first'),
        n_rated=('rating', 'size'),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    by = per_listener['birth_year'].dropna()
    ax.hist(by, bins=range(int(by.min()), int(by.max()) + 2, 2), color='#5b8db8')
    ax.set_title(f'Listener birth year (n={len(per_listener)} listeners)')
    ax.set_xlabel('birth year')
    ax.set_ylabel('# listeners')

    ax = axes[0, 1]
    ne = per_listener['native_eng'].map({True: 'native', False: 'non-native'}) \
        .value_counts()
    ax.bar(ne.index.astype(str), ne.values, color=['#5b8db8', '#b8875b'])
    ax.set_title('Native English speaker')
    ax.set_ylabel('# listeners')

    ax = axes[1, 0]
    pr = per_listener['is_prolific'].map(
        {True: 'Prolific', False: 'organic'}).fillna('unknown').value_counts()
    ax.bar(pr.index.astype(str), pr.values, color='#7aa66b')
    ax.set_title('Recruitment source')
    ax.set_ylabel('# listeners')

    ax = axes[1, 1]
    ax.hist(per_listener['n_rated'], bins=range(1, per_listener['n_rated'].max() + 2),
            color='#a67ba6')
    ax.set_title('Ratings completed per listener')
    ax.set_xlabel('# speakers rated')
    ax.set_ylabel('# listeners')

    fig.suptitle('Listener demographics', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(DEMO_DIR / 'listener_demographics.png', dpi=120)
    plt.close(fig)
    per_listener.to_csv(TABLES / 'listeners.csv', index=False)
    print(f'  [2] listener demographics -> {DEMO_DIR}  '
          f'({len(per_listener)} listeners)')


# ----------------------------------------------------------------------------
# 3. Perceived vs actual
# ----------------------------------------------------------------------------
def perceived_vs_actual(speakers):
    ensure_dirs(DIAG_DIR, TABLES)
    d = speakers.dropna(subset=[KINSEY, 'perceived_mean']).copy()
    rho, prho = spearmanr(d[KINSEY], d['perceived_mean'])
    r, pr = pearsonr(d[KINSEY], d['perceived_mean'])

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, len(d))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={'width_ratios': [1, 1.1]})
    # scatter with SD error bars
    ax1.errorbar(d[KINSEY] + jitter, d['perceived_mean'], yerr=d['perceived_sd'],
                 fmt='o', ms=6, color='#3a6ea5', ecolor='#bbccdd',
                 elinewidth=1, capsize=2, alpha=0.9)
    ax1.plot([1, 5], [1, 5], 'k:', alpha=0.5, label='perfect agreement')
    ax1.set_xlabel('true Kinsey (self-reported)')
    ax1.set_ylabel('perceived mean (crowd) ± SD')
    ax1.set_title(f'Perceived vs actual\nSpearman r={rho:.3f} (p={prho:.1e}), '
                  f'Pearson r={r:.3f}')
    ax1.set_xticks(range(1, 6))
    ax1.legend(fontsize=8)

    # biggest misses: |perceived_mean - true|
    d['gap'] = d['perceived_mean'] - d[KINSEY]
    top = d.reindex(d['gap'].abs().sort_values(ascending=False).index).head(15)
    colors = ['#b8875b' if g > 0 else '#5b8db8' for g in top['gap']]
    ax2.barh(range(len(top)), top['gap'], color=colors)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels([f'{short(f)[:20]} (K={int(k)})'
                         for f, k in zip(top['file_id'], top[KINSEY])], fontsize=7)
    ax2.axvline(0, color='k', lw=0.8)
    ax2.invert_yaxis()
    ax2.set_xlabel('perceived mean − true Kinsey')
    ax2.set_title('15 biggest perception gaps\n(orange = heard gayer than reported, '
                  'blue = straighter)')
    fig.tight_layout()
    fig.savefig(DIAG_DIR / 'perceived_vs_actual.png', dpi=120)
    plt.close(fig)

    out = d[['file_id', 'initials', KINSEY, 'perceived_mean', 'perceived_sd',
             'perceived_median', 'frac_low', 'frac_high', 'polarization',
             'is_split', 'gap']] \
        .sort_values('gap', key=abs, ascending=False)
    out.to_csv(TABLES / 'perceived_vs_actual.csv', index=False)
    print(f'  [3] perceived vs actual -> {DIAG_DIR}  '
          f'(Spearman r={rho:.3f}, Pearson r={r:.3f})')


# ----------------------------------------------------------------------------
# 4. Feature diagnostics (normality + linearity)
# ----------------------------------------------------------------------------
def feature_diagnostics(speakers):
    ensure_dirs(DIAG_DIR, TABLES)
    feat_cols = [c for c in pd.read_csv(FEATURES_CSV).columns if c != 'ID']

    # Shapiro-Wilk normality per feature (across 50 speakers)
    rows = []
    for c in feat_cols:
        x = speakers[c].dropna().to_numpy()
        try:
            w, p = shapiro(x)
        except Exception:
            w, p = np.nan, np.nan
        rows.append({'feature': c, 'shapiro_W': w, 'shapiro_p': p,
                     'normal_at_.05': bool(p > 0.05) if np.isfinite(p) else np.nan})
    norm = pd.DataFrame(rows).sort_values('shapiro_p')
    norm.to_csv(TABLES / 'feature_normality.csv', index=False)
    n_normal = int(norm['normal_at_.05'].sum())

    # rank features by |Spearman| with perceived_mean, for the visual grid
    d = speakers.dropna(subset=['perceived_mean'])
    strengths = []
    for c in feat_cols:
        rho, _ = spearmanr(d[c], d['perceived_mean'])
        strengths.append((c, abs(rho) if np.isfinite(rho) else 0))
    top = [c for c, _ in sorted(strengths, key=lambda t: -t[1])[:9]]

    # Q-Q grid for the 9 strongest features
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for ax, c in zip(axes.flat, top):
        probplot(speakers[c].dropna(), plot=ax)
        wp = norm.loc[norm.feature == c, 'shapiro_p'].iloc[0]
        ax.set_title(f'{c[:34]}\nShapiro p={wp:.3f}', fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle('Q-Q plots: 9 features most related to perceived gayness\n'
                 '(points on the line = roughly normal)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(DIAG_DIR / 'feature_qq_top9.png', dpi=110)
    plt.close(fig)

    # scatter of those features vs perceived_mean (linearity eyeball)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for ax, c in zip(axes.flat, top):
        ax.scatter(speakers[c], speakers['perceived_mean'], s=22,
                   color='#3a6ea5', alpha=0.8)
        rho, _ = spearmanr(d[c], d['perceived_mean'])
        rr, _ = pearsonr(d[c], d['perceived_mean'])
        ax.set_title(f'{c[:30]}\nSpearman={rho:.2f}  Pearson={rr:.2f}', fontsize=8)
        ax.set_ylabel('perceived mean', fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle('Feature vs perceived gayness (linearity check)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(DIAG_DIR / 'feature_scatter_top9.png', dpi=110)
    plt.close(fig)
    print(f'  [4] feature diagnostics -> {DIAG_DIR}  '
          f'({n_normal}/{len(feat_cols)} features pass Shapiro at .05)')


def main():
    ensure_dirs(FIG, TABLES)
    print('01_descriptive.py')
    speakers = pd.read_csv(SPEAKERS_CSV)
    ratings = pd.read_csv(RATINGS_CLEAN_CSV)
    rating_histograms(speakers, ratings)
    listener_demographics(ratings)
    perceived_vs_actual(speakers)
    feature_diagnostics(speakers)
    print('done.')


if __name__ == '__main__':
    main()
