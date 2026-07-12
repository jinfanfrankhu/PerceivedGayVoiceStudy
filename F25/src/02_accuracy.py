"""02_accuracy.py  --  the perceived-vs-actual "accuracy story" figures.

A Phase-1 companion to 01_descriptive.py, focused entirely on how well the crowd's
perceived gayness tracks the speaker's true Kinsey. Four artifacts:

  1. accuracy/perceived_vs_actual_lsrl.png  scatter + least-squares regression line
     (LSRL) with equation + R^2, y=x perfect-agreement reference for contrast.
  2. accuracy/kinsey_distribution.png        bar chart of true Kinsey across 50 speakers.
  3. accuracy/perceived_mean_hist.png        histogram of perceived means across 50.
  4. accuracy/per_speaker_error.png          grouped bars per speaker (true vs perceived);
     the gap between the paired bars is the signed perception error.

Everything here is descriptive (n=50 speakers). Nothing is a model prediction --
"predicted" = the crowd's perceived mean, "actual" = self-reported Kinsey.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import SPEAKERS_CSV, FIG, TABLES, ensure_dirs

ACC_DIR = FIG / 'accuracy'
KINSEY = 'Kinsey Scale (1-5)'

C_ACTUAL = '#c1553b'      # true Kinsey
C_PERC = '#3a6ea5'        # perceived mean
C_OVER = '#b8875b'        # heard gayer than reported
C_UNDER = '#5b8db8'       # heard straighter than reported


def short(pseudonym):
    return pseudonym.replace('_', ' ')


# ----------------------------------------------------------------------------
# 1. Scatter with a real least-squares regression line
# ----------------------------------------------------------------------------
def scatter_lsrl(d):
    x = d[KINSEY].to_numpy(float)
    y = d['perceived_mean'].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)             # LSRL: y = slope*x + intercept
    rho, prho = spearmanr(x, y)                         # rank-based; robust to bimodality

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, len(d))

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.scatter(x + jitter, y, s=55, color=C_PERC, alpha=0.85, edgecolor='white',
               linewidth=0.5, zorder=3)
    # LSRL across the observed Kinsey range -- descriptive slope only. We do NOT report
    # Pearson r / R^2: Kinsey is bimodal (clusters at 1 and 5), so the linear-association
    # statistic and its normal-theory p are inappropriate. Inference is Spearman.
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, color=C_ACTUAL, lw=2.2, zorder=2,
            label=f'LSRL (descriptive): perceived = {slope:.2f}·Kinsey + {intercept:.2f}')
    ax.set_xlabel('true Kinsey (self-reported, 1=straight ... 5=gay)')
    ax.set_ylabel('perceived mean (crowd)')
    ax.set_title('Perceived vs actual with least-squares fit\n'
                 f'Spearman ρ={rho:.3f} (p={prho:.1e})', fontsize=11)
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.8, 5.2)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(ACC_DIR / 'perceived_vs_actual_lsrl.png', dpi=130)
    plt.close(fig)
    return slope, intercept, rho, prho


# ----------------------------------------------------------------------------
# 2. Distribution of true Kinsey scores
# ----------------------------------------------------------------------------
def kinsey_distribution(d):
    counts = d[KINSEY].round().astype(int).value_counts().reindex(range(1, 6),
                                                                  fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=C_ACTUAL, width=0.75)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(int(v)),
                ha='center', fontsize=9)
    ax.set_xlabel('true Kinsey (self-reported)')
    ax.set_ylabel('# speakers')
    ax.set_title(f'Distribution of actual reported sexuality\n(n={len(d)} speakers)',
                 fontsize=11)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, counts.max() + 2)
    fig.tight_layout()
    fig.savefig(ACC_DIR / 'kinsey_distribution.png', dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. Histogram of perceived means
# ----------------------------------------------------------------------------
def perceived_mean_hist(d):
    y = d['perceived_mean'].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y, bins=np.arange(1, 5.26, 0.25), color=C_PERC, edgecolor='white')
    ax.axvline(y.mean(), color=C_ACTUAL, ls='--', lw=2,
               label=f'grand mean = {y.mean():.2f}')
    ax.set_xlabel('perceived mean (crowd)')
    ax.set_ylabel('# speakers')
    ax.set_title(f'Distribution of perceived means\n(n={len(d)} speakers)', fontsize=11)
    ax.set_xlim(1, 5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ACC_DIR / 'perceived_mean_hist.png', dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Per-speaker two-way (grouped) bars: true Kinsey vs perceived mean
# ----------------------------------------------------------------------------
def per_speaker_error(d):
    s = d.sort_values([KINSEY, 'perceived_mean']).reset_index(drop=True)
    y = np.arange(len(s))
    h = 0.4
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.barh(y + h / 2, s[KINSEY], height=h, color=C_ACTUAL, label='true Kinsey')
    ax.barh(y - h / 2, s['perceived_mean'], height=h, color=C_PERC,
            label='perceived mean')
    # annotate the signed error at the end of each speaker's pair
    gap = s['perceived_mean'] - s[KINSEY]
    for yi, g in zip(y, gap):
        xmax = max(s.loc[yi, KINSEY], s.loc[yi, 'perceived_mean'])
        ax.text(xmax + 0.08, yi, f'{g:+.1f}', va='center', fontsize=6.5,
                color=C_OVER if g > 0 else C_UNDER)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{short(f)[:22]}' for f in s['file_id']], fontsize=6.5)
    ax.set_ylim(-1, len(s))
    ax.set_xlim(0, 5.6)
    ax.set_xlabel('scale value (1=straight ... 5=gay)')
    ax.set_title('Per-speaker accuracy: true Kinsey vs perceived mean\n'
                 '(sorted by Kinsey then perceived; number = perceived − true, '
                 'brown=heard gayer, blue=heard straighter)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.15)
    fig.tight_layout()
    fig.savefig(ACC_DIR / 'per_speaker_error.png', dpi=120)
    plt.close(fig)
    return gap


def main():
    ensure_dirs(ACC_DIR, TABLES)
    print('02_accuracy.py')
    sp = pd.read_csv(SPEAKERS_CSV)
    d = sp.dropna(subset=[KINSEY, 'perceived_mean']).copy()

    slope, intercept, rho, prho = scatter_lsrl(d)
    kinsey_distribution(d)
    perceived_mean_hist(d)
    gap = per_speaker_error(d)

    mae = gap.abs().mean()
    print(f'  [1] LSRL perceived = {slope:.2f}*Kinsey + {intercept:.2f} '
          f'(descriptive line); Spearman rho={rho:.3f} (p={prho:.1e})')
    print(f'  [2] Kinsey distribution + [3] perceived-mean hist + [4] per-speaker bars')
    print(f'      mean |perceived - true| = {mae:.2f}   -> figures/accuracy/')
    print('done.')


if __name__ == '__main__':
    main()
