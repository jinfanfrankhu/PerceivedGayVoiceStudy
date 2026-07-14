"""08_ablation.py  --  targeted feature ablation: WHICH cues carry the perceived signal?

07/07c already answered "is there a multivariate signal?" three ways (Ridge, Lasso,
Elastic Net all agree: PERCEIVED is predictable from the 13 confirmatory features,
ACTUAL is not). This script asks the next, more interesting question -- the mechanism:
of those 13 cues, WHICH phonetic families actually do the work?

We decompose the 13 confirmatory features into 5 theory-motivated BLOCKS (sibilant /s/,
vowels, diphthongs, pitch/F0, phonation) and run three complementary passes:

  1. BLOCK-ALONE      -- each block on its own. Its STANDALONE predictive power.
  2. LEAVE-ONE-BLOCK-OUT -- full 13 minus one block. Its UNIQUE contribution beyond the
                           rest, measured as the drop  Δρ = ρ(full) − ρ(without block).
  3. LEAVE-ONE-FEATURE-OUT -- the same at single-feature granularity (finest view).

Read together, alone-vs-removal separates ROBUST non-redundant cues ("predicts alone AND
its removal hurts") from REDUNDANT ones ("predicts alone but removal is free because a
correlated cue covers for it"). That distinction is the paper's mechanism paragraph.

Two inference tools, one per question (compute is cheap here, so we do both properly):
  * "Does this block beat chance?"  (alone)  -> LABEL-PERMUTATION null, exactly as in 07:
    shuffle the labels, re-run the whole LOOCV 1000x, place the real ρ against that.
  * "Does removing this block reliably hurt?" (leave-out) -> BOOTSTRAP CI on Δρ. A 10-feature
    leave-out model still beats chance even if the dropped block mattered, so "vs chance" is
    the wrong test. Instead we resample the 50 speakers and watch how much the drop wobbles;
    if the 95% CI stays above 0, removal reliably hurts. We bootstrap the fixed out-of-fold
    PREDICTIONS (not re-run LOOCV inside each resample) -> no train/test duplicate leakage;
    this captures evaluation uncertainty (the standard way to CI a test metric).

MODELS: Ridge is primary -- it keeps every feature shrunk rather than doing its own internal
selection, so leave-one-out cleanly measures "what the model loses without this block."
Elastic Net is run as a block-level ROBUSTNESS TWIN: if the mechanism claim ("/s/ carries
perceived") holds under a second model too, it is not a Ridge artefact -- exactly the
Ridge+EN invariance our headline already rests on. Single-feature LOFO is Ridge-only (finest
granularity, most exposed to collinearity noise -> Ridge is the right lens).

Framing note: this is a DESCRIPTIVE decomposition of one already-significant model, not 20
fresh hypothesis tests. We lead with effect sizes + CIs, not Bonferroni (which would just
nuke power at n=50). Read the per-block p-values in that spirit.

Both targets run: PERCEIVED (where some block should matter) and ACTUAL as the contrast
(no block should matter -- there is no signal to decompose).

Runtime knobs via env (defaults = the real overnight run):
  ABL_NPERM (1000)  permutation-null draws      ABL_BOOT (5000)  bootstrap resamples
  ABL_ENET  (1)     run the Elastic Net twin    ABL_LOFO (1)     run single-feature LOFO
Set ABL_NPERM=6 ABL_BOOT=50 for a fast end-to-end smoke test.

Outputs (shared prediction/ folder, alongside ridge_* / enet_*):
  outputs/tables/ablation_summary.csv          one row per (model x target x pass x block/feature)
  outputs/tables/ablation_nulls.npz            permutation nulls, for re-plotting without recompute
  outputs/figures/prediction/ablation_ridge.png   block-alone ρ + leave-out Δρ (Ridge)
  outputs/figures/prediction/ablation_enet.png    same, Elastic Net twin
  outputs/figures/prediction/ablation_lofo.png    single-feature leave-one-out Δρ (Ridge)
  outputs/figures/prediction/ablation_compare.png block Δρ, Ridge vs Elastic Net (invariance)
"""
import os
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
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

from common import SPEAKERS_CSV, FEATURES_CSV, PROC, FIG, TABLES, ensure_dirs

PRED_DIR = FIG / 'prediction'
KINSEY = 'Kinsey Scale (1-5)'
TARGETS = {'perceived': 'perceived_mean', 'actual': KINSEY}

SEED = 0
N_PERM = int(os.environ.get('ABL_NPERM', 1000))
BOOT = int(os.environ.get('ABL_BOOT', 5000))
DO_ENET = os.environ.get('ABL_ENET', '1') != '0'
DO_LOFO = os.environ.get('ABL_LOFO', '1') != '0'
N_JOBS = -1

# Ridge grid = 07's; Elastic Net grid = 07c's (kept identical for cross-script comparability).
ALPHAS = np.logspace(-1, 3, 12)
L1_RATIOS = [0.5, 0.9, 0.99]
N_ALPHAS = 30
MAX_ITER = 5000

F0R = 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2'
F0S = 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm'
F0M = 'F0semitoneFrom27.5Hz_sma3nz_amean'
HNR = 'HNRdBACF_sma3nz_amean'

# The pre-registered 13, grouped into theory-motivated construct blocks (order = display order).
BLOCKS = {
    'sibilant /s/': ['S_cog', 'S_skew', 'S_dur'],
    'vowels':       ['vowel_space_area', 'front_f2'],
    'diphthongs':   ['diph_dynamism', 'diph_duration'],
    'pitch/F0':     [F0R, F0S, F0M],
    'phonation':    ['v_cpps', HNR, 'v_h1h2'],
}
SEGMENTAL_CONF = [f for cols in BLOCKS.values() for f in cols]   # the flat 13, block order

LABEL = {'S_cog': 'S cog', 'S_skew': 'S skew', 'S_dur': 'S dur',
         'vowel_space_area': 'vowel space', 'front_f2': 'front F2',
         'diph_dynamism': 'diph dyn', 'diph_duration': 'diph dur',
         F0R: 'F0 range', F0S: 'F0 sd', F0M: 'F0 mean',
         'v_cpps': 'CPPS', HNR: 'HNR', 'v_h1h2': 'H1-H2'}

BLUE, RED, GREY = '#3a6ea5', '#c1553b', '#c9c9c9'


def build_derived(df, tokens):
    """diph_dynamism (mean z of trajlen) + diph_duration -- identical to 04/06/07."""
    tl = ['AY_trajlen', 'EY_trajlen', 'OW_trajlen', 'AW_trajlen']
    z = df[tl].apply(lambda c: (c - c.mean()) / c.std(ddof=1))
    df['diph_dynamism'] = z.mean(axis=1)
    good = tokens[(~tokens['dropped']) & (tokens['base'].isin(['AY', 'EY', 'OW', 'AW']))]
    df['diph_duration'] = df['file_id'].map(good.groupby('file_id')['dur'].mean())
    return df


def make_model(kind):
    """A-priori pipeline, refit from scratch inside every LOOCV fold. Inner CV picks the
    penalty (RidgeCV inner-LOO alpha; ElasticNetCV inner 5-fold alpha+l1_ratio) -> no leak."""
    if kind == 'ridge':
        est = RidgeCV(alphas=ALPHAS)
    else:
        est = ElasticNetCV(l1_ratio=L1_RATIOS, n_alphas=N_ALPHAS, cv=5,
                           max_iter=MAX_ITER, random_state=SEED, n_jobs=1)
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), est)


def loocv_oof(kind, X, y):
    """Out-of-fold predictions: for each speaker, train on the other 49 and predict it."""
    n = len(y)
    oof = np.empty(n)
    idx = np.arange(n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for i in range(n):
            tr = idx != i
            oof[i] = make_model(kind).fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    return oof


def _perm_stat(kind, X, y, seed):
    """One permutation: shuffle labels, re-run the whole LOOCV, return null Spearman."""
    yp = np.random.default_rng(seed).permutation(y)
    return spearmanr(loocv_oof(kind, X, yp), yp).statistic


def permutation_null(kind, X, y):
    """Re-run the whole LOOCV on shuffled labels N_PERM times (parallel) -> null rhos."""
    seeds = np.random.SeedSequence(SEED).spawn(N_PERM)   # reproducible independent streams
    nulls = Parallel(n_jobs=N_JOBS)(delayed(_perm_stat)(kind, X, y, s) for s in seeds)
    return np.asarray(nulls)


def eval_abs(kind, X, y):
    """Absolute performance of a feature set: oof preds, CV rho/R2, permutation null + p."""
    oof = loocv_oof(kind, X, y)
    rho = spearmanr(oof, y).statistic
    r2 = 1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    nulls = permutation_null(kind, X, y)
    p = (1 + int((nulls >= rho).sum())) / (N_PERM + 1)
    return oof, rho, r2, nulls, p


def boot_delta(oof_full, oof_abl, y):
    """Bootstrap CI on Δρ = ρ(full) − ρ(ablated). Resample speakers over the FIXED oof
    predictions (no refit -> no train/test duplicate leakage). >0 means the block helps."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    ds = np.empty(BOOT)
    for b in range(BOOT):
        s = rng.integers(0, n, n)
        ds[b] = (spearmanr(oof_full[s], y[s]).statistic
                 - spearmanr(oof_abl[s], y[s]).statistic)
    ds = ds[~np.isnan(ds)]                       # rare degenerate resample -> drop
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(lo), float(hi), float((ds <= 0).mean())   # last = one-sided boot p


def _row(model, target, pas, name, k, rho, r2, p, drho=np.nan, lo=np.nan, hi=np.nan,
         pd0=np.nan, nulls=None):
    return {'model': model, 'target': target, 'pass': pas, 'name': name, 'n_features': k,
            'cv_spearman': rho, 'cv_r2': r2, 'perm_p': p, 'delta_rho': drho,
            'delta_lo': lo, 'delta_hi': hi, 'p_delta_le0': pd0, '_nulls': nulls}


def run_model(kind, df, do_lofo):
    rows = []
    for tgt, col in TARGETS.items():
        d = df.dropna(subset=[col])
        y = d[col].to_numpy(float)
        Xall = d[SEGMENTAL_CONF].to_numpy(float)

        # --- baseline: the full 13-feature model (the thing we ablate against) ---
        oof_full, rho_full, r2_full, nl_full, p_full = eval_abs(kind, Xall, y)
        rows.append(_row(kind, tgt, 'baseline', 'ALL-13', len(SEGMENTAL_CONF),
                         rho_full, r2_full, p_full, nulls=nl_full))
        print(f"  [{kind:5}] {tgt:10} baseline    ALL-13         "
              f"rho={rho_full:+.3f} R2={r2_full:+.2f} p={p_full:.3f}", flush=True)

        for bname, cols in BLOCKS.items():
            # --- block ALONE: standalone power (beats chance?) ---
            _, rho_a, r2_a, nl_a, p_a = eval_abs(kind, d[cols].to_numpy(float), y)
            rows.append(_row(kind, tgt, 'block-alone', bname, len(cols),
                             rho_a, r2_a, p_a, nulls=nl_a))
            print(f"  [{kind:5}] {tgt:10} alone       {bname:14} "
                  f"rho={rho_a:+.3f} p={p_a:.3f}", flush=True)

            # --- LEAVE block OUT: unique contribution (does removal hurt?) ---
            keep = [c for c in SEGMENTAL_CONF if c not in cols]
            oof_k, rho_k, r2_k, nl_k, p_k = eval_abs(kind, d[keep].to_numpy(float), y)
            lo, hi, pd0 = boot_delta(oof_full, oof_k, y)
            rows.append(_row(kind, tgt, 'leave-block', bname, len(keep),
                             rho_k, r2_k, p_k, rho_full - rho_k, lo, hi, pd0, nl_k))
            print(f"  [{kind:5}] {tgt:10} leave-out   {bname:14} "
                  f"drho={rho_full - rho_k:+.3f} CI[{lo:+.3f},{hi:+.3f}] pboot={pd0:.3f}",
                  flush=True)

        if do_lofo:
            for f in SEGMENTAL_CONF:
                keep = [c for c in SEGMENTAL_CONF if c != f]
                oof_k, rho_k, r2_k, nl_k, p_k = eval_abs(kind, d[keep].to_numpy(float), y)
                lo, hi, pd0 = boot_delta(oof_full, oof_k, y)
                rows.append(_row(kind, tgt, 'leave-feature', LABEL[f], len(keep),
                                 rho_k, r2_k, p_k, rho_full - rho_k, lo, hi, pd0, nl_k))
                print(f"  [{kind:5}] {tgt:10} leave-feat  {LABEL[f]:14} "
                      f"drho={rho_full - rho_k:+.3f} CI[{lo:+.3f},{hi:+.3f}]", flush=True)
    return rows


# ---------------------------------------------------------------- figures ----

def _alone_panel(ax, rows, tgt, xlim):
    order = list(BLOCKS)
    sub = {r['name']: r for r in rows if r['target'] == tgt and r['pass'] == 'block-alone'}
    ys = np.arange(len(order))
    for yi, b in zip(ys, order):
        r = sub[b]
        nl = r['_nulls']
        lo, hi = np.percentile(nl, [5, 95])
        ax.plot([lo, hi], [yi, yi], color=GREY, lw=9, solid_capstyle='butt', zorder=1,
                label='perm null (5-95%)' if yi == 0 else None)
        ax.plot([nl.mean()] * 2, [yi - .25, yi + .25], color='#9a9a9a', lw=1.5, zorder=2)
        beats = r['perm_p'] < 0.05
        ax.scatter([r['cv_spearman']], [yi], s=90, zorder=3,
                   color=BLUE if beats else RED, edgecolor='k', lw=.6,
                   label='observed CV rho' if yi == 0 else None)
        ax.annotate(f"{r['cv_spearman']:.2f} (p={r['perm_p']:.3f})",
                    (r['cv_spearman'], yi), xytext=(6, 7), textcoords='offset points',
                    fontsize=8, color='#28425f' if beats else '#7a2f1f')
    ax.axvline(0, color='#bbb', lw=.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(order)
    ax.set_xlim(*xlim)


def _delta_panel(ax, rows, tgt, pas, order):
    sub = {r['name']: r for r in rows if r['target'] == tgt and r['pass'] == pas}
    ys = np.arange(len(order))
    for yi, b in zip(ys, order):
        r = sub[b]
        d, lo, hi = r['delta_rho'], r['delta_lo'], r['delta_hi']
        hurts = lo > 0
        ax.plot([lo, hi], [yi, yi], color=BLUE if hurts else GREY, lw=4,
                solid_capstyle='round', zorder=1)
        ax.scatter([d], [yi], s=90, zorder=3, color=BLUE if hurts else RED,
                   edgecolor='k', lw=.6)
        ax.annotate(f"{d:+.2f} [{lo:+.2f},{hi:+.2f}]", (d, yi), xytext=(6, 7),
                    textcoords='offset points', fontsize=8,
                    color='#28425f' if hurts else '#7a2f1f')
    ax.axvline(0, color='#888', lw=.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(order)


def fig_blocks(rows, kind, path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ri, tgt in enumerate(['perceived', 'actual']):
        _alone_panel(axes[ri, 0], rows, tgt, (-1.05, 0.9))
        axes[ri, 0].set_title(f'{tgt.upper()} — each block ALONE', fontsize=10)
        _delta_panel(axes[ri, 1], rows, tgt, 'leave-block', list(BLOCKS))
        axes[ri, 1].set_yticklabels([])
        axes[ri, 1].set_title(f'{tgt.upper()} — cost of REMOVING block (Δρ, 95% boot CI)',
                              fontsize=10)
    axes[1, 0].set_xlabel('cross-validated Spearman')
    axes[1, 1].set_xlabel('Δρ = ρ(full) − ρ(without block)   (>0 = block contributes)')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'Targeted feature ablation — {kind.upper()} (n=50)\n'
                 'left: standalone power (vs perm null) · right: unique contribution '
                 '(blue = 95% CI excludes 0)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_lofo(rows, path):
    order = [LABEL[f] for f in SEGMENTAL_CONF]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, tgt in zip(axes, ['perceived', 'actual']):
        _delta_panel(ax, rows, tgt, 'leave-feature', order)
        ax.set_title(f'{tgt.upper()}', fontsize=11)
        ax.set_xlabel('Δρ = ρ(full) − ρ(without feature)   (>0 = feature contributes)')
    fig.suptitle('Single-feature ablation — Ridge, leave-one-feature-out (n=50)\n'
                 'blue = 95% bootstrap CI excludes 0', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_compare(rows, path):
    order = list(BLOCKS)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, tgt in zip(axes, ['perceived', 'actual']):
        ys = np.arange(len(order))
        for kind, off, color in [('ridge', 0.16, BLUE), ('enet', -0.16, '#b5651d')]:
            sub = {r['name']: r for r in rows
                   if r['model'] == kind and r['target'] == tgt and r['pass'] == 'leave-block'}
            for yi, b in zip(ys, order):
                r = sub[b]
                ax.plot([r['delta_lo'], r['delta_hi']], [yi + off] * 2, color=color, lw=3,
                        solid_capstyle='round')
                ax.scatter([r['delta_rho']], [yi + off], s=60, color=color, edgecolor='k',
                           lw=.5, label=kind.upper() if yi == 0 else None)
        ax.axvline(0, color='#888', lw=.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(order)
        ax.set_title(f'{tgt.upper()}', fontsize=11)
        ax.set_xlabel('Δρ (cost of removing block, 95% CI)')
    axes[0].legend(fontsize=9, loc='lower right')
    fig.suptitle('Model-invariance of the ablation: Ridge vs Elastic Net\n'
                 'agreement across models = the mechanism is not a single-model artefact',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def dump(rows):
    """Persist CSV + nulls after each model so partial results survive a long run."""
    out = pd.DataFrame([{k: v for k, v in r.items() if k != '_nulls'} for r in rows])
    out = out[['model', 'target', 'pass', 'name', 'n_features', 'cv_spearman', 'cv_r2',
               'perm_p', 'delta_rho', 'delta_lo', 'delta_hi', 'p_delta_le0']]
    out.to_csv(TABLES / 'ablation_summary.csv', index=False)
    nz = {}
    for r in rows:
        if r['_nulls'] is not None:
            key = f"{r['model']}__{r['target']}__{r['pass']}__{r['name']}"
            nz[key.replace('/', '-').replace(' ', '_')] = r['_nulls']
    np.savez_compressed(TABLES / 'ablation_nulls.npz', **nz)


def main():
    ensure_dirs(TABLES, PRED_DIR)
    print(f"08_ablation.py  (N_PERM={N_PERM} BOOT={BOOT} enet={DO_ENET} lofo={DO_LOFO})",
          flush=True)
    sp = pd.read_csv(SPEAKERS_CSV)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv')
    tokens = pd.read_csv(PROC / 'segmental_tokens.csv')
    df = sp.merge(seg, on='file_id', how='left', suffixes=('', '_seg'))
    df = build_derived(df, tokens)

    all_rows = []

    # Ridge first (fast) -> its figures + tables land on disk before the slow EN twin runs.
    print('-- Ridge (primary: all three passes) --', flush=True)
    ridge_rows = run_model('ridge', df, do_lofo=DO_LOFO)
    all_rows += ridge_rows
    dump(all_rows)
    fig_blocks(ridge_rows, 'ridge', PRED_DIR / 'ablation_ridge.png')
    if DO_LOFO:
        fig_lofo(ridge_rows, PRED_DIR / 'ablation_lofo.png')
    print('  -> ridge tables + figures written', flush=True)

    if DO_ENET:
        print('-- Elastic Net (robustness twin: block passes only) --', flush=True)
        enet_rows = run_model('enet', df, do_lofo=False)
        all_rows += enet_rows
        dump(all_rows)
        fig_blocks(enet_rows, 'enet', PRED_DIR / 'ablation_enet.png')
        fig_compare(all_rows, PRED_DIR / 'ablation_compare.png')
        print('  -> enet tables + figures written', flush=True)

    print('  -> tables/ablation_summary.csv + tables/ablation_nulls.npz', flush=True)
    print('done.', flush=True)


if __name__ == '__main__':
    main()
