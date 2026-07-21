"""10b_wavlm_probe.py -- honest prediction from frozen WavLM embeddings (Ridge + Elastic Net).

Consumes the cache from 10_wavlm_extract.py and runs it through the SAME honest
harness as 07_ridge.py / 07c_elasticnet.py (LOOCV out-of-fold prediction + a
permutation null + cross-validated Spearman, both targets). The flagship question:

    Does a SOTA speech representation BEAT your hand-measured /s/ cue,
    or does it just REDUCE TO it?

TWO heads, exactly as 07 (Ridge) is twinned by 07c (Elastic Net):
  * ridge -- L2, spreads weight across correlated dims: the right tool for a DENSE
    signal, and the primary here.
  * enet  -- L1+L2, wants SPARSITY. On a 768-dim dense embedding this is a direct
    test of the "distributed cue" story: if Ridge succeeds but Elastic Net cannot
    (collapses / underperforms), the perceptual signal is genuinely smeared across
    the representation, not carried by a sparse handful of dimensions.

Two rigor upgrades over 07, both because WavLM gives us 13 layers:
  * WHICH layer predicts best is a hyperparameter -> picking it post-hoc is double-
    dipping. Handled with a MAX-STATISTIC permutation null: each permutation shuffles
    the label ONCE and records all 13 layers' null scores together; the null of the
    *max across layers* is the family-wise null for "the best of 13 layers beats
    chance." Per-layer (uncorrected) p-values are also reported for the profile.
  * S_cog-alone is run through each head as a reference bar -> "beats vs reduces" is
    apples-to-apples within a head.

The money test: take the best layer's out-of-fold PERCEIVED predictions and regress
them on S_cog. High R^2 => the black box rediscovered the sibilant (reduces). WavLM
CV-rho >> S_cog-alone with residual signal => it found more (beats).

Env knobs:  WAVLM_NPERM (1000, ridge)   WAVLM_EN_NPERM (300, enet)   WAVLM_JOBS (all)

Outputs:
  outputs/tables/wavlm_summary.csv      one row per (model x target x layer) + refs
  outputs/tables/wavlm_moneytest.csv    best-layer vs S_cog head-to-head, per model
  outputs/figures/prediction/wavlm/layers.png    per-layer CV-rho profile, both heads
  outputs/figures/prediction/wavlm/vs_scog.png   best-layer OOF pred vs S_cog, both heads
"""
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

from common import SPEAKERS_CSV, PROC, FIG, TABLES, ensure_dirs

PRED_DIR = FIG / 'prediction' / 'wavlm'
EMB_NPZ = PROC / 'wavlm_embeddings.npz'
KINSEY = 'Kinsey Scale (1-5)'
TARGETS = {'perceived': 'perceived_mean', 'actual': KINSEY}

JOBS = int(os.environ.get('WAVLM_JOBS', os.cpu_count() or 1))
SEED = 0
NPERM = {'ridge': int(os.environ.get('WAVLM_NPERM', 1000)),
         'enet': int(os.environ.get('WAVLM_EN_NPERM', 300))}

# ridge: grid widened vs 07 (p=768 >> n needs heavier shrinkage); inner-LOO picks per fold.
ALPHAS = np.logspace(-1, 5, 16)
# enet: lighter inner grid than 07c (cv=3, fewer alphas) -- it is the robustness twin and
# coordinate descent on 768 dims is the bottleneck; l1_ratio spans balanced -> near-Lasso.
EN_L1 = [0.5, 0.9, 0.99]
EN_NALPHAS = 20
EN_MAXITER = 3000

MODELS = ['ridge', 'enet']
LINESTYLE = {'ridge': '-', 'enet': '--'}


def make_model(kind):
    """The single a-priori pipeline for a head, refit from scratch inside every LOOCV fold."""
    head = (RidgeCV(alphas=ALPHAS) if kind == 'ridge'
            else ElasticNetCV(l1_ratio=EN_L1, n_alphas=EN_NALPHAS, cv=3,
                              max_iter=EN_MAXITER, random_state=SEED, n_jobs=1))
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), head)


def loocv_oof(X, y, kind):
    """Out-of-fold predictions: train on the other n-1 speakers, predict each held-out one."""
    n = len(y)
    oof = np.empty(n)
    idx = np.arange(n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for i in range(n):
            tr = idx != i
            oof[i] = make_model(kind).fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    return oof


def cv_score(X, y, kind):
    """Cross-validated Spearman / R^2 / MAE between out-of-fold predictions and truth."""
    oof = loocv_oof(X, y, kind)
    rho = spearmanr(oof, y).statistic
    r2 = 1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()   # OOF R^2 (can be < 0)
    mae = np.abs(y - oof).mean()
    return rho, r2, mae, oof


def _null_chunk(seeds, Xs, y, kind):
    """One worker: for each seed, shuffle y ONCE and score all layers on that shuffle."""
    out = np.empty((len(seeds), len(Xs)))
    for j, s in enumerate(seeds):
        yp = np.random.default_rng(s).permutation(y)
        for L, XL in enumerate(Xs):
            out[j, L] = spearmanr(loocv_oof(XL, yp, kind), yp).statistic
    return out


def layer_permutation_null(Xs, y, kind, n_perm):
    """(n_perm, n_layers) null Spearman matrix; same shuffle shared across layers per row."""
    chunks = [c for c in np.array_split(np.arange(SEED, SEED + n_perm), JOBS) if len(c)]
    res = Parallel(n_jobs=JOBS)(delayed(_null_chunk)(c, Xs, y, kind) for c in chunks)
    return np.vstack(res)


def probe_target(tgt_name, y, Xs, scog, kind, n_perm):
    """Full per-layer probe for one (head, target) + max-stat correction + S_cog reference."""
    n_layers = len(Xs)
    obs = np.array([cv_score(XL, y, kind)[:3] for XL in Xs])    # (n_layers, 3): rho, r2, mae
    obs_rho = obs[:, 0]
    best = int(np.argmax(obs_rho))
    _, _, _, best_oof = cv_score(Xs[best], y, kind)             # keep best layer's OOF preds
    scog_rho, scog_r2, scog_mae, _ = cv_score(scog.reshape(-1, 1), y, kind)

    nulls = layer_permutation_null(Xs, y, kind, n_perm)         # (n_perm, n_layers)
    null_max = nulls.max(axis=1)
    per_layer_p = [(1 + int((nulls[:, L] >= obs_rho[L]).sum())) / (n_perm + 1)
                   for L in range(n_layers)]
    best_fw_p = (1 + int((null_max >= obs_rho[best]).sum())) / (n_perm + 1)

    collapsed = obs_rho[best] <= -0.999                         # enet zeroed everything
    print(f"\n[{kind}/{tgt_name}] best layer = {best} (CV rho={obs_rho[best]:+.3f}, "
          f"family-wise p={best_fw_p:.3f}{'  COLLAPSED' if collapsed else ''})   "
          f"S_cog-alone CV rho={scog_rho:+.3f}")
    for L in range(n_layers):
        star = ' *' if per_layer_p[L] < 0.05 else '  '
        mark = ' <-- best' if L == best else ''
        print(f"   layer {L:2d}  CV rho={obs_rho[L]:+.3f}  R2={obs[L,1]:+.2f}  "
              f"p={per_layer_p[L]:.3f}{star}{mark}")

    return {'model': kind, 'target': tgt_name,
            'obs_rho': obs_rho, 'obs_r2': obs[:, 1], 'obs_mae': obs[:, 2],
            'per_layer_p': per_layer_p, 'nulls': nulls, 'best': best,
            'best_fw_p': best_fw_p, 'best_oof': best_oof, 'collapsed': collapsed,
            'scog_rho': scog_rho, 'scog_r2': scog_r2, 'scog_mae': scog_mae}


def money_test(res, scog):
    """Does the best layer's signal reduce to S_cog?  (headline = perceived target)"""
    pred = res['best_oof']
    r_ps = pearsonr(pred, scog).statistic
    rho_ps = spearmanr(pred, scog).statistic
    r2_on_scog = r_ps ** 2
    delta = res['obs_rho'][res['best']] - res['scog_rho']
    if res['collapsed']:
        verdict = "COLLAPSED: Elastic Net zeroed the embedding (no sparse subset predicts)."
    elif res['best_fw_p'] >= 0.05:                             # gate: no verdict without signal
        verdict = f"NO SIGNAL: best layer does not beat the permutation null (fw p={res['best_fw_p']:.3f})."
    elif r2_on_scog >= 0.5 and abs(delta) < 0.10:
        verdict = "REDUCES to /s/: perception signal is largely S_cog re-encoded."
    elif delta >= 0.10:
        verdict = f"BEATS /s/: predicts {res['target']} better than S_cog alone."
    else:
        verdict = "MIXED: overlaps S_cog but neither cleanly beats nor fully reduces."
    print(f"[money/{res['model']}/{res['target']}]  WavLM CV rho={res['obs_rho'][res['best']]:+.3f}"
          f"  S_cog-alone={res['scog_rho']:+.3f}  corr(pred,S_cog) r={r_ps:+.3f} (R2={r2_on_scog:.2f})"
          f"  -> {verdict}")
    return {'model': res['model'], 'target': res['target'], 'best_layer': res['best'],
            'best_fw_p': res['best_fw_p'],
            'wavlm_cv_rho': res['obs_rho'][res['best']], 'scog_cv_rho': res['scog_rho'],
            'pred_vs_scog_pearson': r_ps, 'pred_vs_scog_r2': r2_on_scog,
            'pred_vs_scog_spearman': rho_ps, 'delta_rho': delta,
            'collapsed': res['collapsed'], 'verdict': verdict}


def fig_layers(results, path):
    ensure_dirs(PRED_DIR)
    n_layers = len(next(iter(results.values()))['obs_rho'])
    x = np.arange(n_layers)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, tgt in zip(axes, ['perceived', 'actual']):
        for kind in MODELS:
            res = results[(kind, tgt)]
            c = '#3a6ea5' if kind == 'ridge' else '#d08a2e'
            if kind == 'ridge':                                # null band from the primary head
                lo = np.percentile(res['nulls'], 5, axis=0)
                hi = np.percentile(res['nulls'], 95, axis=0)
                ax.fill_between(x, lo, hi, color='#9a9a9a', alpha=0.15, zorder=1,
                                label='ridge null (5-95%)')
            lbl = (f"{kind} (best L{res['best']}, fw p={res['best_fw_p']:.3f}"
                   + ('  COLLAPSED' if res['collapsed'] else '') + ')')
            ax.plot(x, res['obs_rho'], LINESTYLE[kind] + 'o', color=c, lw=2, zorder=3, label=lbl)
            ax.scatter([res['best']], [res['obs_rho'][res['best']]], s=170,
                       facecolor='none', edgecolor=c, linewidth=2, zorder=4)
        # S_cog-alone reference (ridge head = the hand-feature bar)
        sc = results[('ridge', tgt)]['scog_rho']
        ax.axhline(sc, color='#555', ls=':', lw=1.3, zorder=2)
        ax.annotate(f"S_cog-alone rho={sc:.2f}", (0, sc), xytext=(2, 4),
                    textcoords='offset points', fontsize=8, color='#555')
        ax.axhline(0, color='#bbb', lw=0.8)
        ax.set_xticks(x)
        ax.set_xlabel('WavLM hidden-state layer (0 = CNN proj ... 12 = final)')
        ax.set_title(f'{tgt.upper()}  '
                     + ('(perceived_mean)' if tgt == 'perceived' else '(true Kinsey)'), fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(-1.05, 0.9)
    axes[0].set_ylabel('cross-validated Spearman  (out-of-fold pred vs truth)')
    fig.suptitle('WavLM per-layer prediction vs the hand-measured /s/ cue (n=50)\n'
                 'solid = Ridge, dashed = Elastic Net; ring = best layer; '
                 'dotted = S_cog-alone; grey = ridge permutation null', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_vs_scog(results, scog, mts, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, kind in zip(axes, MODELS):
        res = results[(kind, 'perceived')]
        mt = mts[(kind, 'perceived')]
        pred = res['best_oof']
        ax.scatter(scog, pred, s=55, color='#3a6ea5' if kind == 'ridge' else '#d08a2e',
                   edgecolor='k', linewidth=0.5, alpha=0.85)
        if np.std(pred) > 1e-9:
            b, a = np.polyfit(scog, pred, 1)
            xs = np.linspace(scog.min(), scog.max(), 50)
            ax.plot(xs, a + b * xs, color='#c1553b', lw=2)
        ax.set_xlabel('S_cog (hand-measured /s/ center of gravity)')
        ax.set_ylabel(f"{kind} best-layer (L{res['best']}) OOF pred of PERCEIVED")
        ax.set_title(f"{kind.upper()}: R2(pred~S_cog)={mt['pred_vs_scog_r2']:.2f}  |  "
                     f"CV rho={mt['wavlm_cv_rho']:.2f} vs S_cog {mt['scog_cv_rho']:.2f}\n"
                     f"{mt['verdict']}", fontsize=9)
    fig.suptitle("Does WavLM's perceived signal reduce to /s/?  (Ridge vs Elastic Net)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ensure_dirs(TABLES, PRED_DIR)
    print(f'10b_wavlm_probe.py  (LOOCV Ridge+ElasticNet on frozen WavLM | '
          f"NPERM ridge={NPERM['ridge']} enet={NPERM['enet']} | jobs={JOBS})")

    d = np.load(EMB_NPZ, allow_pickle=True)
    E, ids = d['embeddings'], list(d['file_ids'])           # (n, 13, 768)
    n_layers = E.shape[1]

    sp = pd.read_csv(SPEAKERS_CSV).set_index('file_id').reindex(ids)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv').set_index('file_id')['S_cog'].reindex(ids)

    results, mts, summary_rows, money_rows = {}, {}, [], []
    scog_perceived = None
    for tgt_name, tgt_col in TARGETS.items():
        y_full = sp[tgt_col].to_numpy(float)
        scog_full = seg.to_numpy(float)
        mask = ~np.isnan(y_full) & ~np.isnan(scog_full)     # speakers with target + S_cog
        y, scog = y_full[mask], scog_full[mask]
        Xs = [E[mask, L, :] for L in range(n_layers)]       # one (n,768) matrix per layer
        if tgt_name == 'perceived':
            scog_perceived = scog
        print(f"\n===== target: {tgt_name}  (n={mask.sum()}) =====")

        for kind in MODELS:
            res = probe_target(tgt_name, y, Xs, scog, kind, NPERM[kind])
            results[(kind, tgt_name)] = res
            mt = money_test(res, scog)
            mts[(kind, tgt_name)] = mt
            money_rows.append(mt)
            for L in range(n_layers):
                summary_rows.append({'model': kind, 'target': tgt_name, 'layer': L,
                                     'n': int(mask.sum()), 'cv_spearman': res['obs_rho'][L],
                                     'cv_r2': res['obs_r2'][L], 'cv_mae': res['obs_mae'][L],
                                     'perm_p': res['per_layer_p'][L], 'is_best': L == res['best'],
                                     'best_fw_p': res['best_fw_p']})
            summary_rows.append({'model': kind, 'target': tgt_name, 'layer': 'S_cog-alone',
                                 'n': int(mask.sum()), 'cv_spearman': res['scog_rho'],
                                 'cv_r2': res['scog_r2'], 'cv_mae': res['scog_mae'],
                                 'perm_p': np.nan, 'is_best': False, 'best_fw_p': np.nan})

    fig_layers(results, PRED_DIR / 'layers.png')
    fig_vs_scog(results, scog_perceived, mts, PRED_DIR / 'vs_scog.png')
    pd.DataFrame(summary_rows).to_csv(TABLES / 'wavlm_summary.csv', index=False)
    pd.DataFrame(money_rows).to_csv(TABLES / 'wavlm_moneytest.csv', index=False)
    print('\n  -> tables/wavlm_summary.csv, tables/wavlm_moneytest.csv')
    print('  -> figures/prediction/wavlm/layers.png, vs_scog.png')
    print('done.')


if __name__ == '__main__':
    main()
