"""10b_wavlm_probe.py -- honest prediction from frozen WavLM embeddings.

Consumes the cache from 10_wavlm_extract.py and runs it through the SAME honest
harness as 07_ridge.py (LOOCV out-of-fold Ridge + permutation null + cross-
validated Spearman, both targets). The flagship question:

    Does a SOTA speech representation BEAT your hand-measured /s/ cue,
    or does it just REDUCE TO it?

Two rigor upgrades over 07, both because WavLM gives us 13 layers:

  * WHICH layer predicts best is a hyperparameter -> picking it post-hoc is
    double-dipping. We handle it with a MAX-STATISTIC permutation null: each
    permutation shuffles the label ONCE and records all 13 layers' null scores
    together; the null of the *max across layers* is the family-wise null for
    "the best of 13 layers beats chance." Rigorous and cheap (one perm pass).
    Per-layer (uncorrected) p-values are also reported for the profile.

  * S_cog-alone is run through the identical harness as a reference bar, so
    "beats vs reduces" is apples-to-apples.

The money test: take the best layer's out-of-fold PERCEIVED predictions and
regress them on S_cog. High R^2 => the black box rediscovered the sibilant
(reduces). WavLM CV-rho >> S_cog-alone with residual signal => it found more
(beats).

Env knobs:  WAVLM_NPERM (1000)   WAVLM_JOBS (all cores)

Outputs:
  outputs/tables/wavlm_summary.csv      one row per (target x layer) + refs
  outputs/tables/wavlm_moneytest.csv    best-layer vs S_cog head-to-head
  outputs/figures/prediction/wavlm_layers.png    per-layer CV-rho profile + null
  outputs/figures/prediction/wavlm_vs_scog.png   best-layer OOF pred vs S_cog
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed

from common import SPEAKERS_CSV, PROC, FIG, TABLES, ensure_dirs

PRED_DIR = FIG / 'prediction'
EMB_NPZ = PROC / 'wavlm_embeddings.npz'
KINSEY = 'Kinsey Scale (1-5)'
TARGETS = {'perceived': 'perceived_mean', 'actual': KINSEY}

N_PERM = int(os.environ.get('WAVLM_NPERM', 1000))
JOBS = int(os.environ.get('WAVLM_JOBS', os.cpu_count() or 1))
SEED = 0
# grid widened vs 07 (p=768 >> n needs heavier shrinkage); inner-LOO still picks per fold
ALPHAS = np.logspace(-1, 5, 16)


def make_model():
    """The single a-priori pipeline, refit from scratch inside every LOOCV fold."""
    return make_pipeline(SimpleImputer(strategy='median'),
                         StandardScaler(),
                         RidgeCV(alphas=ALPHAS))       # inner-LOO alpha, no held-out leak


def loocv_oof(X, y):
    """Out-of-fold predictions: train on the other n-1 speakers, predict each held-out one."""
    n = len(y)
    oof = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        tr = idx != i
        oof[i] = make_model().fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    return oof


def cv_score(X, y):
    """Cross-validated Spearman / R^2 / MAE between out-of-fold predictions and truth."""
    oof = loocv_oof(X, y)
    rho = spearmanr(oof, y).statistic
    r2 = 1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()   # OOF R^2 (can be < 0)
    mae = np.abs(y - oof).mean()
    return rho, r2, mae, oof


def _null_chunk(seeds, Xs, y):
    """One worker: for each seed, shuffle y ONCE and score all layers on that shuffle."""
    out = np.empty((len(seeds), len(Xs)))
    for j, s in enumerate(seeds):
        yp = np.random.default_rng(s).permutation(y)
        for L, XL in enumerate(Xs):
            out[j, L] = spearmanr(loocv_oof(XL, yp), yp).statistic
    return out


def layer_permutation_null(Xs, y):
    """(N_PERM, n_layers) null Spearman matrix; same shuffle shared across layers per row."""
    chunks = [c for c in np.array_split(np.arange(SEED, SEED + N_PERM), JOBS) if len(c)]
    res = Parallel(n_jobs=JOBS)(delayed(_null_chunk)(c, Xs, y) for c in chunks)
    return np.vstack(res)


def probe_target(tgt_name, y, Xs, scog):
    """Full per-layer probe for one target + max-stat correction + S_cog-alone reference."""
    n_layers = len(Xs)
    # observed per-layer scores
    obs = np.array([cv_score(XL, y)[:3] for XL in Xs])          # (n_layers, 3): rho, r2, mae
    obs_rho = obs[:, 0]
    best = int(np.argmax(obs_rho))
    _, _, _, best_oof = cv_score(Xs[best], y)                   # keep best layer's OOF preds

    # S_cog-alone through the identical harness (reference bar)
    scog_rho, scog_r2, scog_mae, scog_oof = cv_score(scog.reshape(-1, 1), y)

    # max-statistic permutation null (family-wise across the 13 layers)
    nulls = layer_permutation_null(Xs, y)                       # (N_PERM, n_layers)
    null_max = nulls.max(axis=1)
    per_layer_p = [(1 + int((nulls[:, L] >= obs_rho[L]).sum())) / (N_PERM + 1)
                   for L in range(n_layers)]
    best_fw_p = (1 + int((null_max >= obs_rho[best]).sum())) / (N_PERM + 1)

    print(f"\n[{tgt_name}] best layer = {best} (CV rho={obs_rho[best]:+.3f}, "
          f"family-wise p={best_fw_p:.3f})   S_cog-alone CV rho={scog_rho:+.3f}")
    for L in range(n_layers):
        star = ' *' if per_layer_p[L] < 0.05 else '  '
        mark = ' <-- best' if L == best else ''
        print(f"   layer {L:2d}  CV rho={obs_rho[L]:+.3f}  R2={obs[L,1]:+.2f}  "
              f"p={per_layer_p[L]:.3f}{star}{mark}")

    return {
        'target': tgt_name, 'obs_rho': obs_rho, 'obs_r2': obs[:, 1], 'obs_mae': obs[:, 2],
        'per_layer_p': per_layer_p, 'nulls': nulls, 'null_max': null_max,
        'best': best, 'best_fw_p': best_fw_p, 'best_oof': best_oof,
        'scog_rho': scog_rho, 'scog_r2': scog_r2, 'scog_mae': scog_mae, 'scog_oof': scog_oof,
    }


def money_test(res, scog):
    """Does the best layer's perceived signal reduce to S_cog? (perceived target only)"""
    pred = res['best_oof']
    r_ps = pearsonr(pred, scog)
    rho_ps = spearmanr(pred, scog).statistic
    r2_on_scog = r_ps.statistic ** 2                           # var(WavLM pred) explained by S_cog
    print(f"\n[money test]  best-layer WavLM CV rho = {res['obs_rho'][res['best']]:+.3f}")
    print(f"              S_cog-alone   CV rho = {res['scog_rho']:+.3f}")
    print(f"              corr(WavLM OOF pred, S_cog): Pearson r={r_ps.statistic:+.3f} "
          f"(R2={r2_on_scog:.2f}), Spearman rho={rho_ps:+.3f}")
    delta = res['obs_rho'][res['best']] - res['scog_rho']
    if r2_on_scog >= 0.5 and abs(delta) < 0.10:
        verdict = "REDUCES to /s/: WavLM's perception signal is largely S_cog re-encoded."
    elif delta >= 0.10:
        verdict = "BEATS /s/: WavLM predicts perceived better than S_cog alone."
    else:
        verdict = "MIXED: overlaps S_cog but neither cleanly beats nor fully reduces."
    print(f"              -> {verdict}")
    return {'wavlm_cv_rho': res['obs_rho'][res['best']], 'scog_cv_rho': res['scog_rho'],
            'pred_vs_scog_pearson': r_ps.statistic, 'pred_vs_scog_r2': r2_on_scog,
            'pred_vs_scog_spearman': rho_ps, 'delta_rho': delta, 'verdict': verdict}


def fig_layers(results, path):
    ensure_dirs(PRED_DIR)
    n_layers = len(results['perceived']['obs_rho'])
    x = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {'perceived': '#3a6ea5', 'actual': '#c1553b'}
    for tgt, res in results.items():
        c = colors[tgt]
        # null band: 5-95th pct of the per-layer null
        lo = np.percentile(res['nulls'], 5, axis=0)
        hi = np.percentile(res['nulls'], 95, axis=0)
        ax.fill_between(x, lo, hi, color=c, alpha=0.10, zorder=1)
        ax.plot(x, res['obs_rho'], '-o', color=c, lw=2, zorder=3,
                label=f"{tgt} (best L{res['best']}, fw p={res['best_fw_p']:.3f})")
        # S_cog-alone reference line
        ax.axhline(res['scog_rho'], color=c, ls='--', lw=1.2, alpha=0.7, zorder=2)
        ax.annotate(f"S_cog-alone ({tgt}) rho={res['scog_rho']:.2f}",
                    (n_layers - 1, res['scog_rho']), xytext=(-4, 4),
                    textcoords='offset points', ha='right', fontsize=8, color=c)
        ax.scatter([res['best']], [res['obs_rho'][res['best']]], s=180,
                   facecolor='none', edgecolor=c, linewidth=2, zorder=4)
    ax.axhline(0, color='#bbb', lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel('WavLM hidden-state layer (0 = CNN projection ... 12 = final)')
    ax.set_ylabel('cross-validated Spearman  (out-of-fold pred vs truth)')
    ax.set_title('WavLM per-layer prediction vs the hand-measured /s/ cue (n=50)\n'
                 'solid = WavLM by layer; dashed = S_cog-alone; ring = best layer; '
                 'band = 5-95% permutation null', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_vs_scog(res, scog, mt, path):
    pred = res['best_oof']
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scog, pred, s=60, color='#3a6ea5', edgecolor='k', linewidth=0.5, alpha=0.85)
    b, a = np.polyfit(scog, pred, 1)
    xs = np.linspace(scog.min(), scog.max(), 50)
    ax.plot(xs, a + b * xs, color='#c1553b', lw=2)
    ax.set_xlabel('S_cog  (hand-measured /s/ center of gravity)')
    ax.set_ylabel(f"WavLM best-layer (L{res['best']}) OOF prediction of PERCEIVED")
    ax.set_title("Does WavLM's perception signal reduce to /s/?\n"
                 f"R2(pred ~ S_cog) = {mt['pred_vs_scog_r2']:.2f}  |  "
                 f"WavLM CV rho={mt['wavlm_cv_rho']:.2f} vs S_cog-alone {mt['scog_cv_rho']:.2f}\n"
                 f"{mt['verdict']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ensure_dirs(TABLES, PRED_DIR)
    print(f'10b_wavlm_probe.py  (LOOCV Ridge on frozen WavLM | N_PERM={N_PERM} | jobs={JOBS})')

    d = np.load(EMB_NPZ, allow_pickle=True)
    E, ids = d['embeddings'], list(d['file_ids'])           # (n, 13, 768)
    n_layers = E.shape[1]

    sp = pd.read_csv(SPEAKERS_CSV).set_index('file_id').reindex(ids)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv').set_index('file_id')['S_cog'].reindex(ids)

    results, money_rows, summary_rows = {}, [], []
    for tgt_name, tgt_col in TARGETS.items():
        y_full = sp[tgt_col].to_numpy(float)
        scog_full = seg.to_numpy(float)
        mask = ~np.isnan(y_full) & ~np.isnan(scog_full)     # keep speakers with target + S_cog
        y = y_full[mask]
        scog = scog_full[mask]
        Xs = [E[mask, L, :] for L in range(n_layers)]       # one (n,768) matrix per layer
        print(f"\n===== target: {tgt_name}  (n={mask.sum()}) =====")

        res = probe_target(tgt_name, y, Xs, scog)
        results[tgt_name] = res
        for L in range(n_layers):
            summary_rows.append({'target': tgt_name, 'layer': L, 'n': int(mask.sum()),
                                 'cv_spearman': res['obs_rho'][L], 'cv_r2': res['obs_r2'][L],
                                 'cv_mae': res['obs_mae'][L], 'perm_p': res['per_layer_p'][L],
                                 'is_best': L == res['best'], 'best_fw_p': res['best_fw_p']})
        summary_rows.append({'target': tgt_name, 'layer': 'S_cog-alone', 'n': int(mask.sum()),
                             'cv_spearman': res['scog_rho'], 'cv_r2': res['scog_r2'],
                             'cv_mae': res['scog_mae'], 'perm_p': np.nan,
                             'is_best': False, 'best_fw_p': np.nan})

        mt = money_test(res, scog)                          # meaningful for both; headline = perceived
        mt = {'target': tgt_name, 'best_layer': res['best'], **mt}
        money_rows.append(mt)
        if tgt_name == 'perceived':
            fig_vs_scog(res, scog, mt, PRED_DIR / 'wavlm_vs_scog.png')

    fig_layers(results, PRED_DIR / 'wavlm_layers.png')
    pd.DataFrame(summary_rows).to_csv(TABLES / 'wavlm_summary.csv', index=False)
    pd.DataFrame(money_rows).to_csv(TABLES / 'wavlm_moneytest.csv', index=False)
    print('\n  -> tables/wavlm_summary.csv, tables/wavlm_moneytest.csv')
    print('  -> figures/prediction/wavlm_layers.png, wavlm_vs_scog.png')
    print('done.')


if __name__ == '__main__':
    main()
