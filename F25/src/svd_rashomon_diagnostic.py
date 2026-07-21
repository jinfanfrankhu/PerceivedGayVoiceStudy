"""svd_rashomon_diagnostic.py -- how big is the linear-probe Rashomon set, really?

Exploratory diagnostic (NOT a pipeline step, no methodological choice settled here).
Pure linear algebra on the CACHED pooled embeddings -- no WavLM forward passes, no GPU.

The pooled embedding matrix X is (n_speakers=50, 768) at the canonical layer. Because
n<<768 and centered, X has rank <= n-1 = 49: the 50 speakers live in an <=49-dim
subspace and the remaining ~719 directions carry ZERO between-speaker variance -- they
are the "free" directions a linear probe's weight can move along without changing any
prediction. Those free directions are the fuel for attribution multiplicity (10c/10d).

This script measures four things, each a direct readout of the Rashomon set's geometry:
  1. SPECTRUM / effective dimension -- singular values, variance fractions, entropy
     effective rank, participation ratio, condition number. How many directions the
     data actually pins vs leaves free.
  2. RASHOMON AXIS LENGTHS -- for a least-squares probe the epsilon-Rashomon set is an
     ellipsoid with axis length along direction i proportional to 1/sigma_i. We report
     the relative axis lengths so "which directions are free" is explicit.
  3. BOOTSTRAP STABILITY -- resample the 50 speakers, recompute the SVD, and measure
     how well the top-k subspace replicates (mean cosine of principal angles). At n=50
     only the first few PCs are expected to be stable; the tail is sampling noise.
  4. SIGNAL LOCATION (the decisive plot) -- Spearman(perceived_mean, PC score) for each
     PC. If perceived sits in the STABLE top PCs, the ridge rho~0.73 rests on identifiable
     structure (good fork). If it sits in an unstable low-variance tail PC, the prediction
     rides a direction that barely replicates and attribution there is unidentified
     (cautionary fork).

Columns are standardized before the SVD to match the probe pipeline (SimpleImputer +
StandardScaler in 10b/10c/10d), so the geometry reflects what the fitted probes see.

Outputs:
  outputs/tables/wavlm_svd_rashomon.csv     per-PC: sigma, var frac, cumulative, rel axis
                                            length, bootstrap stability, perceived Spearman
  outputs/figures/prediction/wavlm_svd_rashomon.png   scree + stability + perceived-vs-PC
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.linalg import subspace_angles

from common import PROC, SPEAKERS_CSV, TABLES, FIG, ensure_dirs

EMB_NPZ = PROC / 'wavlm_embeddings.npz'
PRED_DIR = FIG / 'prediction'
N_BOOT = int(os.environ.get('SVD_NBOOT', 500))
SEED = int(os.environ.get('SVD_SEED', 0))
K_STAB = 15                                    # report subspace stability for k = 1..K_STAB


def best_layer():
    if 'WAVLM_SAL_LAYER' in os.environ:
        return int(os.environ['WAVLM_SAL_LAYER'])
    try:
        mt = pd.read_csv(TABLES / 'wavlm_moneytest.csv')
        row = mt[(mt['model'] == 'ridge') & (mt['target'] == 'perceived')]
        return int(row['best_layer'].iloc[0])
    except Exception:
        return 6


def standardize(X):
    """Median-impute + column standardize, matching the 10b/10c/10d probe pipeline."""
    X = X.copy()
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(med, idx[1])
    mu = X.mean(0)
    sd = X.std(0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def spectrum(Xz):
    """SVD of standardized X -> singular values and derived effective-dimension stats."""
    s = np.linalg.svd(Xz, compute_uv=False)
    s = s[s > 1e-9 * s[0]]                      # drop numerical-zero (null-space) directions
    var = s ** 2
    frac = var / var.sum()
    stats = {
        'rank': len(s),
        'eff_rank_entropy': float(np.exp(-(frac * np.log(frac)).sum())),
        'participation_ratio': float(var.sum() ** 2 / (var ** 2).sum()),
        'condition_number': float(s[0] / s[-1]),
        'n_pc_90': int(np.searchsorted(np.cumsum(frac), 0.90) + 1),
        'n_pc_95': int(np.searchsorted(np.cumsum(frac), 0.95) + 1),
    }
    return s, frac, stats


def bootstrap_stability(Xz_full, n_boot, kmax, rng):
    """Mean cosine of principal angles between full and bootstrap top-k subspaces.

    1.0 = the top-k subspace replicates perfectly; ->0 = the directions are sampling
    noise. Computed on the standardized full matrix; each bootstrap resamples speakers
    with replacement and re-standardizes (what refitting a probe would actually see)."""
    n = Xz_full.shape[0]
    _, _, Vt_full = np.linalg.svd(Xz_full, full_matrices=False)
    ks = np.arange(1, kmax + 1)
    acc = np.zeros((n_boot, kmax))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb = standardize(Xz_full[idx])          # re-standardize the resample
        _, _, Vt_b = np.linalg.svd(Xb, full_matrices=False)
        for j, k in enumerate(ks):
            ang = subspace_angles(Vt_full[:k].T, Vt_b[:k].T)
            acc[b, j] = np.cos(ang).mean()      # mean alignment across the k angles
    return ks, acc.mean(0), acc.std(0)


def main():
    ensure_dirs(TABLES, PRED_DIR)
    layer = best_layer()
    d = np.load(EMB_NPZ, allow_pickle=True)
    E, ids = d['embeddings'], list(d['file_ids'])
    sp = pd.read_csv(SPEAKERS_CSV).set_index('file_id').reindex(ids)
    y = sp['perceived_mean'].to_numpy(float)
    X = E[:, layer, :].astype(float)            # (50, 768) at canonical layer
    print(f'svd_rashomon_diagnostic  (layer L{layer}; X = {X.shape[0]} speakers x {X.shape[1]} dims)')

    Xz = standardize(X)
    s, frac, stats = spectrum(Xz)
    cum = np.cumsum(frac)
    rel_axis = s[0] / s                          # Rashomon ellipsoid axis length ~ 1/sigma
    scores = Xz @ np.linalg.svd(Xz, full_matrices=False)[2].T   # PC scores (50 x rank)

    print(f"\n  nominal dims 768  ->  rank {stats['rank']}  (n-1 ceiling; the other "
          f"{768 - stats['rank']} directions are EXACT null space = fully free)")
    print(f"  effective rank (entropy) = {stats['eff_rank_entropy']:.1f}   "
          f"participation ratio = {stats['participation_ratio']:.1f}")
    print(f"  {stats['n_pc_90']} PCs for 90% var, {stats['n_pc_95']} for 95%   "
          f"condition number = {stats['condition_number']:.1f}")

    # bootstrap subspace stability
    rng = np.random.default_rng(SEED)
    ks, stab_mean, stab_sd = bootstrap_stability(Xz, N_BOOT, K_STAB, rng)
    stab_full = np.full(len(s), np.nan)
    stab_full[:K_STAB] = stab_mean
    print(f"\n  bootstrap top-k subspace stability (mean cos principal angle, {N_BOOT} resamples):")
    for k in (1, 2, 3, 5, 8, 10, 15):
        if k <= K_STAB:
            print(f"    top-{k:2d}: {stab_mean[k-1]:.3f}")

    # signal location: perceived vs each PC
    rp = np.array([spearmanr(scores[:, i], y)[0] for i in range(scores.shape[1])])
    pp = np.array([spearmanr(scores[:, i], y)[1] for i in range(scores.shape[1])])
    order = np.argsort(-np.abs(rp))
    print(f"\n  perceived_mean vs PC score (Spearman) -- top |rho| PCs:")
    for i in order[:6]:
        flag = 'STABLE' if (i < K_STAB and stab_mean[i] > 0.7) else 'unstable/tail'
        print(f"    PC{i+1:2d}: rho={rp[i]:+.3f} p={pp[i]:.3f}  var={100*frac[i]:4.1f}%  [{flag}]")

    # table
    tab = pd.DataFrame({
        'pc': np.arange(1, len(s) + 1), 'sigma': s, 'var_frac': frac, 'cum_var': cum,
        'rel_axis_len': rel_axis, 'boot_stability': stab_full,
        'perceived_spearman': rp[:len(s)], 'perceived_p': pp[:len(s)],
    })
    tab.to_csv(TABLES / 'wavlm_svd_rashomon.csv', index=False)

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    a = ax[0]
    a.bar(np.arange(1, min(30, len(s)) + 1), frac[:30] * 100, color='#3a6ea5')
    a2 = a.twinx()
    a2.plot(np.arange(1, min(30, len(s)) + 1), cum[:30] * 100, 'o-', color='#d08a2e', ms=3)
    a2.axhline(90, color='#aaa', lw=0.7, ls='--')
    a.set_xlabel('PC'); a.set_ylabel('% variance'); a2.set_ylabel('cumulative %', color='#d08a2e')
    a.set_title(f'Scree (L{layer}): effective rank {stats["eff_rank_entropy"]:.1f} of {stats["rank"]}')

    a = ax[1]
    a.errorbar(ks, stab_mean, yerr=stab_sd, marker='o', color='#2ca25f', ms=4, capsize=2)
    a.axhline(0.7, color='#aaa', lw=0.7, ls='--')
    a.set_ylim(0, 1.02); a.set_xlabel('top-k'); a.set_ylabel('subspace stability (cos angle)')
    a.set_title(f'Bootstrap subspace stability ({N_BOOT} resamples)')

    a = ax[2]
    npc = min(30, len(s))
    cols = ['#c0392b' if (i < K_STAB and stab_mean[i] > 0.7) else '#888'
            for i in range(npc)]
    a.bar(np.arange(1, npc + 1), rp[:npc], color=cols)
    a.axhline(0, color='#aaa', lw=0.7)
    a.set_xlabel('PC'); a.set_ylabel('Spearman(perceived, PC score)')
    a.set_title('Where the perceived signal lives (red = stable PC)')
    fig.suptitle('Linear-probe Rashomon geometry from the SVD of pooled WavLM embeddings (n=50)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PRED_DIR / 'wavlm_svd_rashomon.png', dpi=120)
    plt.close(fig)
    print('\n  -> tables/wavlm_svd_rashomon.csv, figures/prediction/wavlm_svd_rashomon.png')
    print('done.')


if __name__ == '__main__':
    main()
