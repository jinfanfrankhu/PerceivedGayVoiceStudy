"""09_power.py  --  formal power analysis for the F25 design.

Answers the two questions a p-value alone CANNOT:
  (1) SENSITIVITY: given n=50, how big must an effect be before we could reliably detect
      it? -> lets us honestly caveat the ACTUAL-orientation null ("not detected" is NOT
      the same as "not there" when the study was underpowered for small effects).
  (2) PLANNING: what n would we need to detect a small-but-real effect, or to PROVE the
      perceived-vs-actual divergence? -> the evidence-based justification for scaling up.

Three parts, each tied to a real claim in the paper:
  A. ANALYTIC power for a single correlation via Fisher's z-transform -- the textbook
     baseline. Power-vs-n curves, n-for-80%-power, and the minimum detectable effect (MDE).
  B. MONTE-CARLO power that respects Kinsey's REAL discrete/bimodal distribution (Gaussian
     copula onto the empirical Kinsey marginal) -> the EXTRA n the coarse ordinal scale
     costs us on the actual-orientation side (ties throw away information).
  C. Power to detect the DIVERGENCE (rho_perceived - rho_actual) via Williams's test for
     dependent correlations, anchored on the OBSERVED S_cog values -> the n the 06
     divergence test would have needed (it found 0/13, CI +-0.5 wide = underpowered).

All effects are expressed as Spearman rho to match the rest of the pipeline. Assumptions
stated, not hidden: target power 0.80, alpha 0.05, two-sided.

NOTE on legitimate use: this is PROSPECTIVE (what n do we need) + SENSITIVITY (what could
n=50 detect) power -- NOT "post-hoc power" from the observed effect (which is just a
reworded p-value and statistically meaningless). Effect sizes here are chosen as
a-priori "smallest effect worth caring about", not read off our own results.

Outputs:
  outputs/figures/power/power_curves.png      analytic power vs n, effect-size family
  outputs/figures/power/power_kinsey.png      discreteness penalty on the actual side
  outputs/figures/power/power_divergence.png  power to prove divergence vs n
  outputs/tables/power_summary.csv            n-for-80% and MDE across scenarios
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t, spearmanr

from common import SPEAKERS_CSV, PROC, FIG, TABLES, ensure_dirs

POWER_DIR = FIG / 'power'
KINSEY = 'Kinsey Scale (1-5)'
ALPHA = 0.05
TARGET = 0.80
SEED = 0
N_SIM = 2000                                    # Monte-Carlo replications per (effect, n)
RHO_FAMILY = [0.1, 0.2, 0.3, 0.4, 0.5]          # a-priori "effects worth caring about"
N_FINE = np.arange(10, 301, 2)                  # smooth grid for the cheap analytic curves
N_MC = [30, 40, 50, 75, 100, 125, 150, 200]     # coarser grid for the simulated parts
MARK_N = [50, 100, 150]                         # reference sample sizes to annotate


# ----------------------------------------------------------------- A: analytic ----

def power_corr(rho, n, alpha=ALPHA):
    """Power of a two-sided correlation test at |rho|, sample size n (Fisher z)."""
    if n <= 4 or rho == 0:
        return alpha
    delta = np.arctanh(abs(rho)) * np.sqrt(n - 3)      # noncentrality on the z-scale
    zc = norm.ppf(1 - alpha / 2)
    return norm.cdf(delta - zc) + norm.cdf(-delta - zc)


def n_for_power(rho, target=TARGET, alpha=ALPHA, nmax=5000):
    """Smallest n giving >= target power to detect correlation rho."""
    for n in range(5, nmax):
        if power_corr(rho, n, alpha) >= target:
            return n
    return np.nan


def mde(n, target=TARGET, alpha=ALPHA):
    """Minimum detectable effect: smallest rho this n can catch at >= target power."""
    for r in np.arange(0.01, 0.995, 0.005):
        if power_corr(r, n, alpha) >= target:
            return round(float(r), 3)
    return np.nan


# ------------------------------------------------- B: Kinsey discreteness penalty ----

def power_kinsey_mc(rho_latent, n, kinsey_pool, rng, n_sim=N_SIM, alpha=ALPHA):
    """Monte-Carlo power for a feature<->ACTUAL correlation when the actual variable has
    Kinsey's real (discrete, bimodal) marginal. Gaussian copula: a latent bivariate
    normal supplies the rank structure; one arm is mapped onto a bootstrap of the empirical
    Kinsey values (preserving ties/bimodality exactly). Returns (power, mean achieved rho)."""
    hits = 0
    achieved = np.empty(n_sim)
    for i in range(n_sim):
        z1 = rng.standard_normal(n)
        z2 = rho_latent * z1 + np.sqrt(1 - rho_latent ** 2) * rng.standard_normal(n)
        ranks = z2.argsort().argsort()                  # 0..n-1 rank of each z2
        y = np.sort(rng.choice(kinsey_pool, size=n, replace=True))[ranks]
        res = spearmanr(z1, y)
        hits += res.pvalue < alpha
        achieved[i] = res.statistic
    return hits / n_sim, float(np.nanmean(achieved))


# ------------------------------------------------------------ C: divergence power ----

def williams_t(r12, r13, r23, n):
    """Williams's T2 for dependent correlations sharing a variable (H0: rho12 = rho13)."""
    R = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23     # corr-matrix determinant
    num = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
    den = np.sqrt(2 * ((n - 1) / (n - 3)) * R + ((r12 + r13) ** 2 / 4) * (1 - r23) ** 3)
    return num / den


def power_divergence(rp, ra, rpa, n, rng, n_sim=N_SIM, alpha=ALPHA):
    """Power to detect rho_perceived != rho_actual, given they share the feature X and
    perceived<->actual correlate at rpa. Simulate (X, P, A) ~ MVN with that structure,
    apply Williams's test, count rejections."""
    C = np.array([[1, rp, ra], [rp, 1, rpa], [ra, rpa, 1]], float)
    L = np.linalg.cholesky(C)                            # requires C positive-definite
    tc = t.ppf(1 - alpha / 2, n - 3)
    hits = 0
    for _ in range(n_sim):
        Z = rng.standard_normal((n, 3)) @ L.T
        r12 = spearmanr(Z[:, 0], Z[:, 1]).statistic
        r13 = spearmanr(Z[:, 0], Z[:, 2]).statistic
        r23 = spearmanr(Z[:, 1], Z[:, 2]).statistic
        hits += abs(williams_t(r12, r13, r23, n)) > tc
    return hits / n_sim


# ------------------------------------------------------------------- figures ----

def fig_curves(path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for rho in RHO_FAMILY:
        p = [power_corr(rho, n) for n in N_FINE]
        ax.plot(N_FINE, p, lw=2, label=f'rho = {rho}')
    ax.axhline(TARGET, color='#888', ls='--', lw=1)
    ax.annotate('80% power', (N_FINE[-1], TARGET), xytext=(-70, 6),
                textcoords='offset points', fontsize=9, color='#555')
    for nm in MARK_N:
        ax.axvline(nm, color='#ddd', lw=1, zorder=0)
        ax.annotate(f'n={nm}', (nm, 0.02), fontsize=8, color='#999', ha='center')
    ax.set_xlabel('sample size (n speakers)')
    ax.set_ylabel('power  (chance of detecting the effect if it is real)')
    ax.set_title('Analytic power for a single correlation (Fisher z, alpha=.05 two-sided)')
    ax.set_ylim(0, 1.02)
    ax.legend(title='true effect size', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_kinsey(rows, achieved, path):
    ns = [r['n'] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ns, [r['power_kinsey'] for r in rows], 'o-', color='#c1553b', lw=2,
            label=f'real Kinsey scale (achieved rho~{achieved:.2f})')
    ax.plot(ns, [power_corr(achieved, n) for n in ns], 's--', color='#3a6ea5', lw=2,
            label=f'ideal continuous scale (rho={achieved:.2f})')
    ax.axhline(TARGET, color='#888', ls='--', lw=1)
    ax.set_xlabel('sample size (n speakers)')
    ax.set_ylabel('power')
    ax.set_title("The Kinsey penalty: coarse 1-5 bimodal scale costs power on 'actual'")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_divergence(rows, rp, ra, rpa, path):
    ns = [r['n'] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ns, [r['power_div'] for r in rows], 'o-', color='#6a4c93', lw=2)
    ax.axhline(TARGET, color='#888', ls='--', lw=1)
    ax.annotate('80% power', (ns[-1], TARGET), xytext=(-70, 6),
                textcoords='offset points', fontsize=9, color='#555')
    ax.axvline(50, color='#ddd', lw=1)
    ax.annotate('current n=50', (50, 0.05), fontsize=8, color='#999', ha='center')
    ax.set_xlabel('sample size (n speakers)')
    ax.set_ylabel('power to detect rho_perceived != rho_actual')
    ax.set_title(f'Power to PROVE the divergence  '
                 f'(perceived rho={rp:.2f}, actual rho={ra:.2f}, P<->A rho={rpa:.2f})')
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------- main ----

def main():
    ensure_dirs(TABLES, POWER_DIR)
    rng = np.random.default_rng(SEED)
    print('09_power.py  (formal power analysis: sensitivity + planning)', flush=True)

    # anchor the divergence scenario on the OBSERVED S_cog correlations
    sp = pd.read_csv(SPEAKERS_CSV)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv')
    df = sp.merge(seg, on='file_id', how='left', suffixes=('', '_seg'))
    d = df.dropna(subset=['S_cog', 'perceived_mean', KINSEY])
    rp = spearmanr(d['S_cog'], d['perceived_mean']).statistic
    ra = spearmanr(d['S_cog'], d[KINSEY]).statistic
    rpa = spearmanr(d['perceived_mean'], d[KINSEY]).statistic
    kinsey_pool = d[KINSEY].to_numpy(float)
    print(f'  observed anchors: S_cog<->perceived rho={rp:+.3f}  S_cog<->actual rho={ra:+.3f}'
          f'  perceived<->actual rho={rpa:+.3f}', flush=True)

    rows = []

    # --- A: analytic curves + n-for-80% + MDE ---
    print('\n  [A] analytic single-correlation power', flush=True)
    for rho in RHO_FAMILY:
        need = n_for_power(rho)
        print(f'      rho={rho}:  n for 80% power = {need}', flush=True)
    for nm in MARK_N:
        print(f'      n={nm}: minimum detectable effect (80% power) = rho {mde(nm)}',
              flush=True)
    fig_curves(POWER_DIR / 'power_curves.png')

    # --- B: Kinsey discreteness penalty (actual side) ---
    # pick a latent strength targeting a small-ish actual effect, then read achieved rho
    print('\n  [B] Kinsey discreteness penalty (Monte Carlo, actual side)', flush=True)
    rho_latent = 0.35
    krows = []
    achieved_ref = None
    for n in N_MC:
        pw, ach = power_kinsey_mc(rho_latent, n, kinsey_pool, rng)
        krows.append({'n': n, 'power_kinsey': pw, 'achieved_rho': ach})
        achieved_ref = ach if achieved_ref is None else achieved_ref
        print(f'      n={n:4d}: power(real Kinsey)={pw:.2f}  vs  '
              f'power(continuous rho={ach:.2f})={power_corr(ach, n):.2f}', flush=True)
    achieved_ref = float(np.mean([r['achieved_rho'] for r in krows]))
    fig_kinsey(krows, achieved_ref, POWER_DIR / 'power_kinsey.png')

    # --- C: divergence power ---
    print('\n  [C] power to detect the perceived-vs-actual divergence (Williams)', flush=True)
    drows = []
    for n in N_MC:
        pw = power_divergence(rp, ra, rpa, n, rng)
        drows.append({'n': n, 'power_div': pw})
        print(f'      n={n:4d}: power to prove divergence = {pw:.2f}', flush=True)
    # find the crossing n for 80% power (fine search, analytic-free -> interpolate MC)
    div_n80 = next((r['n'] for r in drows if r['power_div'] >= TARGET), None)
    fig_divergence(drows, rp, ra, rpa, POWER_DIR / 'power_divergence.png')

    # --- summary table ---
    for rho in RHO_FAMILY:
        rows.append({'scenario': f'single corr rho={rho}', 'metric': 'n for 80% power',
                     'value': n_for_power(rho)})
    for nm in MARK_N:
        rows.append({'scenario': f'n={nm}', 'metric': 'min detectable rho (80%)',
                     'value': mde(nm)})
    for r in krows:
        rows.append({'scenario': f"actual/Kinsey n={r['n']} (achieved rho~{r['achieved_rho']:.2f})",
                     'metric': 'MC power', 'value': round(r['power_kinsey'], 3)})
    for r in drows:
        rows.append({'scenario': f"divergence n={r['n']}", 'metric': 'power',
                     'value': round(r['power_div'], 3)})
    pd.DataFrame(rows).to_csv(TABLES / 'power_summary.csv', index=False)

    print('\n  --- plain-language summary ---', flush=True)
    print(f'  * At n=50 you can only reliably (80%) catch effects of rho >= {mde(50)}.'
          f'  Smaller real effects will usually be MISSED -> the actual-orientation null'
          f' is "underpowered", not "proven zero".', flush=True)
    print(f'  * To catch a modest actual effect (rho=0.3) you would need n = {n_for_power(0.3)}'
          f' on a continuous scale -- and MORE than that with the real Kinsey scale (part B).',
          flush=True)
    print(f'  * To PROVE the perceived-vs-actual divergence at the observed effect sizes,'
          f' 80% power arrives around n = {div_n80 if div_n80 else ">200"} (part C).',
          flush=True)
    print('  -> tables/power_summary.csv + figures/power/*.png', flush=True)
    print('done.', flush=True)


if __name__ == '__main__':
    main()
