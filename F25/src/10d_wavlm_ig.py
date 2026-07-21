"""10d_wavlm_ig.py -- integrated gradients: where in the SIGNAL does WavLM read "gay voice"?

10c attributed the perceived prediction to frames using the linear HEAD's weights on
the frozen embedding (s_t = w_eff . h_t). That is exact for the head, but it is NOT
head-invariant: ridge (dense, 768 dims) and enet (sparse, 30 dims) predict equally well
yet decompose to CONTRADICTORY per-phone stories (ridge: vowels+, sibilants-; enet the
reverse; frame agreement r=0.45). The disagreement lives in WEIGHT space, because the
768 embedding dims are collinear -- many weight vectors predict the pooled mean equally.

This script asks the question one level deeper, in INPUT (time) space. Integrated
gradients backprops the prediction through the ENTIRE frozen WavLM to the input waveform:

    IG_i = (x_i - x0_i) * mean_alpha  d f(x0 + alpha (x - x0)) / d x_i ,   alpha in [0,1]

with baseline x0 and f(x) = w_eff . mean_t WavLM(x).hidden[L]. Averaging the gradient along
the x0->clip path (not just at the clip) defeats saturation; completeness guarantees
sum_i IG_i = f(x) - f(x0). PRIMARY baseline is matched-spectrum NOISE (E11): the silence
baseline was found to be ill-posed here -- IG completeness is unsatisfiable at any practical
grid because WavLM's response to the silence->signal onset is near-singular (see E8/E10).
Noise makes g(alpha) near-linear, completeness passes, and the heads agree better. Silence
(IG_BASELINE=silence) is retained only to reproduce that failure as a reported result.

Why this might BREAK THE TIE 10c hit: the collinear dims ridge/enet weight differently
may trace back through WavLM to the SAME moments in the audio. So the heads can disagree
on dimensions yet AGREE on time -- which is exactly the "where does it listen" claim we
want. If they still disagree in time, the under-determination is deep (a stronger n=50
argument). Either way is a real result; the ridge-vs-enet cross-check is the guardrail.

BUT IG DOES NOT ESCAPE THE PROBE ARBITRARINESS -- do not read agreement here as proof that
it does. By IG's own Linearity axiom, for f = sum_j w_j * pooled_j:

    IG_i(f; x, x0)  =  sum_j  w_j * IG_i(pooled_j; x, x0)

i.e. the attribution map is a LINEAR FUNCTION OF w. Perturb w by delta and the map moves by
sum_j delta_j IG_i(pooled_j), which vanishes only if delta lies in the null space of the
path-averaged encoder Jacobian -- a different object from the data-covariance null space that
made delta unidentifiable in the first place. Ridge and enet are two points in one Rashomon
set; a third equally-good probe can give a third story, and nothing here forbids it.
(Implementation Invariance does not help: it covers re-parameterising ONE function, not
choosing among many that merely fit equally well.) So any empirical ridge/enet agreement
below is an OBSERVATION, not a guarantee -- see Bilodeau et al., PNAS 121(2) 2024, which
proves complete+linear attribution methods (IG and SHAP by name) can fail to beat random
guessing for inferring model behaviour. The defensible version of this analysis samples the
probe's equivalence class and reports the RANGE of attributions across it.

COMPLETENESS IS CHECKED, NOT ASSUMED. The path integral is approximated by a quadrature over
IG_STEPS points; completeness holds iff sum_i IG_i == f(x) - f(x0) to within numerical slop,
so we compute both endpoints (2 extra forwards/speaker, ~3% overhead) and report the relative
error per head. The ORIGINAL *linearly* spaced midpoint rule was run and FAILED badly (rel.err
78-213% at 20-50 steps): with the silence baseline ~100%+ of f(x)-f(x0) accrues in alpha in
[0, 0.01] (WavLM reacts violently to the silence->signal onset, then saturates), so uniform
alpha lands ~1 point in the spike. Extrapolating the convergence rate, uniform spacing would
need ~7000 steps -- impractical. The fix (E10) is a change-of-variables 'power' grid
(alpha=u^p) that clusters points near 0 while staying an unbiased quadrature of the same
integral. rel.err <= COMP_TOL => the grid was enough; above it => raise IG_STEPS / IG_ALPHA_POWER,
and treat the per-phone attributions as provisional until it converges.

Cost: backprop through WavLM on CPU is heavy, so we use one ~IG_SECONDS window/speaker
and IG_STEPS path points. ~2.8 h for 50 speakers at 128 steps on this CPU (run by day, or GPU).

Env knobs: IG_SECONDS (10)  IG_STEPS (128)  IG_NSPK (all)  WAVLM_SAL_LAYER (ridge best)
           IG_BASELINE (noise|silence)  IG_ALPHA (linear|power)  IG_ALPHA_POWER (3.0)  IG_SEED (0)

Outputs:
  outputs/tables/wavlm_ig_summary.csv     per-head /s/ contrast + head agreement (time space)
                                          + completeness (scale-floored) / completeness_ok
  outputs/tables/wavlm_ig_completeness.csv  per-(speaker, head) ig_sum, delta_f, abs/rel error,
                                          small_signal flag -- the completeness audit trail (E12)
  outputs/tables/wavlm_ig_phones.csv      per-(phone class, head) mean IG attribution
  outputs/tables/wavlm_ig_path.csv        g(alpha) = f(alpha*x) per (speaker, head, alpha),
                                          endpoints included -- plot to see path curvature
  outputs/figures/prediction/wavlm_ig_phones.png     phone-class profile + /s/ contrast
  outputs/figures/prediction/wavlm_ig_examples.png   spectrogram + IG, /s/ shaded, 2 speakers
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.exceptions import ConvergenceWarning
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

from common import ROOT, PROC, SPEAKERS_CSV, FIG, TABLES, ensure_dirs

PRED_DIR = FIG / 'prediction'
EMB_NPZ = PROC / 'wavlm_embeddings.npz'
TG_DIR = ROOT / 'mfa_textgrids'
WAV_DIR = ROOT / 'clean_wavs'
MODEL_NAME = "microsoft/wavlm-base-plus"
TARGET_SR = 16_000
FRAME_S = 0.02
SAMPLES_PER_FRAME = int(TARGET_SR * FRAME_S)                 # 320
ALPHAS_GRID = np.logspace(-1, 5, 16)
EN_L1, EN_NALPHAS, EN_MAXITER = [0.5, 0.9, 0.99], 20, 3000

IG_SECONDS = float(os.environ.get('IG_SECONDS', 10))
IG_STEPS = int(os.environ.get('IG_STEPS', 128))            # 128 converges both heads (E12)
IG_NSPK = int(os.environ.get('IG_NSPK', 0))                 # 0 = all
IG_BASELINE = os.environ.get('IG_BASELINE', 'noise')        # noise (primary) | silence (see E11)
IG_ALPHA = os.environ.get('IG_ALPHA', 'linear')             # linear | power  (see E10)
IG_ALPHA_POWER = float(os.environ.get('IG_ALPHA_POWER', 3.0))
IG_SEED = int(os.environ.get('IG_SEED', 0))
COMP_TOL = 0.05             # completeness rel. error above this => integral under-resolved


def best_layer():
    if 'WAVLM_SAL_LAYER' in os.environ:
        return int(os.environ['WAVLM_SAL_LAYER'])
    try:
        mt = pd.read_csv(TABLES / 'wavlm_moneytest.csv')
        row = mt[(mt['model'] == 'ridge') & (mt['target'] == 'perceived')]
        return int(row['best_layer'].iloc[0])
    except Exception:
        return 6


def parse_phones(tg_path):
    t = tg_path.read_text(encoding='utf-8')
    block = [x for x in t.split('item [') if re.search(r'name = "phones"', x)][0]
    ivs = re.findall(r'xmin = ([0-9.]+)\s+xmax = ([0-9.]+)\s+text = "([^"]*)"', block)
    out = []
    for a, b, lab in ivs:
        lab = lab.strip()
        m = re.match(r'^([A-Z]+)\d?$', lab) if lab else None
        out.append((float(a), float(b), m.group(1) if m else None))
    return out


def phone_class(base):
    if base is None:
        return 'sil'
    if base == 'S':
        return '/s/'
    if base in {'SH', 'Z', 'ZH'}:
        return 'other sib'
    if base in {'F', 'V', 'TH', 'DH', 'HH'}:
        return 'non-sib fric'
    if base in {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
                'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}:
        return 'vowel'
    if base in {'M', 'N', 'NG', 'L', 'R', 'W', 'Y'}:
        return 'sonorant'
    return 'stop/affr'


def fit_head(kind, Xpool, y):
    head = (RidgeCV(alphas=ALPHAS_GRID) if kind == 'ridge'
            else ElasticNetCV(l1_ratio=EN_L1, n_alphas=EN_NALPHAS, cv=3,
                              max_iter=EN_MAXITER, random_state=0, n_jobs=1))
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), head)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        pipe.fit(Xpool, y)
    return head.coef_ / pipe.named_steps['standardscaler'].scale_        # w_eff


def head_score(input_values, model, layer, w_torch):
    """f(x) = w_eff . mean_t hidden[L](x), per head. One forward pass, no grad."""
    with torch.no_grad():
        pooled = model(input_values=input_values,
                       output_hidden_states=True).hidden_states[layer][0].mean(dim=0)
    return {k: float((w * pooled).sum()) for k, w in w_torch.items()}


def path_quadrature(steps):
    """Sample points + weights for the [0,1] path integral  int_0^1 grad(alpha) d alpha.

    'linear' = the uniform midpoint rule (alpha_j evenly spaced, weight 1/steps) -- the
    original scheme. It was empirically shown to UNDER-RESOLVE this integral: for the
    silence baseline, ~100%+ of f(x)-f(x0) accrues in alpha in [0, 0.01] (WavLM reacts
    violently to the silence->signal onset, then saturates), so uniform sampling lands
    ~1 point in the spike and completeness fails by 78-213% at 20-50 steps (see E10).

    'power' = the change of variables alpha = u^p on a uniform-u midpoint grid. Then
        int_0^1 grad(alpha) d alpha = int_0^1 grad(u^p) * p u^{p-1} du,
    so weight_j = p u_j^{p-1} / steps. This clusters alpha near 0 (weights shrink there
    to compensate), resolving the onset spike while remaining an UNBIASED quadrature of
    the same integral -- completeness stays the independent test that it worked. p=1
    recovers 'linear'."""
    u = (np.arange(1, steps + 1) - 0.5) / steps
    if IG_ALPHA == 'linear':
        return u, np.full(steps, 1.0 / steps)
    p = IG_ALPHA_POWER
    return u ** p, (p * u ** (p - 1)) / steps


def make_baseline(kind, y_audio, iv, feat, seed):
    """IG reference input x0. 'silence' = zeros (E7 default). 'noise' = a phase-
    randomized surrogate of THIS speaker's clip: identical magnitude spectrum, random
    phase, so it destroys temporal/phonetic structure while matching the long-term
    spectral envelope (E11). This is the standard Fourier "surrogate data" null (Theiler
    et al., Physica D 58, 1992) used as a non-zero IG baseline (cf. Sturmfels, Lundberg &
    Lee, Distill 2020, on baseline choice). Run through the same feature extractor as x so
    baseline and input share the extractor's per-clip normalization."""
    if kind == 'silence':
        return torch.zeros_like(iv)
    rng = np.random.default_rng(seed)
    Y = np.fft.rfft(y_audio)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, len(Y)))
    phase[0] = 1.0                                          # DC must stay real
    if len(y_audio) % 2 == 0:
        phase[-1] = 1.0                                    # Nyquist bin too (even N)
    noise = np.fft.irfft(np.abs(Y) * phase, n=len(y_audio)).astype(np.float32)
    return feat(noise, sampling_rate=TARGET_SR, return_tensors='pt').input_values


def integrated_gradients(input_values, base, model, layer, w_torch, steps):
    """IG w.r.t. the model input for each head, along the base -> input_values path.
    One forward + 2 backward/step, plus 2 extra forwards for the completeness targets.

    Returns (ig, comp, path, alphas):
      ig[head]   : (n_samples,) per-sample attribution
      comp[head] : (f_x, f_base) -- completeness holds iff sum(ig) == f_x - f_base.
                   The quadrature over `steps` points is only an approximation of the
                   path integral, so this is the convergence diagnostic: a large relative
                   error means the integral is under-resolved -> raise IG_STEPS / IG_ALPHA_POWER.
      path[head] : [g(alpha) for alpha in alphas], g(alpha) = f(base + alpha*(x-base)).
                   Recorded rather than discarded (free -- already computed); its SHAPE is
                   the finer diagnostic (g near-linear => grid is plenty; sharply curved
                   near alpha=0 => onset spike, needs a denser grid there -- which 'power' gives).
      alphas     : the sample points actually used (non-uniform under 'power')."""
    alphas, weights = path_quadrature(steps)
    grads = {k: torch.zeros_like(input_values) for k in w_torch}
    path = {k: [] for k in w_torch}
    delta = input_values - base
    for a, wq in zip(alphas, weights):
        xa = (base + float(a) * delta).detach().requires_grad_(True)
        out = model(input_values=xa, output_hidden_states=True)
        pooled = out.hidden_states[layer][0].mean(dim=0)                 # (768,) mean over frames
        for ki, (k, w) in enumerate(w_torch.items()):
            target = (w * pooled).sum()
            path[k].append(float(target))                                # g(alpha)
            target.backward(retain_graph=(ki < len(w_torch) - 1))
            grads[k] = grads[k] + float(wq) * xa.grad.detach()           # quadrature-weighted
            xa.grad = None
    ig = {k: (delta * g)[0].numpy() for k, g in grads.items()}
    # completeness targets: the path endpoints themselves (alphas are interior points, so
    # neither alpha=0 nor alpha=1 was visited above -> both need their own forward pass)
    f_x = head_score(input_values, model, layer, w_torch)
    f_base = head_score(base, model, layer, w_torch)
    comp = {k: (f_x[k], f_base[k]) for k in w_torch}
    return ig, comp, path, alphas


def sample_times(n_samples):
    return (np.arange(n_samples) + 0.5) / TARGET_SR


def map_phone(times, intervals):
    starts = np.array([iv[0] for iv in intervals])
    bases = [iv[2] for iv in intervals]
    idx = np.clip(np.searchsorted(starts, times, side='right') - 1, 0, len(intervals) - 1)
    return np.array([bases[i] for i in idx], dtype=object)


def main():
    ensure_dirs(TABLES, PRED_DIR)
    layer = best_layer()
    sched = f'{IG_ALPHA}' + (f'(p={IG_ALPHA_POWER:g})' if IG_ALPHA == 'power' else '')
    print(f'10d_wavlm_ig.py  (integrated gradients | L{layer} | {IG_SECONDS}s window | '
          f'{IG_STEPS} steps | alpha={sched} | baseline={IG_BASELINE} | heads ridge,enet)')

    d = np.load(EMB_NPZ, allow_pickle=True)
    E, ids = d['embeddings'], list(d['file_ids'])
    sp = pd.read_csv(SPEAKERS_CSV).set_index('file_id').reindex(ids)
    y = sp['perceived_mean'].to_numpy(float)
    w = {k: fit_head(k, E[:, layer, :], y) for k in ('ridge', 'enet')}
    w_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in w.items()}
    print(f"  heads: ridge |w|={np.linalg.norm(w['ridge']):.2f}  "
          f"enet nonzero={int((w['enet'] != 0).sum())}/{len(w['enet'])}")

    feat = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = WavLMModel.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    speakers = ids if not IG_NSPK else ids[:IG_NSPK]
    examples = {ids[int(np.argmax(y))]: 'highest perceived',
                ids[int(np.argmin(y))]: 'lowest perceived'}
    ex_cache = {}

    rows, phone_accum, agree = [], {}, {'ridge': [], 'enet': []}
    path_rows = []
    for i, fid in enumerate(speakers, 1):
        y_audio = librosa.load(WAV_DIR / f'{fid}.wav', sr=TARGET_SR, mono=True)[0]
        y_audio = y_audio[:int(IG_SECONDS * TARGET_SR)].astype(np.float32)
        iv = feat(y_audio, sampling_rate=TARGET_SR, return_tensors='pt').input_values
        base_iv = make_baseline(IG_BASELINE, y_audio, iv, feat, seed=IG_SEED + i)
        ig, comp, path, alphas = integrated_gradients(iv, base_iv, model, layer, w_torch, IG_STEPS)

        times = sample_times(ig['ridge'].shape[0])
        intervals = parse_phones(TG_DIR / fid / f'{fid}.TextGrid')
        klass = np.array([phone_class(b) for b in map_phone(times, intervals)])
        speech = klass != 'sil'
        is_s = klass == '/s/'

        rec = {'file_id': fid}
        for k in ('ridge', 'enet'):
            a = ig[k]                                                    # per-sample attribution
            sp_mean = a[speech].mean() if speech.any() else np.nan
            rec[f'{k}_s_mean'] = a[is_s].mean() if is_s.any() else np.nan
            rec[f'{k}_nons_mean'] = a[speech & ~is_s].mean()
            rec[f'{k}_contrast'] = rec[f'{k}_s_mean'] - rec[f'{k}_nons_mean']
            # completeness: sum of attributions must equal the endpoint score gap.
            # Report the ABSOLUTE error too -- the raw per-speaker relative error is
            # ill-defined when a speaker's clip score ~= its baseline score (delta_f ~= 0,
            # a substantive "voice indistinguishable from its spectral surrogate" case, not
            # an integration failure). A scale-floored relative error (E12) is added post-loop.
            f_x, f_b = comp[k]
            rec[f'{k}_ig_sum'] = a.sum()
            rec[f'{k}_delta_f'] = f_x - f_b
            rec[f'{k}_comp_abserr'] = abs(a.sum() - (f_x - f_b))
            rec[f'{k}_comp_relerr'] = abs(a.sum() - (f_x - f_b)) / (abs(f_x - f_b) + 1e-12)
            # g(alpha) curve, endpoints included (alpha=0 baseline .. alpha=1 full clip)
            al = np.concatenate(([0.0], alphas, [1.0]))
            gv = np.array([f_b] + path[k] + [f_x], dtype=float)
            for aa, g_a in zip(al, gv):
                path_rows.append({'file_id': fid, 'head': k, 'alpha': aa, 'g_alpha': g_a})
            # curvature summary: if g were linear in alpha this is exactly 0.5; far from
            # 0.5 means the score accumulates unevenly along the path -> denser grid needed
            rec[f'{k}_path_frac_first_half'] = (
                (float(np.interp(0.5, al, gv)) - f_b) / (f_x - f_b)
                if abs(f_x - f_b) > 1e-12 else np.nan)
            for kl in np.unique(klass[speech]):
                phone_accum.setdefault((kl, k), []).append(a[klass == kl].mean() - sp_mean)
            # frame-aggregated attribution for the agreement r (sum within 20 ms frames)
            nf = len(a) // SAMPLES_PER_FRAME
            fa = a[:nf * SAMPLES_PER_FRAME].reshape(nf, SAMPLES_PER_FRAME).sum(1)
            fk = klass[:nf * SAMPLES_PER_FRAME:SAMPLES_PER_FRAME][:nf]
            agree[k].append(fa[fk != 'sil'])
        rec['n_s_frames'] = int(is_s.sum() // SAMPLES_PER_FRAME)
        rows.append(rec)
        if fid in examples:
            ex_cache[fid] = (y_audio, ig['ridge'], ig['enet'], intervals)
        if i % 5 == 0 or i == len(speakers):
            print(f"  [{i:2d}/{len(speakers)}] {fid[:30]:30} "
                  f"ridge /s/-contrast={rec['ridge_contrast']:+.2e}  "
                  f"enet={rec['enet_contrast']:+.2e}")

    fr = pd.DataFrame(rows)
    ar = np.concatenate(agree['ridge'])
    ae = np.concatenate(agree['enet'])
    r_agree = pearsonr(ar, ae).statistic

    # scale-floored completeness (E12): normalise the absolute error by each speaker's own
    # |delta_f|, but floor the denominator at the head's median |delta_f| so a speaker whose
    # clip score ~= its baseline score (delta_f ~= 0) is judged on absolute closeness to the
    # typical signal rather than blowing up a near-zero denominator. Equals the raw relative
    # error for on-scale speakers; caps the pathology for small-signal ones.
    comp_rows = []
    for k in ('ridge', 'enet'):
        scale = fr[f'{k}_delta_f'].abs().median()
        denom = fr[f'{k}_delta_f'].abs().clip(lower=scale)
        fr[f'{k}_comp_relerr_robust'] = fr[f'{k}_comp_abserr'] / denom
        for _, r in fr.iterrows():
            comp_rows.append({'file_id': r['file_id'], 'head': k, 'ig_sum': r[f'{k}_ig_sum'],
                              'delta_f': r[f'{k}_delta_f'], 'comp_abserr': r[f'{k}_comp_abserr'],
                              'comp_relerr_raw': r[f'{k}_comp_relerr'],
                              'comp_relerr_robust': r[f'{k}_comp_relerr_robust'],
                              'small_signal': bool(abs(r[f'{k}_delta_f']) < 0.1 * scale)})

    summ = []
    for k in ('ridge', 'enet'):
        c = fr[f'{k}_contrast'].dropna()
        stat, p = wilcoxon(c, alternative='greater') if len(c) > 1 and c.std() > 0 else (np.nan, np.nan)
        cr = fr[f'{k}_comp_relerr_robust'].dropna()          # verdict metric (scale-floored)
        ce = fr[f'{k}_comp_relerr'].dropna()                 # raw, for reference
        scale = fr[f'{k}_delta_f'].abs().median()
        n_small = int((fr[f'{k}_delta_f'].abs() < scale * 0.1).sum())
        ok = bool(len(cr) and cr.max() <= COMP_TOL)
        verdict = 'OK' if ok else (f'UNDER-RESOLVED -> raise IG_STEPS / IG_ALPHA_POWER '
                                   f'(now {IG_STEPS} steps, alpha={sched})')
        summ.append({'head': k, 'layer': layer, 'window_s': IG_SECONDS, 'ig_steps': IG_STEPS,
                     'baseline': IG_BASELINE, 'alpha_schedule': IG_ALPHA,
                     'alpha_power': IG_ALPHA_POWER if IG_ALPHA == 'power' else np.nan,
                     'n_speakers': len(c), 's_contrast_mean': c.mean(),
                     's_contrast_median': c.median(), 'frac_speakers_positive': (c > 0).mean(),
                     'wilcoxon_p_greater': p, 'ridge_enet_frame_r': r_agree,
                     'completeness_relerr_robust_mean': cr.mean(),
                     'completeness_relerr_robust_max': cr.max(),
                     'completeness_relerr_raw_max': ce.max(),
                     'completeness_abserr_max': fr[f'{k}_comp_abserr'].max(),
                     'n_small_signal': n_small, 'completeness_ok': ok,
                     'path_frac_first_half': fr[f'{k}_path_frac_first_half'].mean()})
        print(f"\n[{k}] /s/ IG contrast: mean={c.mean():+.2e} median={c.median():+.2e}  "
              f"{100*(c>0).mean():.0f}% speakers positive  Wilcoxon p(>0)={p:.4f}")
        print(f"[{k}] completeness (scale-floored |sum(IG)-df|): mean={cr.mean():.2%} "
              f"max={cr.max():.2%}  (tol {COMP_TOL:.0%})  -> {verdict}")
        print(f"[{k}]   raw per-speaker relerr max={ce.max():.2%} "
              f"({n_small} small-signal speaker(s) with |df|<10% of median -> raw relerr not meaningful)")
        print(f"[{k}] path shape: {fr[f'{k}_path_frac_first_half'].mean():.1%} of the score "
              f"gap accrues by alpha=0.5  (0.5 = g linear in alpha; far from it = curved)")
    print(f"\n[agreement] ridge vs enet IG (time space): Pearson r={r_agree:+.3f} "
          f"(over {len(ar)} speech frames)  [10c weight-space r was 0.45]")

    pd.DataFrame(summ).to_csv(TABLES / 'wavlm_ig_summary.csv', index=False)
    pd.DataFrame(comp_rows).to_csv(TABLES / 'wavlm_ig_completeness.csv', index=False)
    prof = pd.DataFrame([{'phone_class': kl, 'head': k, 'mean_ig': np.mean(v),
                          'sd': np.std(v, ddof=1) if len(v) > 1 else np.nan, 'n_speakers': len(v)}
                         for (kl, k), v in sorted(phone_accum.items())])
    prof.to_csv(TABLES / 'wavlm_ig_phones.csv', index=False)
    pd.DataFrame(path_rows).to_csv(TABLES / 'wavlm_ig_path.csv', index=False)

    fig_phones(prof, summ, fr, r_agree, PRED_DIR / 'wavlm_ig_phones.png')
    fig_examples(ex_cache, examples, layer, PRED_DIR / 'wavlm_ig_examples.png')
    print('\n  -> tables/wavlm_ig_summary.csv, wavlm_ig_phones.csv, wavlm_ig_completeness.csv')
    print('  -> figures/prediction/wavlm_ig_phones.png, wavlm_ig_examples.png')
    print('done.')


def fig_phones(prof, summ, fr, r_agree, path):
    order = ['/s/', 'other sib', 'non-sib fric', 'vowel', 'sonorant', 'stop/affr']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    width = 0.38
    for j, k in enumerate(('ridge', 'enet')):
        sub = prof[prof['head'] == k].set_index('phone_class').reindex(order)
        ax.barh(np.arange(len(order)) + (j - 0.5) * width, sub['mean_ig'], height=width,
                color='#3a6ea5' if k == 'ridge' else '#d08a2e', label=k)
    ax.axvline(0, color='#888', lw=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel('mean IG attribution (per-sample, speaker-centered)')
    ax.set_title('Which phone classes drive "sounds gay"?  (input-space IG)', fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    for j, k in enumerate(('ridge', 'enet')):
        c = fr[f'{k}_contrast'].dropna()
        ax.scatter(np.full(len(c), j) + np.random.uniform(-0.08, 0.08, len(c)), c,
                   s=28, color='#3a6ea5' if k == 'ridge' else '#d08a2e', alpha=0.7,
                   edgecolor='k', linewidth=0.3)
        ax.plot([j - 0.2, j + 0.2], [c.mean(), c.mean()], color='k', lw=2)
    ax.axhline(0, color='#888', lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"ridge\np={summ[0]['wilcoxon_p_greater']:.3f}",
                        f"enet\np={summ[1]['wilcoxon_p_greater']:.3f}"])
    ax.set_ylabel('per-speaker /s/ IG contrast  (/s/ - other speech)')
    ax.set_title(f'Does /s/ stand out?  (ridge-enet time-space r={r_agree:.2f}; 10c weight r=0.45)',
                 fontsize=11)
    fig.suptitle('WavLM integrated-gradients saliency by phone (input space, n='
                 f"{summ[0]['n_speakers']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_examples(ex_cache, examples, layer, path):
    examples = {k: v for k, v in examples.items() if k in ex_cache}
    if not examples:
        print('  (skipping example figure: exemplar speakers not in this run)')
        return
    fig, axes = plt.subplots(2, len(examples), figsize=(15, 7), squeeze=False)
    for col, (fid, label) in enumerate(examples.items()):
        if fid not in ex_cache:
            continue
        y_audio, ig_r, ig_e, intervals = ex_cache[fid]
        dur = len(y_audio) / TARGET_SR
        S = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y_audio, sr=TARGET_SR, n_mels=80, fmax=8000), ref=np.max)
        axtop = axes[0][col]
        axtop.imshow(S, origin='lower', aspect='auto', extent=[0, dur, 0, 8000], cmap='magma')
        axtop.set_ylabel('freq (Hz)')
        axtop.set_title(f'{fid}  ({label})', fontsize=10)
        # per-frame aggregated IG
        nf = len(ig_r) // SAMPLES_PER_FRAME
        t = (np.arange(nf) + 0.5) * FRAME_S
        fr_r = ig_r[:nf * SAMPLES_PER_FRAME].reshape(nf, SAMPLES_PER_FRAME).sum(1)
        fr_e = ig_e[:nf * SAMPLES_PER_FRAME].reshape(nf, SAMPLES_PER_FRAME).sum(1)
        axb = axes[1][col]
        axb.plot(t, fr_r, color='#3a6ea5', lw=0.9, label='ridge')
        axb.plot(t, fr_e, color='#d08a2e', lw=0.9, alpha=0.8, label='enet')
        for a, b, base in intervals:
            if base == 'S' and a < dur:
                axb.axvspan(a, b, color='#2ca25f', alpha=0.30, lw=0)
        axb.axhline(0, color='#aaa', lw=0.6)
        axb.set_xlim(0, dur)
        axb.set_xlabel('time (s)')
        axb.set_ylabel('IG (per 20 ms frame)')
        if col == 0:
            axb.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Integrated-gradients saliency (L{layer}); green = /s/ intervals (MFA)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
