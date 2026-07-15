"""10c_wavlm_saliency.py -- where in the signal does WavLM read "gay voice"?

10b showed a frozen WavLM predicts PERCEIVED gayness strongly (ridge L6 rho=0.73)
and that its signal does NOT reduce to the hand-measured /s/ cue (S_cog). This asks
the natural follow-up: WHERE in each utterance does the model's perceptual signal
live, and does it land on the /s/?

Method -- exact linear frame attribution (not occlusion).
  The head is LINEAR on a MEAN-POOLED embedding, so the prediction decomposes
  exactly into per-frame contributions -- no perturbation, no re-fit, no leak:

      pred = intercept + coef . z,   z = (pooled - mu) / sigma,   pooled = mean_t h_t
           = const + mean_t [ (coef/sigma) . h_t ]
           = const + mean_t s_t ,    s_t := w_eff . h_t ,  w_eff = coef / sigma

  Each 20 ms frame t therefore pushes the prediction toward "sounds gay" by exactly
  s_t / T. (Occlusion is the wrong tool here: silencing a ~2%-of-frames /s/ barely
  moves a whole-utterance mean-pool; the linear decomposition is faithful AND free.)

We fit BOTH heads (ridge, enet) on all 50 speakers at the canonical layer, map every
frame to its MFA phone, and ask:
  1. Do /s/ frames carry a positive within-speaker saliency contrast (model leans on
     /s/ more than on the speaker's other speech)?  Paired across 50 speakers.
  2. Which phone classes light up most (what ELSE, beyond /s/, does the model read)?
  3. Do the ridge and enet saliency landscapes AGREE (head-invariant "where")?

Canonical layer = ridge's best PERCEIVED layer from 10b (read from wavlm_moneytest.csv;
env WAVLM_SAL_LAYER overrides). Both heads are fit at this SAME layer so the ridge-vs-
enet comparison is controlled (only the estimator differs).

Outputs:
  outputs/tables/wavlm_saliency_summary.csv    per-head /s/ contrast + head agreement
  outputs/tables/wavlm_saliency_phones.csv     per-(phone class, head) mean contribution
  outputs/figures/prediction/wavlm_saliency_examples.png   spectrogram + saliency, 2 speakers
  outputs/figures/prediction/wavlm_saliency_phones.png     phone-class profile + /s/ contrast
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
CHUNK_S = 20
FRAME_S = 0.02                                  # WavLM stride: 20 ms/frame
ALPHAS = np.logspace(-1, 5, 16)
EN_L1, EN_NALPHAS, EN_MAXITER = [0.5, 0.9, 0.99], 20, 3000


def best_layer():
    if 'WAVLM_SAL_LAYER' in os.environ:
        return int(os.environ['WAVLM_SAL_LAYER'])
    try:
        mt = pd.read_csv(TABLES / 'wavlm_moneytest.csv')
        row = mt[(mt['model'] == 'ridge') & (mt['target'] == 'perceived')]
        return int(row['best_layer'].iloc[0])
    except Exception:
        return 6                                # ridge best from the 10b run


def parse_phones(tg_path):
    """Ordered (start, end, base) intervals of the phones tier; base=None for silence."""
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
    """Collapse ARPABET base -> a small display class for the profile."""
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


def load_wavlm():
    feat = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = WavLMModel.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return feat, model


def frame_states(y, feat, model, layer):
    """Per-frame hidden states at `layer`: (n_frames, 768) with global frame times (s)."""
    chunk = CHUNK_S * TARGET_SR
    states, times = [], []
    for c, s in enumerate(range(0, len(y), chunk)):
        seg = y[s:s + chunk]
        if len(seg) < int(0.5 * TARGET_SR):
            continue
        inp = feat(seg, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            out = model(inp.input_values, output_hidden_states=True)
        h = out.hidden_states[layer][0].numpy()                  # (T, 768)
        states.append(h)
        times.append(c * CHUNK_S + (np.arange(h.shape[0]) + 0.5) * FRAME_S)
    return np.concatenate(states), np.concatenate(times)


def fit_head(kind, Xpool, y):
    """Fit head on pooled embeddings; return effective per-dim weights w_eff = coef/sigma."""
    head = (RidgeCV(alphas=ALPHAS) if kind == 'ridge'
            else ElasticNetCV(l1_ratio=EN_L1, n_alphas=EN_NALPHAS, cv=3,
                              max_iter=EN_MAXITER, random_state=0, n_jobs=1))
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), head)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        pipe.fit(Xpool, y)
    sigma = pipe.named_steps['standardscaler'].scale_
    coef = head.coef_
    return coef / sigma                                          # w_eff


def map_phone(times, intervals):
    """Vectorized frame-time -> phone base via interval starts."""
    starts = np.array([iv[0] for iv in intervals])
    bases = [iv[2] for iv in intervals]
    idx = np.searchsorted(starts, times, side='right') - 1
    idx = np.clip(idx, 0, len(intervals) - 1)
    return [bases[i] for i in idx]


def main():
    ensure_dirs(TABLES, PRED_DIR)
    layer = best_layer()
    print(f'10c_wavlm_saliency.py  (canonical layer L{layer}; heads = ridge, enet)')

    d = np.load(EMB_NPZ, allow_pickle=True)
    E, ids = d['embeddings'], list(d['file_ids'])
    sp = pd.read_csv(SPEAKERS_CSV).set_index('file_id').reindex(ids)
    y = sp['perceived_mean'].to_numpy(float)
    Xpool = E[:, layer, :]                                       # (n, 768) at canonical layer

    w = {k: fit_head(k, Xpool, y) for k in ('ridge', 'enet')}
    print(f"  fitted heads: ridge |w_eff|={np.linalg.norm(w['ridge']):.2f}  "
          f"enet nonzero={int((w['enet'] != 0).sum())}/{len(w['enet'])}")

    feat, model = load_wavlm()
    examples = {ids[int(np.argmax(y))]: 'highest perceived',
                ids[int(np.argmin(y))]: 'lowest perceived'}
    ex_cache = {}

    frame_rows = []           # per (speaker, head): S vs non-S speech means
    agree_r, agree_e = [], []  # frame-level saliency for ridge/enet agreement (speech frames)
    phone_accum = {}          # (klass, head) -> list of frame saliencies (speaker-mean-centered)

    for i, fid in enumerate(ids, 1):
        y_audio = librosa.load(WAV_DIR / f'{fid}.wav', sr=TARGET_SR, mono=True)[0].astype(np.float32)
        H, times = frame_states(y_audio, feat, model, layer)
        intervals = parse_phones(TG_DIR / fid / f'{fid}.TextGrid')
        bases = map_phone(times, intervals)
        klass = np.array([phone_class(b) for b in bases])
        is_speech = klass != 'sil'
        is_s = klass == '/s/'

        rec = {'file_id': fid}
        for k in ('ridge', 'enet'):
            s = H @ w[k]                                         # (n_frames,) frame saliency
            sp_mean = s[is_speech].mean() if is_speech.any() else np.nan
            rec[f'{k}_s_mean'] = s[is_s].mean() if is_s.any() else np.nan
            rec[f'{k}_nons_mean'] = s[is_speech & ~is_s].mean()
            rec[f'{k}_contrast'] = rec[f'{k}_s_mean'] - rec[f'{k}_nons_mean']
            # per phone-class contribution, centered on this speaker's speech mean
            for kl in np.unique(klass[is_speech]):
                phone_accum.setdefault((kl, k), []).append(s[klass == kl].mean() - sp_mean)
            if k == 'ridge':
                agree_r.append(s[is_speech])
            else:
                agree_e.append(s[is_speech])
        rec['n_s_frames'] = int(is_s.sum())
        frame_rows.append(rec)
        if fid in examples:
            ex_cache[fid] = (y_audio, times, H @ w['ridge'], H @ w['enet'], intervals, klass)
        if i % 10 == 0 or i == len(ids):
            print(f"  [{i:2d}/{len(ids)}] {fid[:30]:30} s-frames={int(is_s.sum()):4d}")

    fr = pd.DataFrame(frame_rows)

    # --- 1. /s/ contrast: paired Wilcoxon across speakers (is /s/ > other speech?) ---
    summ = []
    for k in ('ridge', 'enet'):
        c = fr[f'{k}_contrast'].dropna()
        stat, p = wilcoxon(c, alternative='greater')
        summ.append({'head': k, 'layer': layer, 'n_speakers': len(c),
                     's_contrast_mean': c.mean(), 's_contrast_median': c.median(),
                     'frac_speakers_positive': (c > 0).mean(),
                     'wilcoxon_p_greater': p})
        print(f"\n[{k}] /s/ saliency contrast: mean={c.mean():+.3f} "
              f"median={c.median():+.3f}  {100*(c>0).mean():.0f}% of speakers positive  "
              f"Wilcoxon p(>0)={p:.4f}")

    # --- 3. ridge vs enet frame-level agreement ---
    ar = np.concatenate(agree_r)
    ae = np.concatenate(agree_e)
    r_agree = pearsonr(ar, ae).statistic
    for row in summ:
        row['ridge_enet_frame_r'] = r_agree
    print(f"\n[agreement] ridge vs enet frame saliency: Pearson r={r_agree:+.3f} "
          f"(over {len(ar)} speech frames)")

    pd.DataFrame(summ).to_csv(TABLES / 'wavlm_saliency_summary.csv', index=False)

    # --- 2. phone-class profile table ---
    prof = []
    for (kl, k), vals in sorted(phone_accum.items()):
        v = np.array(vals)
        prof.append({'phone_class': kl, 'head': k, 'mean_contribution': v.mean(),
                     'sd': v.std(ddof=1) if len(v) > 1 else np.nan, 'n_speakers': len(v)})
    prof = pd.DataFrame(prof)
    prof.to_csv(TABLES / 'wavlm_saliency_phones.csv', index=False)

    fig_phones(prof, summ, fr, PRED_DIR / 'wavlm_saliency_phones.png')
    fig_examples(ex_cache, examples, layer, PRED_DIR / 'wavlm_saliency_examples.png')
    print('\n  -> tables/wavlm_saliency_summary.csv, wavlm_saliency_phones.csv')
    print('  -> figures/prediction/wavlm_saliency_phones.png, wavlm_saliency_examples.png')
    print('done.')


def fig_phones(prof, summ, fr, path):
    order = ['/s/', 'other sib', 'non-sib fric', 'vowel', 'sonorant', 'stop/affr']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    width = 0.38
    for j, k in enumerate(('ridge', 'enet')):
        sub = prof[prof['head'] == k].set_index('phone_class').reindex(order)
        xpos = np.arange(len(order)) + (j - 0.5) * width
        ax.barh(xpos, sub['mean_contribution'], height=width,
                color='#3a6ea5' if k == 'ridge' else '#d08a2e', label=k)
    ax.axvline(0, color='#888', lw=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel('mean saliency contribution\n(frame saliency - speaker speech mean)')
    ax.set_title('Which phone classes push "sounds gay"?', fontsize=11)
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
    ax.set_ylabel('per-speaker /s/ saliency contrast  (/s/ - other speech)')
    ax.set_title(f"Does /s/ stand out?  (ridge-enet frame r={summ[0]['ridge_enet_frame_r']:.2f})",
                 fontsize=11)
    fig.suptitle('WavLM saliency by phone: where the model reads perceived "gay voice" (n=50)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_examples(ex_cache, examples, layer, path):
    fig, axes = plt.subplots(2, len(ex_cache), figsize=(15, 7), squeeze=False)
    for col, (fid, label) in enumerate(examples.items()):
        y_audio, times, s_r, s_e, intervals, klass = ex_cache[fid]
        # top: log-mel spectrogram
        S = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y_audio, sr=TARGET_SR, n_mels=80, fmax=8000), ref=np.max)
        dur = len(y_audio) / TARGET_SR
        t0, t1 = 0, min(dur, 12)                                # show first 12 s for legibility
        axtop = axes[0][col]
        axtop.imshow(S, origin='lower', aspect='auto', extent=[0, dur, 0, 8000], cmap='magma')
        axtop.set_xlim(t0, t1)
        axtop.set_ylabel('freq (Hz)')
        axtop.set_title(f'{fid}  ({label})', fontsize=10)
        # bottom: ridge + enet saliency, /s/ intervals shaded
        axb = axes[1][col]
        axb.plot(times, s_r, color='#3a6ea5', lw=0.9, label='ridge')
        axb.plot(times, s_e, color='#d08a2e', lw=0.9, alpha=0.8, label='enet')
        for a, b, base in intervals:
            if base == 'S' and a < t1:
                axb.axvspan(a, b, color='#2ca25f', alpha=0.30, lw=0)
        axb.axhline(0, color='#aaa', lw=0.6)
        axb.set_xlim(t0, t1)
        axb.set_xlabel('time (s)')
        axb.set_ylabel('frame saliency  s_t')
        if col == 0:
            axb.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Frame-level WavLM saliency (L{layer}); green = /s/ intervals (MFA)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
