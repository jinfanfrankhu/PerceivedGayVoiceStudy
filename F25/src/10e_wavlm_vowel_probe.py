"""10e_wavlm_vowel_probe.py -- WHICH vowel property carries WavLM's perceived signal?

10d (integrated gradients) showed both heads agree the model reads perceived "gay voice"
in the VOWELS, not /s/. This pins that down: it isolates the model's VOWEL-REGION readout
per speaker and asks which hand-measured vowel feature it lines up with -- turning "vowels"
into a specific, interpretable cue.

Method (10c-speed, no IG re-run needed). The head is linear on a mean-pooled embedding, so a
"region readout" is just the head applied to the region-restricted mean embedding:

    readout_R(speaker) = w_eff . mean_{t in region R} h_t ,   w_eff = coef / scale

We compute readouts over (a) ALL frames [= the model's perceived prediction], (b) VOWEL
frames, (c) /s/ frames, and (d) each vowel TYPE (AY, EY, ...). Then Spearman-correlate the
VOWEL readout against the hand vowel features (segmental_speaker.csv) across 50 speakers,
with S_cog and the /s/ readout as reference contrasts. Both heads (ridge, enet) reported --
if they agree on which hand feature aligns, the mechanism is head-robust.

Not circular: the readout is the model's vowel REPRESENTATION projected onto the perceived-
predicting direction; the hand features are independent acoustic measurements. We also report
the /s/ readout and a first-order partial (vowel readout vs AY_z2 controlling for S_cog) to
separate the vowel cue from the sibilant one.

Outputs:
  outputs/tables/wavlm_vowel_probe.csv     hand-feature x head Spearman (vowel & /s/ readouts)
  outputs/tables/wavlm_vowel_regions.csv   region/per-vowel readout vs perceived
  outputs/figures/prediction/wavlm/vowel_probe.png     region-vs-perceived + which hand feature
  outputs/figures/prediction/wavlm/vowel_byvowel.png   which vowel type carries it
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
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.exceptions import ConvergenceWarning
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

from common import ROOT, PROC, SPEAKERS_CSV, FIG, TABLES, ensure_dirs

PRED_DIR = FIG / 'prediction' / 'wavlm'
EMB_NPZ = PROC / 'wavlm_embeddings.npz'
TG_DIR = ROOT / 'mfa_textgrids'
WAV_DIR = ROOT / 'clean_wavs'
MODEL_NAME = "microsoft/wavlm-base-plus"
TARGET_SR = 16_000
CHUNK_S = 20
ALPHAS = np.logspace(-1, 5, 16)
EN_L1, EN_NALPHAS, EN_MAXITER = [0.5, 0.9, 0.99], 20, 3000
N_MIN_FRAMES = 5

VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
          'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
VOWEL_TYPES = ['AY', 'EY', 'AE', 'AW', 'AO', 'OW', 'IY', 'IH', 'EH', 'AA', 'AH', 'UW']
# curated interpretable hand features (all verified present in segmental_speaker.csv)
HAND_FEATS = ['AY_z2', 'EY_z2', 'AE_z2', 'OW_z2', 'IY_z2', 'AW_z1', 'AO_z1',
              'front_f2', 'vowel_space_area', 'AY_trajlen', 'EY_trajlen',
              'v_cpps', 'v_h1h2', 'S_cog']


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


def map_phone(times, intervals):
    starts = np.array([iv[0] for iv in intervals])
    bases = [iv[2] for iv in intervals]
    idx = np.clip(np.searchsorted(starts, times, side='right') - 1, 0, len(intervals) - 1)
    return np.array([bases[i] for i in idx], dtype=object)


def fit_head(kind, Xpool, y):
    head = (RidgeCV(alphas=ALPHAS) if kind == 'ridge'
            else ElasticNetCV(l1_ratio=EN_L1, n_alphas=EN_NALPHAS, cv=3,
                              max_iter=EN_MAXITER, random_state=0, n_jobs=1))
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), head)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        pipe.fit(Xpool, y)
    return head.coef_ / pipe.named_steps['standardscaler'].scale_


def frame_states(y, feat, model, layer):
    chunk = CHUNK_S * TARGET_SR
    states, times = [], []
    for c, s in enumerate(range(0, len(y), chunk)):
        seg = y[s:s + chunk]
        if len(seg) < int(0.5 * TARGET_SR):
            continue
        inp = feat(seg, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            out = model(inp.input_values, output_hidden_states=True)
        h = out.hidden_states[layer][0].numpy()
        states.append(h)
        times.append(c * CHUNK_S + (np.arange(h.shape[0]) + 0.5) * 0.02)
    return np.concatenate(states), np.concatenate(times)


def partial_spearman(x, y, z):
    """First-order partial Spearman: corr(x,y) controlling for z."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    rxy = spearmanr(x, y).statistic
    rxz = spearmanr(x, z).statistic
    ryz = spearmanr(y, z).statistic
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / denom if denom > 0 else np.nan


def main():
    ensure_dirs(TABLES, PRED_DIR)
    layer = best_layer()
    print(f'10e_wavlm_vowel_probe.py  (L{layer}; heads ridge, enet)')

    d = np.load(EMB_NPZ, allow_pickle=True)
    E, ids = d['embeddings'], list(d['file_ids'])
    sp = pd.read_csv(SPEAKERS_CSV).set_index('file_id').reindex(ids)
    seg = pd.read_csv(PROC / 'segmental_speaker.csv').set_index('file_id').reindex(ids)
    y = sp['perceived_mean'].to_numpy(float)
    w = {k: fit_head(k, E[:, layer, :], y) for k in ('ridge', 'enet')}
    print(f"  heads: ridge |w|={np.linalg.norm(w['ridge']):.2f}  "
          f"enet nonzero={int((w['enet'] != 0).sum())}/{len(w['enet'])}")

    feat = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = WavLMModel.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for i, fid in enumerate(ids, 1):
        y_audio = librosa.load(WAV_DIR / f'{fid}.wav', sr=TARGET_SR, mono=True)[0].astype(np.float32)
        H, times = frame_states(y_audio, feat, model, layer)
        bases = map_phone(times, parse_phones(TG_DIR / fid / f'{fid}.TextGrid'))
        vmask = np.array([b in VOWELS for b in bases])
        smask = bases == 'S'
        rec = {'file_id': fid}
        for k in ('ridge', 'enet'):
            rec[f'{k}_full'] = float(w[k] @ H.mean(0))
            rec[f'{k}_vowel'] = float(w[k] @ H[vmask].mean(0)) if vmask.sum() else np.nan
            rec[f'{k}_s'] = float(w[k] @ H[smask].mean(0)) if smask.sum() >= N_MIN_FRAMES else np.nan
            for vw in VOWEL_TYPES:
                msk = bases == vw
                rec[f'{k}_{vw}'] = float(w[k] @ H[msk].mean(0)) if msk.sum() >= N_MIN_FRAMES else np.nan
        rows.append(rec)
        if i % 10 == 0 or i == len(ids):
            print(f"  [{i:2d}/{len(ids)}] {fid[:30]:30} vowel-frames={int(vmask.sum()):4d}")

    R = pd.DataFrame(rows).set_index('file_id')

    # --- 1. region readout vs perceived ---
    print("\n[region readout vs perceived]  (does the vowel region carry it? does /s/?)")
    region_rows = []
    for k in ('ridge', 'enet'):
        for reg in ('full', 'vowel', 's'):
            rho = spearmanr(R[f'{k}_{reg}'], y, nan_policy='omit').statistic
            region_rows.append({'head': k, 'region': reg, 'spearman_vs_perceived': rho})
            print(f"   {k:5} {reg:6} rho(readout, perceived) = {rho:+.3f}")

    # --- 2. vowel readout vs hand features (the money question) ---
    print("\n[vowel readout vs hand features]  (which hand cue does the model's vowel signal track?)")
    feat_rows = []
    for hf in HAND_FEATS:
        h = seg[hf].to_numpy(float)
        row = {'hand_feature': hf}
        for k in ('ridge', 'enet'):
            row[f'{k}_vowel_rho'] = spearmanr(R[f'{k}_vowel'], h, nan_policy='omit').statistic
            row[f'{k}_s_rho'] = spearmanr(R[f'{k}_s'], h, nan_policy='omit').statistic
        row['ridge_vowel_partial_vs_Scog'] = partial_spearman(
            R['ridge_vowel'].to_numpy(float), h, seg['S_cog'].to_numpy(float)) if hf != 'S_cog' else np.nan
        feat_rows.append(row)
    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df.reindex(feat_df['ridge_vowel_rho'].abs().sort_values(ascending=False).index)
    for _, r in feat_df.iterrows():
        print(f"   {r['hand_feature']:16} vowel-readout rho: ridge={r['ridge_vowel_rho']:+.3f} "
              f"enet={r['enet_vowel_rho']:+.3f}   (/s/-readout ridge={r['ridge_s_rho']:+.3f})")

    # --- 3. per-vowel readout vs perceived ---
    byv = []
    for vw in VOWEL_TYPES:
        row = {'vowel': vw}
        for k in ('ridge', 'enet'):
            col = R[f'{k}_{vw}']
            row[f'{k}_rho'] = spearmanr(col, y, nan_policy='omit').statistic
            row[f'{k}_n'] = int(col.notna().sum())
        byv.append(row)
    byv_df = pd.DataFrame(byv)

    pd.DataFrame(region_rows).to_csv(TABLES / 'wavlm_vowel_regions.csv', index=False)
    feat_df.to_csv(TABLES / 'wavlm_vowel_probe.csv', index=False)
    byv_df.to_csv(TABLES / 'wavlm_vowel_byvowel.csv', index=False)

    fig_probe(region_rows, feat_df, PRED_DIR / 'vowel_probe.png')
    fig_byvowel(byv_df, PRED_DIR / 'vowel_byvowel.png')
    print('\n  -> tables/wavlm_vowel_probe.csv, wavlm_vowel_regions.csv, wavlm_vowel_byvowel.csv')
    print('  -> figures/prediction/wavlm/vowel_probe.png, vowel_byvowel.png')
    print('done.')


def fig_probe(region_rows, feat_df, path):
    rr = pd.DataFrame(region_rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.5]})
    ax = axes[0]
    regs = ['full', 'vowel', 's']
    xpos = np.arange(len(regs))
    for j, k in enumerate(('ridge', 'enet')):
        vals = [rr[(rr['head'] == k) & (rr['region'] == rg)]['spearman_vs_perceived'].iloc[0] for rg in regs]
        ax.bar(xpos + (j - 0.5) * 0.38, vals, width=0.38,
               color='#3a6ea5' if k == 'ridge' else '#d08a2e', label=k)
    ax.axhline(0, color='#888', lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(['full\n(=prediction)', 'VOWEL\nregion', '/s/\nregion'])
    ax.set_ylabel('Spearman(region readout, perceived)')
    ax.set_title('Region readout vs perceived (BETWEEN-speaker)\n'
                 'vowel highest; /s/ also tracks (within-utterance /s/ is neg — see 10d)',
                 fontsize=9)
    ax.legend(fontsize=9)

    ax = axes[1]
    order = feat_df['hand_feature'].tolist()[::-1]
    ypos = np.arange(len(order))
    for j, k in enumerate(('ridge', 'enet')):
        vals = [feat_df[feat_df['hand_feature'] == hf][f'{k}_vowel_rho'].iloc[0] for hf in order]
        ax.barh(ypos + (j - 0.5) * 0.38, vals, height=0.38,
                color='#3a6ea5' if k == 'ridge' else '#d08a2e', label=k)
    ax.axvline(0, color='#888', lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([('* ' + hf) if hf == 'S_cog' else hf for hf in order])
    ax.set_xlabel('Spearman(model VOWEL readout, hand feature)')
    ax.set_title('Which hand vowel property does the model\'s vowel signal track?\n'
                 '(* S_cog = the /s/ cue, shown as reference)', fontsize=11)
    ax.legend(fontsize=9)
    fig.suptitle('WavLM vowel-region probe: pinning the mechanism (n=50)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_byvowel(byv_df, path):
    order = byv_df['vowel'].tolist()
    xpos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, k in enumerate(('ridge', 'enet')):
        ax.bar(xpos + (j - 0.5) * 0.38, byv_df[f'{k}_rho'], width=0.38,
               color='#3a6ea5' if k == 'ridge' else '#d08a2e', label=k)
    ax.axhline(0, color='#888', lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{v}\n(n={int(byv_df[f'ridge_n'].iloc[i])})" for i, v in enumerate(order)])
    ax.set_ylabel('Spearman(per-vowel readout, perceived)')
    ax.set_title('Which vowel type carries WavLM\'s perceived signal? (n speakers with >=5 frames)',
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
