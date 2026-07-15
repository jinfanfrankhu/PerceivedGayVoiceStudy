"""10_wavlm_extract.py — frozen WavLM embedding extraction (run-once cache).

Feeds each speaker's audio through a *frozen*, pretrained WavLM and caches the
per-layer, mean-pooled embedding. This is the extraction half of the flagship
"does a SOTA speech model reduce to /s/?" experiment; the honest prediction
harness lives in 10b_wavlm_probe.py and consumes the cache written here.

Design notes (why these choices):
  - Model: microsoft/wavlm-base-plus. "base" = 12 transformer layers (13 hidden
    states incl. the CNN feature projection), 768-dim; "plus" was trained on the
    largest/most varied mix and is strong on speaker/paralinguistic traits.
  - FROZEN. We never update WavLM's weights (that would overfit catastrophically
    at n=50). All the heavy learning already happened on 94k hrs; we only borrow
    its ears. The tiny ridge head is fit downstream in 10b.
  - All 13 hidden-state layers are kept. WHICH layer predicts best is a
    hyperparameter that must be chosen inside cross-validation (done in 10b),
    never here — so we cache every layer and decide later.
  - Pooling = mean over time -> one vector per (speaker, layer). Long (~60s)
    clips are chunked into 20s windows purely to bound self-attention memory on
    CPU; each chunk's mean is length-weighted so the result equals a single
    global mean-pool over the whole utterance. n stays honestly 50 (one row/spk).

Output: data/processed/wavlm_embeddings.npz
  embeddings : float32 (n_speakers, n_layers=13, dim=768)
  file_ids   : str    (n_speakers,)   canonical names, == wav filename stem
  n_frames   : int    (n_speakers,)   total 16kHz frames pooled (sanity)
  duration_s : float  (n_speakers,)   audio duration in seconds (sanity)
"""
import sys
import time
import numpy as np
import librosa
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from common import ROOT, PROC, ensure_dirs

MODEL_NAME = "microsoft/wavlm-base-plus"
TARGET_SR = 16_000              # WavLM expects 16 kHz mono
CHUNK_S = 20                    # window length (s) to bound attention memory
WAV_DIR = ROOT / "clean_wavs"
OUT_NPZ = PROC / "wavlm_embeddings.npz"


def load_audio(path):
    """Load -> mono, 16 kHz float32."""
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)  # librosa resamples + mixes down
    return y.astype(np.float32)


def embed_utterance(y, feat_extractor, model):
    """Return (n_layers, dim) length-weighted mean-pooled embedding for one clip."""
    chunk = CHUNK_S * TARGET_SR
    # split into <=20s windows; drop a final scrap shorter than 0.5s (nothing to pool)
    starts = list(range(0, len(y), chunk))
    acc = None                 # running sum over frames of per-frame hidden states
    total_frames = 0
    for s in starts:
        seg = y[s:s + chunk]
        if len(seg) < int(0.5 * TARGET_SR):
            continue
        inputs = feat_extractor(seg, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            out = model(inputs.input_values, output_hidden_states=True)
        # hidden_states: tuple of 13 tensors, each (1, T, 768)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]          # (13, T, 768)
        n_t = hs.shape[1]
        seg_sum = hs.sum(dim=1)                                    # (13, 768) sum over frames
        acc = seg_sum if acc is None else acc + seg_sum
        total_frames += n_t
    emb = (acc / total_frames).numpy().astype(np.float32)          # global mean-pool
    return emb, total_frames


def main():
    ensure_dirs(PROC)
    wavs = sorted(WAV_DIR.glob("*.wav"))
    print(f"[extract] {len(wavs)} wav files | model={MODEL_NAME} | device=cpu")

    print("[extract] loading frozen WavLM (first run downloads ~360MB from HF Hub)...")
    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = WavLMModel.from_pretrained(MODEL_NAME)
    model.eval()                                                  # frozen: no training
    for p in model.parameters():
        p.requires_grad_(False)

    file_ids, embs, nframes, durs = [], [], [], []
    t0 = time.time()
    for i, wav in enumerate(wavs, 1):
        y = load_audio(wav)
        emb, nf = embed_utterance(y, feat_extractor, model)
        file_ids.append(wav.stem)
        embs.append(emb)
        nframes.append(nf)
        durs.append(len(y) / TARGET_SR)
        print(f"  [{i:2d}/{len(wavs)}] {wav.stem:38s} "
              f"dur={len(y)/TARGET_SR:5.1f}s frames={nf:5d} "
              f"|emb|={np.linalg.norm(emb[-1]):6.2f}  ({time.time()-t0:5.1f}s)")

    embeddings = np.stack(embs).astype(np.float32)                # (n, 13, 768)
    np.savez_compressed(
        OUT_NPZ,
        embeddings=embeddings,
        file_ids=np.array(file_ids),
        n_frames=np.array(nframes),
        duration_s=np.array(durs, dtype=np.float32),
    )
    print(f"\n[extract] saved {embeddings.shape} -> {OUT_NPZ}")
    print(f"[extract] layers={embeddings.shape[1]} dim={embeddings.shape[2]} "
          f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
