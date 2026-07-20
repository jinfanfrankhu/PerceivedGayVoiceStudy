"""Does the CURRENT env reproduce the cached wavlm_embeddings.npz?

10c/10d re-run WavLM and assume the per-frame states match the ones that were
mean-pooled into the npz. If transformers/torch versions have moved since the npz
was written, that assumption breaks silently. This checks 2 speakers.

Run:  python src/check_npz_consistency.py
"""
import sys
from pathlib import Path

import numpy as np
import librosa
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent.parent          # F25/ — cwd-independent
sys.path.insert(0, str(ROOT / "src"))

MODEL_NAME = "microsoft/wavlm-base-plus"
TARGET_SR = 16_000
CHUNK_S = 20
N_CHECK = 2

d = np.load(ROOT / "data/processed/wavlm_embeddings.npz", allow_pickle=True)
E, ids = d["embeddings"], list(d["file_ids"])
print(f"npz: {E.shape}  ({len(ids)} speakers)")

print("loading WavLM (first run downloads ~360MB)...")
feat = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)


def embed(y):
    """Reproduces 10_wavlm_extract.embed_utterance exactly."""
    chunk = CHUNK_S * TARGET_SR
    acc, total = None, 0
    for s in range(0, len(y), chunk):
        seg = y[s:s + chunk]
        if len(seg) < int(0.5 * TARGET_SR):
            continue
        inp = feat(seg, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            out = model(inp.input_values, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]
        acc = hs.sum(dim=1) if acc is None else acc + hs.sum(dim=1)
        total += hs.shape[1]
    return (acc / total).numpy().astype(np.float32)


ok = True
for fid in ids[:N_CHECK]:
    y = librosa.load(ROOT / "clean_wavs" / f"{fid}.wav", sr=TARGET_SR, mono=True)[0]
    new = embed(y.astype(np.float32))
    old = E[ids.index(fid)]
    absdiff = np.abs(new - old)
    rel = absdiff.max() / (np.abs(old).max() + 1e-12)
    match = np.allclose(new, old, rtol=1e-4, atol=1e-5)
    ok &= match
    print(f"  {fid[:34]:34}  max|diff|={absdiff.max():.3e}  rel={rel:.3e}  "
          f"{'MATCH' if match else 'MISMATCH'}")

print()
if ok:
    print("PASS - current env reproduces the npz. 10c/10d attributions are consistent")
    print("       with the 10b prediction results. Safe to proceed.")
else:
    print("FAIL - the current env does NOT reproduce the cached embeddings.")
    print("       Regenerate before trusting anything downstream:")
    print("         python src/10_wavlm_extract.py     (~3 min)")
    print("         python src/10b_wavlm_probe.py      (re-derives best layer + rho)")
    print("       The published rho=0.73 / layer 6 may shift.")
