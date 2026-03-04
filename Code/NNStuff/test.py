import os
import csv
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load wav2vec2 (frozen)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
wav2vec.eval()
for p in wav2vec.parameters():
    p.requires_grad = False

# 2) Simple model: mean+std -> bottleneck -> two heads
class SimpleVoiceRegressor(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=256):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.head_self = nn.Linear(bottleneck_dim, 1)
        self.head_perc = nn.Linear(bottleneck_dim, 1)

    def forward(self, hidden_states):  # [B, T, D]
        mu = hidden_states.mean(dim=1)             # [B, D]
        sd = hidden_states.std(dim=1)              # [B, D]
        z = torch.cat([mu, sd], dim=-1)            # [B, 2D]
        z = self.bottleneck(z)                     # [B, 256]
        return self.head_self(z), self.head_perc(z)

model = SimpleVoiceRegressor().to(DEVICE)

# 3) Audio loader (16kHz, mono)
def load_audio_16k(path):
    wav, sr = torchaudio.load(path)               # [C, N]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)       # mono
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0), 16000                  # [N], 16000

# 4) Batch processing helper
BATCH_SIZE = 8  # adjust to your VRAM

def masked_mean_std(hs, attn):  # hs:[B,T,D], attn:[B,T] with 1=real,0=pad
    mask = attn.unsqueeze(-1).to(hs.dtype)         # [B,T,1]
    lengths = mask.sum(dim=1).clamp_min(1.0)       # [B,1]
    mu = (hs * mask).sum(dim=1) / lengths          # [B,D]
    # variance = E[x^2] - (E[x])^2 using masks
    ex2 = ((hs**2) * mask).sum(dim=1) / lengths    # [B,D]
    var = (ex2 - mu**2).clamp_min(1e-12)
    sd = var.sqrt()
    return mu, sd

# 5) Collect all .wav files from folder
FOLDER = r"C:\Users\jinfa\Desktop\GayStudy\recodedWavs"  # <- set your folder
files = [os.path.join(FOLDER, f) for f in os.listdir(FOLDER) if f.lower().endswith(".wav")]
assert len(files) > 0, "No .wav files found in FOLDER"

N = len(files)
all_rows = []  # (filename, y_self, y_perc)

for i in range(0, N, BATCH_SIZE):
    batch_paths = files[i:i+BATCH_SIZE]
    waveforms = []
    for p in batch_paths:
        w, _ = load_audio_16k(p)
        waveforms.append(w.detach().cpu().float().numpy())  # HF processor likes numpy

    inputs = processor(
        waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to(DEVICE)

    with torch.no_grad():
        out = wav2vec(**inputs, output_attentions=False, return_dict=True)
        hs = out.last_hidden_state                # [B,T,768]
        # Wav2vec outputs its own attention mask for the downsampled sequence
        if hasattr(out, 'extract_features') or 'attention_mask' not in out:
            # No output mask, just use simple mean/std
            mu = hs.mean(dim=1)
            sd = hs.std(dim=1)
        else:
            mu, sd = masked_mean_std(hs, out.attention_mask)
        z = torch.cat([mu, sd], dim=-1)           # [B,1536]
        z = model.bottleneck(z)                   # [B,256]
        y_self = model.head_self(z).squeeze(1).cpu().tolist()
        y_perc = model.head_perc(z).squeeze(1).cpu().tolist()

    for p, ys, yp in zip(batch_paths, y_self, y_perc):
        all_rows.append((str(p), ys, yp))

print(f"Processed {len(all_rows)} files.")
for r in all_rows[:5]:
    print(r)

# 6) Save to CSV
with open("predictions_demo.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filepath", "y_self_raw", "y_perc_raw"])
    w.writerows(all_rows)
print("Saved predictions_demo.csv")
