import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os

# -------------------
# 1. Load wav2vec2
# -------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec.eval()          # inference mode
for p in wav2vec.parameters():
    p.requires_grad = False   # keep frozen

# -------------------
# 2. Simple model
# -------------------
class SimpleVoiceRegressor(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=256):
        super().__init__()
        # bottleneck (reduce 2*768 → 256 after mean+std)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        # two regression heads
        self.head_self = nn.Linear(bottleneck_dim, 1)
        self.head_perc = nn.Linear(bottleneck_dim, 1)

    def forward(self, hidden_states):
        # hidden_states: [B, T, D]
        mu = hidden_states.mean(dim=1)
        sd = hidden_states.std(dim=1)
        z = torch.cat([mu, sd], dim=-1)   # [B, 2D]

        z = self.bottleneck(z)            # [B, 256]
        y_self = self.head_self(z)        # [B, 1]
        y_perc = self.head_perc(z)        # [B, 1]
        return y_self, y_perc

model = SimpleVoiceRegressor()

# -------------------
# 3. Helper: load audio
# -------------------
def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform.squeeze(), 16000

# -------------------
# 4. Run test batch
# -------------------
folder = "your_audio_folder_here"   # replace with your path
files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]

batch = []
for f in files[:2]:   # just test 2 files
    waveform, sr = load_audio(f)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec(**inputs).last_hidden_state   # [1, T, D]
    batch.append(outputs)

# pad batch to same length for demo
batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

# Forward through model
y_self, y_perc = model(batch)
print("Pred self-ID:", y_self)
print("Pred perceived:", y_perc)
