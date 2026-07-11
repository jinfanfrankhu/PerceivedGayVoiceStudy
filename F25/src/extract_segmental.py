"""extract_segmental.py  --  phone-level acoustic measurement.  Run after MFA.

Reads MFA phone boundaries (mfa_textgrids/) and measures on the ORIGINAL high-rate
audio (clean_wavs/, 44.1/96 kHz) so sibilant energy above 8 kHz survives (decision B1).

Measures, by manner class (decisions C1-C13):
  Fricatives (S SH Z F V DH HH): spectral moments (COG, SD, skew, kurtosis) on the
    middle 50%, high-passed at 750 Hz, + peak frequency + duration.
  Vowels (12 stressed): F1/F2/F3 at 20/50/80% (Burg LPC), duration, and a breathiness
    block (CPPS + H1-H2) for the voice-quality target.

Emits two tables in data/processed/:
  segmental_tokens.csv    one row per measured token (long; exploration + diagnostics)
  segmental_speaker.csv   one row per speaker (wide; analysis-ready), N>=5 filter,
                          Lobanov-normalized vowel geometry + diphthong dynamics.

Everything is guarded: a token that fails measurement or a plausibility check is
recorded with a drop reason rather than crashing, so the per-speaker n_dropped
diagnostic (decision C10) stays meaningful.
"""
import re
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

from common import ROOT, PROC, ensure_dirs

TG_DIR = ROOT / 'mfa_textgrids'
WAV_DIR = ROOT / 'clean_wavs'

# ---- phone inventories (decision C8: N>=5 survivors) ----
VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
          'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
MEASURE_VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'EY', 'IH', 'IY', 'OW', 'UW'}
FRICATIVES = ['S', 'SH', 'Z', 'F', 'V', 'DH', 'HH']
DIPHTHONGS = {'AY', 'EY', 'OW', 'AW'}
CORNERS = ['IY', 'AE', 'AA', 'UW']          # polygon order for vowel-space area
FRONT = {'IY', 'IH', 'EY', 'EH', 'AE'}
CONSONANTS = set('P B T D K G CH JH M N NG L R W Y F V TH DH S SH Z ZH HH'.split())

N_MIN = 5                                    # decision C8
HP_CUT = 750                                 # decision C13
# formant settings (decision C12)
FMT = dict(time_step=0.01, max_number_of_formants=5,
           maximum_formant=5000, window_length=0.025, pre_emphasis_from=50)
PITCH_FLOOR, PITCH_CEIL = 60, 350

# plausibility ranges (decision C10-adjacent)
FRIC_DUR = (0.03, 0.40)
FRIC_COG = (1000, 12000)
VOW_DUR = (0.03, 0.40)
F1_RNG, F2_RNG, F3_RNG = (200, 1200), (700, 3000), (1200, 4000)


# ---------------------------------------------------------------------------
def parse_phones(tg_path):
    """Ordered list of dicts for the phones tier (incl. empty=silence markers)."""
    t = tg_path.read_text(encoding='utf-8')
    block = [x for x in t.split('item [') if re.search(r'name = "phones"', x)][0]
    ivs = re.findall(r'xmin = ([0-9.]+)\s+xmax = ([0-9.]+)\s+text = "([^"]*)"', block)
    seq = []
    for a, b, lab in ivs:
        lab = lab.strip()
        m = re.match(r'^([A-Z]+)(\d?)$', lab) if lab else None
        base = m.group(1) if m else None
        stress = m.group(2) if m else ''
        seq.append({'label': lab, 'base': base, 'stress': stress,
                    'start': float(a), 'end': float(b)})
    return seq


def measure_fricative(snd, start, end):
    m0 = start + 0.25 * (end - start)
    m1 = start + 0.75 * (end - start)
    part = snd.extract_part(from_time=m0, to_time=m1,
                            window_shape=parselmouth.WindowShape.HANNING)
    nyq = snd.sampling_frequency / 2
    part = call(part, 'Filter (pass Hann band)', HP_CUT, nyq, 100)
    spec = part.to_spectrum()
    cog = spec.get_centre_of_gravity(power=2)
    sd = spec.get_standard_deviation(power=2)
    skew = spec.get_skewness(power=2)
    kurt = spec.get_kurtosis(power=2)
    v = spec.values
    power = v[0] ** 2 + v[1] ** 2
    freqs = np.arange(power.shape[0]) * spec.get_bin_width()
    peak = float(freqs[np.argmax(power)])
    return {'cog': cog, 'sd': sd, 'skew': skew, 'kurt': kurt, 'peak': peak}


def measure_vowel_formants(formant, start, end):
    out = {}
    for pct, tag in ((0.2, '20'), (0.5, '50'), (0.8, '80')):
        t = start + pct * (end - start)
        out[f'f1_{tag}'] = formant.get_value_at_time(1, t)
        out[f'f2_{tag}'] = formant.get_value_at_time(2, t)
        out[f'f3_{tag}'] = formant.get_value_at_time(3, t)
    return out


def measure_breathiness(snd, pitch, start, end):
    """CPPS over the whole vowel + uncorrected H1-H2 on the middle 50%."""
    res = {'f0': np.nan, 'cpps': np.nan, 'h1h2': np.nan}
    try:
        part = snd.extract_part(from_time=start, to_time=end)
        pcg = call(part, 'To PowerCepstrogram', 60, 0.002, 5000, 50)
        res['cpps'] = call(pcg, 'Get CPPS', 'yes', 0.02, 0.0005, 60, 330, 0.05,
                           'parabolic', 0.001, 0, 'Straight', 'Robust')
    except Exception:
        pass
    try:
        tm = 0.5 * (start + end)
        f0 = pitch.get_value_at_time(tm)
        res['f0'] = f0
        if f0 and np.isfinite(f0):
            m0 = start + 0.25 * (end - start)
            m1 = start + 0.75 * (end - start)
            spec = snd.extract_part(from_time=m0, to_time=m1,
                                    window_shape=parselmouth.WindowShape.HANNING).to_spectrum()
            v = spec.values
            db = 10 * np.log10(v[0] ** 2 + v[1] ** 2 + 1e-12)
            binw = spec.get_bin_width()

            def amp_near(fc, tol=0.15):
                lo = max(0, int(fc * (1 - tol) / binw))
                hi = min(len(db), int(fc * (1 + tol) / binw) + 1)
                return db[lo:hi].max() if hi > lo else np.nan
            res['h1h2'] = amp_near(f0) - amp_near(2 * f0)
    except Exception:
        pass
    return res


# ---------------------------------------------------------------------------
def process_speaker(file_id, wav_path, tg_path):
    snd = parselmouth.Sound(str(wav_path))
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    formant = snd.to_formant_burg(**FMT)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=PITCH_FLOOR,
                         pitch_ceiling=PITCH_CEIL)
    seq = parse_phones(tg_path)

    rows = []
    for i, ph in enumerate(seq):
        base = ph['base']
        if base is None:
            continue
        dur = ph['end'] - ph['start']
        row = {'file_id': file_id, 'phone': ph['label'], 'base': base,
               'stress': ph['stress'], 'start': ph['start'], 'end': ph['end'],
               'dur': dur, 'dropped': False, 'drop_reason': ''}

        if base in FRICATIVES:
            row['klass'] = 'fricative'
            prev_b = seq[i - 1]['base'] if i > 0 else None
            next_b = seq[i + 1]['base'] if i + 1 < len(seq) else None
            row['is_singleton'] = (prev_b not in CONSONANTS) and (next_b not in CONSONANTS)
            try:
                row.update(measure_fricative(snd, ph['start'], ph['end']))
            except Exception as e:
                row['dropped'] = True
                row['drop_reason'] = f'fric_err:{type(e).__name__}'
            if not row['dropped']:
                if not (FRIC_DUR[0] <= dur <= FRIC_DUR[1]):
                    row['dropped'] = True; row['drop_reason'] = 'dur'
                elif not (FRIC_COG[0] <= row.get('cog', np.nan) <= FRIC_COG[1]):
                    row['dropped'] = True; row['drop_reason'] = 'cog'
            rows.append(row)

        elif base in MEASURE_VOWELS and ph['stress'] in ('1', '2'):
            row['klass'] = 'vowel'
            try:
                row.update(measure_vowel_formants(formant, ph['start'], ph['end']))
                row.update(measure_breathiness(snd, pitch, ph['start'], ph['end']))
            except Exception as e:
                row['dropped'] = True
                row['drop_reason'] = f'vow_err:{type(e).__name__}'
            # plausibility on the midpoint formants
            f1, f2, f3 = row.get('f1_50'), row.get('f2_50'), row.get('f3_50')
            ok = all(v is not None and np.isfinite(v) for v in (f1, f2, f3))
            if not (VOW_DUR[0] <= dur <= VOW_DUR[1]):
                row['dropped'] = True; row['drop_reason'] = 'dur'
            elif not ok:
                row['dropped'] = True; row['drop_reason'] = 'formant_nan'
            elif not (F1_RNG[0] <= f1 <= F1_RNG[1] and F2_RNG[0] <= f2 <= F2_RNG[1]
                      and F3_RNG[0] <= f3 <= F3_RNG[1] and f1 < f2 < f3):
                row['dropped'] = True; row['drop_reason'] = 'formant_range'
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _mean_ge_n(s):
    """Mean if at least N_MIN non-null values, else NaN (decision C8)."""
    s = s.dropna()
    return s.mean() if len(s) >= N_MIN else np.nan


def aggregate(tokens):
    """Token table -> one row per speaker (wide)."""
    good = tokens[~tokens['dropped']].copy()
    out = []
    for fid, g in good.groupby('file_id'):
        r = {'file_id': fid}
        fric = g[g['klass'] == 'fricative']
        vow = g[g['klass'] == 'vowel'].copy()

        # --- fricatives: mean of each moment per phone (N>=5) ---
        for ph in FRICATIVES:
            sub = fric[fric['base'] == ph]
            for meas in ('cog', 'sd', 'skew', 'kurt', 'peak', 'dur'):
                r[f'{ph}_{meas}'] = _mean_ge_n(sub[meas]) if len(sub) else np.nan
        # singleton-only /s/ COG (robustness, decision C3)
        s_sing = fric[(fric['base'] == 'S') & (fric['is_singleton'] == True)]
        r['S_cog_singleton'] = _mean_ge_n(s_sing['cog']) if len(s_sing) else np.nan

        # --- breathiness across all stressed vowels ---
        r['v_cpps'] = _mean_ge_n(vow['cpps'])
        r['v_h1h2'] = _mean_ge_n(vow['h1h2'])

        # --- Lobanov normalization of this speaker's vowel formants (C7) ---
        f1m, f1s = vow['f1_50'].mean(), vow['f1_50'].std(ddof=1)
        f2m, f2s = vow['f2_50'].mean(), vow['f2_50'].std(ddof=1)
        if np.isfinite(f1s) and f1s > 0 and np.isfinite(f2s) and f2s > 0:
            for tag in ('20', '50', '80'):
                vow[f'z1_{tag}'] = (vow[f'f1_{tag}'] - f1m) / f1s
                vow[f'z2_{tag}'] = (vow[f'f2_{tag}'] - f2m) / f2s

            # per-vowel normalized midpoint means (broad/exploratory)
            for vw in MEASURE_VOWELS:
                sub = vow[vow['base'] == vw]
                r[f'{vw}_z1'] = _mean_ge_n(sub['z1_50']) if len(sub) else np.nan
                r[f'{vw}_z2'] = _mean_ge_n(sub['z2_50']) if len(sub) else np.nan

            # vowel-space area (shoelace on corner-vowel normalized means)
            pts = []
            for vw in CORNERS:
                sub = vow[vow['base'] == vw]
                pts.append((_mean_ge_n(sub['z2_50']), _mean_ge_n(sub['z1_50'])))  # x=F2,y=F1
            if all(np.isfinite(x) and np.isfinite(y) for x, y in pts):
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i]
                                     for i in range(4)))
                r['vowel_space_area'] = area
            else:
                r['vowel_space_area'] = np.nan

            # front-vowel fronting = mean normalized F2 of front vowels
            r['front_f2'] = _mean_ge_n(vow[vow['base'].isin(FRONT)]['z2_50'])

            # diphthong dynamics: trajectory length + rate (per diphthong)
            for dp in DIPHTHONGS:
                sub = vow[vow['base'] == dp].dropna(subset=['z1_20', 'z1_80'])
                if len(sub) >= N_MIN:
                    traj = np.sqrt((sub['z1_80'] - sub['z1_20']) ** 2 +
                                   (sub['z2_80'] - sub['z2_20']) ** 2)
                    r[f'{dp}_trajlen'] = traj.mean()
                    r[f'{dp}_rate'] = (traj / sub['dur']).mean()
                else:
                    r[f'{dp}_trajlen'] = np.nan
                    r[f'{dp}_rate'] = np.nan
        out.append(r)
    return pd.DataFrame(out)


def main():
    ensure_dirs(PROC)
    print('extract_segmental.py')
    tgs = sorted(TG_DIR.glob('*/*.TextGrid'))
    all_tokens = []
    for tg in tgs:
        fid = tg.stem
        wav = WAV_DIR / f'{fid}.wav'
        if not wav.exists():
            print(f'  WARNING: no wav for {fid}')
            continue
        df = process_speaker(fid, wav, tg)
        all_tokens.append(df)
        nd = int(df['dropped'].sum())
        print(f'  {fid[:28]:28} tokens={len(df):3d}  dropped={nd:2d}')
    tokens = pd.concat(all_tokens, ignore_index=True)
    tokens.to_csv(PROC / 'segmental_tokens.csv', index=False)

    speaker = aggregate(tokens)
    speaker.to_csv(PROC / 'segmental_speaker.csv', index=False)

    nd = int(tokens['dropped'].sum())
    print(f'  --> segmental_tokens.csv ({len(tokens)} tokens, {nd} dropped '
          f'[{100*nd/len(tokens):.1f}%])')
    print(f'  --> segmental_speaker.csv ({len(speaker)} speakers, '
          f'{speaker.shape[1]} cols)')
    # worst offenders on the diagnostic (decision C10)
    per = tokens.groupby('file_id')['dropped'].agg(['size', 'sum'])
    per['rate'] = per['sum'] / per['size']
    worst = per.sort_values('rate', ascending=False).head(5)
    print('  highest drop rates:')
    for fid, row in worst.iterrows():
        print(f'    {fid[:28]:28} {int(row["sum"])}/{int(row["size"])} '
              f'({row["rate"]:.0%})')
    print('done.')


if __name__ == '__main__':
    main()
