"""segmental_census.py  --  how many usable tokens of each phone do we actually
have, per speaker?  Run after MFA alignment, BEFORE writing extract_segmental.py.

Reads the MFA phones tier for all 50 speakers and tallies, per phone type, how many
tokens each speaker produced. For vowels, "usable" = stress-bearing (digit 1 or 2),
since reduced vowels (digit 0) won't be measured (decision C5). For consonants, all
tokens count. The point is to set the minimum-token filter (decision C8) from the
observed distribution instead of guessing, and to see which phones are too rare to
measure reliably.

Outputs:
  outputs/tables/phone_census.csv           one row per phone (counts + per-speaker stats)
  outputs/figures/segmental/phone_census.png bar chart, median tokens/speaker per phone
"""
import matplotlib
matplotlib.use('Agg')
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, TABLES, FIG, ensure_dirs

TG_DIR = ROOT / 'mfa_textgrids'
SEG_FIG = FIG / 'segmental'

VOWELS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
          'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
KLASS = {
    **{p: 'vowel' for p in VOWELS},
    **{p: 'fricative' for p in ['F', 'V', 'TH', 'DH', 'S', 'SH', 'Z', 'ZH', 'HH']},
    **{p: 'stop' for p in ['P', 'B', 'T', 'D', 'K', 'G']},
    **{p: 'affricate' for p in ['CH', 'JH']},
    **{p: 'nasal' for p in ['M', 'N', 'NG']},
    **{p: 'approx' for p in ['L', 'R', 'W', 'Y']},
}
KLASS_COLOR = {'vowel': '#3a6ea5', 'fricative': '#c1553b', 'stop': '#7aa66b',
               'affricate': '#a67ba6', 'nasal': '#b8875b', 'approx': '#8899aa'}

# target phones for our confirmatory analysis, for annotation
TARGETS = {'S', 'SH', 'Z', 'AY', 'EY', 'OW', 'AW', 'IY', 'UW', 'AE', 'AA'}

# ARPABET (stress-stripped) -> IPA, for an alternate-labelled chart
ARPA2IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ', 'IY': 'i', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 'G': 'ɡ', 'HH': 'h',
    'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p',
    'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v', 'W': 'w',
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}


def make_chart(df, use_ipa, path):
    """Horizontal bar chart of median tokens/speaker per phone; ARPABET or IPA labels."""
    d = df.sort_values('median_per_spk')
    y = np.arange(len(d))
    colors = [KLASS_COLOR.get(k, '#cccccc') for k in d['klass']]
    fig, ax = plt.subplots(figsize=(11, 13))
    ax.barh(y, d['median_per_spk'], color=colors)
    ax.plot(d['min_per_spk'], y, 'k|', ms=8, label='min across speakers')
    for thr in (3, 5):
        ax.axvline(thr, color='#888', ls='--', lw=1)
        ax.text(thr, len(d) - 0.5, f'  ≥{thr}', fontsize=8, color='#555')
    if use_ipa:
        labels = [f'/{ARPA2IPA.get(p, p)}/{" *" if t else ""}'
                  for p, t in zip(d['phone'], d['is_target'])]
    else:
        labels = [f'{p}{" *" if t else ""}' for p, t in zip(d['phone'], d['is_target'])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('usable tokens per speaker  (bar = median, | = min)')
    scheme = 'IPA' if use_ipa else 'ARPABET'
    ax.set_title(f'Phone availability across 50 speakers  [{scheme} labels]\n'
                 '(* = confirmatory target; vowels stress-bearing only; '
                 'colour = manner class)', fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KLASS_COLOR.values()]
    ax.legend(handles, KLASS_COLOR.keys(), fontsize=8, loc='lower right',
              title='manner')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_phones(tg_path):
    """Return list of (base_phone, is_usable) for the phones tier."""
    t = tg_path.read_text(encoding='utf-8')
    block = [x for x in t.split('item [') if re.search(r'name = "phones"', x)][0]
    ivs = re.findall(r'text = "([^"]*)"', block)
    out = []
    for lab in ivs:
        lab = lab.strip()
        if not lab:
            continue
        m = re.match(r'^([A-Z]+)(\d?)$', lab)
        if not m:
            continue
        base, stress = m.group(1), m.group(2)
        if base in VOWELS:
            usable = stress in ('1', '2')          # exclude reduced (0)
        else:
            usable = True
        out.append((base, usable))
    return out


def main():
    ensure_dirs(TABLES, SEG_FIG)
    print('segmental_census.py')
    tgs = sorted(TG_DIR.glob('*/*.TextGrid'))
    print(f'  {len(tgs)} TextGrids')

    # per-speaker usable counts: {phone: [count per speaker]}
    speakers = []
    counts = {}          # phone -> dict(speaker -> usable count)
    for tg in tgs:
        spk = tg.stem
        speakers.append(spk)
        for base, usable in parse_phones(tg):
            counts.setdefault(base, {}).setdefault(spk, 0)
            if usable:
                counts[base][spk] += 1

    rows = []
    for phone, per_spk in counts.items():
        vals = np.array([per_spk.get(s, 0) for s in speakers])
        rows.append({
            'phone': phone,
            'klass': KLASS.get(phone, 'other'),
            'is_target': phone in TARGETS,
            'total_usable': int(vals.sum()),
            'median_per_spk': float(np.median(vals)),
            'min_per_spk': int(vals.min()),
            'p25_per_spk': float(np.percentile(vals, 25)),
            'n_spk_ge1': int((vals >= 1).sum()),
            'n_spk_ge3': int((vals >= 3).sum()),
            'n_spk_ge5': int((vals >= 5).sum()),
        })
    df = pd.DataFrame(rows).sort_values('median_per_spk', ascending=False)
    df.to_csv(TABLES / 'phone_census.csv', index=False)

    # --- charts: median tokens/speaker per phone, min marked (ARPABET + IPA) ---
    make_chart(df, use_ipa=False, path=SEG_FIG / 'phone_census_arpa.png')
    make_chart(df, use_ipa=True, path=SEG_FIG / 'phone_census_ipa.png')

    # --- console summary for the targets ---
    print('  target phones (usable tokens/speaker):')
    tt = df[df.is_target].sort_values('median_per_spk', ascending=False)
    for _, r in tt.iterrows():
        print(f'    {r.phone:3} {r.klass:9} median={r.median_per_spk:4.1f} '
              f'min={r.min_per_spk:2d} total={r.total_usable:4d} '
              f'(>=3:{r.n_spk_ge3}/50  >=5:{r.n_spk_ge5}/50)')
    print('  -> tables/phone_census.csv  + figures/segmental/'
          'phone_census_{arpa,ipa}.png')


if __name__ == '__main__':
    main()
