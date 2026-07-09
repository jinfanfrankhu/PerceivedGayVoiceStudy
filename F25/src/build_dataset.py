"""build_dataset.py  --  raw -> processed.  Run this FIRST.

Produces three tidy tables in data/processed/:

  crosswalk.csv       one row per speaker: file_id <-> Initials <-> Pseudonym.
                      The single canonical mapping; everything joins through it.
  speakers.csv        one row per speaker: crosswalk + metadata + 88 acoustic
                      features + perceived-rating aggregates (mean/sd/dip/...).
  ratings_clean.csv   one row per listener rating, joined to canonical ids and
                      the speaker's true Kinsey / self-ID (for mixed models).

Canonicalization note
---------------------
The Master Spreadsheet is the source of truth for speaker names. Three pseudonyms
were originally typed differently in the listener platform; since no further
exports are expected, those were corrected directly in data/raw/ratings.csv. The
check_join() guard below still fails loudly if any rated speaker fails to match
Master, in case a name ever drifts again.
"""
import sys
import numpy as np
import pandas as pd

from common import (MASTER_CSV, RATINGS_CSV, FEATURES_CSV,
                    CROSSWALK_CSV, SPEAKERS_CSV, RATINGS_CLEAN_CSV,
                    PROC, ensure_dirs)

# A speaker is "split" when both opposing camps are substantial. Automated
# unimodality tests (bimodality coefficient, Hartigan dip test) are unreliable
# on 5-point rating data -- both over-flag heavily -- so we measure the two
# camps directly instead. CAMP_MIN is the reporting cutoff for "substantial".
CAMP_MIN = 0.25

META_COLS = [
    'Kinsey Scale (1-5)', 'Self-Described Sexual Orientation', 'Gender ID',
    'Race', 'Birth Location', 'Upbringing Location', 'Year of Birth',
    'First Language', 'Years of English Spoken',
]


def load_master():
    m = pd.read_csv(MASTER_CSV)
    m['file_id'] = m['Pseudonym'].str.strip().str.replace(' ', '_', regex=False)
    return m


def load_ratings():
    return pd.read_csv(RATINGS_CSV)


def build_crosswalk(master):
    cw = master[['file_id', 'Initials', 'Pseudonym']].copy()
    cw = cw.rename(columns={'Initials': 'initials', 'Pseudonym': 'pseudonym'})
    dupe_i = cw['initials'].duplicated().sum()
    dupe_f = cw['file_id'].duplicated().sum()
    if dupe_i or dupe_f:
        print(f'  WARNING: duplicate keys in crosswalk (initials={dupe_i}, file_id={dupe_f})')
    return cw


def check_join(ratings, master):
    r_ids = set(ratings['file_id'].unique())
    m_ids = set(master['file_id'].unique())
    missing = sorted(r_ids - m_ids)
    if missing:
        print('  ERROR: ratings file_ids with no match in Master after rename:')
        for x in missing:
            print('   ', x)
        sys.exit(1)
    print(f'  join OK: all {len(r_ids)} rated speakers matched to Master '
          f'({len(m_ids)} speakers total)')


def perceived_aggregates(ratings):
    """Per-speaker summary of the 1-5 listener ratings."""
    rows = []
    for fid, g in ratings.groupby('file_id'):
        x = g['rating'].to_numpy()
        counts = {f'rated_{k}': int((x == k).sum()) for k in range(1, 6)}
        frac_low = float(np.mean(x <= 2))      # "sounds straight" camp
        frac_high = float(np.mean(x >= 4))     # "sounds gay" camp
        rows.append({
            'file_id': fid,
            'n_ratings': len(x),
            'perceived_mean': x.mean(),
            'perceived_median': float(np.median(x)),
            'perceived_sd': x.std(ddof=1),
            'frac_low': frac_low,
            'frac_high': frac_high,
            # 0 = one-sided, 1 = evenly split between the two camps
            'polarization': 2 * min(frac_low, frac_high),
            'is_split': bool(frac_low >= CAMP_MIN and frac_high >= CAMP_MIN),
            **counts,
        })
    return pd.DataFrame(rows)


def main():
    ensure_dirs(PROC)
    print('build_dataset.py')

    master = load_master()
    ratings = load_ratings()
    features = pd.read_csv(FEATURES_CSV)          # 'ID' column == initials

    check_join(ratings, master)

    crosswalk = build_crosswalk(master)
    crosswalk.to_csv(CROSSWALK_CSV, index=False)
    print(f'  wrote {CROSSWALK_CSV.name}  ({len(crosswalk)} speakers)')

    # --- speakers.csv : one row per speaker ---------------------------------
    meta = master[['file_id', 'Initials'] + META_COLS].rename(
        columns={'Initials': 'initials'})
    perc = perceived_aggregates(ratings)

    speakers = (meta
                .merge(features, left_on='initials', right_on='ID', how='left')
                .drop(columns=['ID'])
                .merge(perc, on='file_id', how='left'))

    n_no_feats = speakers['F0semitoneFrom27.5Hz_sma3nz_amean'].isna().sum() \
        if 'F0semitoneFrom27.5Hz_sma3nz_amean' in speakers else -1
    n_no_ratings = speakers['n_ratings'].isna().sum()
    speakers.to_csv(SPEAKERS_CSV, index=False)
    print(f'  wrote {SPEAKERS_CSV.name}  ({len(speakers)} speakers, '
          f'{speakers.shape[1]} cols; {n_no_feats} missing features, '
          f'{int(n_no_ratings)} missing ratings)')

    # --- ratings_clean.csv : one row per rating ----------------------------
    truth = master[['file_id', 'Initials', 'Kinsey Scale (1-5)',
                    'Self-Described Sexual Orientation']].rename(
        columns={'Initials': 'initials',
                 'Kinsey Scale (1-5)': 'true_kinsey',
                 'Self-Described Sexual Orientation': 'self_orientation'})
    ratings_clean = ratings.merge(truth, on='file_id', how='left')
    ratings_clean.to_csv(RATINGS_CLEAN_CSV, index=False)
    print(f'  wrote {RATINGS_CLEAN_CSV.name}  ({len(ratings_clean)} ratings)')


if __name__ == '__main__':
    main()
