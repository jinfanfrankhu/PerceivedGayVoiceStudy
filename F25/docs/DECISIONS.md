# F25 Methodological Decision Record

Living record of *why* each analysis choice was made — the justification behind the
numbers, so the reasoning survives past memory and can seed the paper's methods
section. Format per entry: **Decision · Rationale · Alternatives considered · Status.**

Status legend: **SETTLED** (in use) · **PENDING** (awaiting data/decision) ·
**REVISED** (superseded, kept for the paper trail).

---

## A. Statistics & inference

### A1. Spearman as the default correlation
- **Decision:** Rank (Spearman) correlation is primary; Pearson reported alongside.
- **Rationale:** Kinsey is ordinal, not interval; and only ~40/88 eGeMAPS features
  pass Shapiro-Wilk normality, so a rank method is the safer default. Spearman asks
  "do they move together monotonically," which is the question we actually have.
- **Alternatives:** Pearson (assumes linear + roughly normal — violated here).
- **Status:** SETTLED.

### A2. Multiple comparisons → Benjamini-Hochberg FDR, within-target
- **Decision:** BH-FDR at q=0.10, applied *separately within* each target family
  (actual-Kinsey vs perceived), never pooled.
- **Rationale:** With 88 tests, ~4–5 cross p<.05 by chance alone. BH controls the
  *fraction* of discoveries that are false (q=0.10 ≈ "≈1 in 10 hits is a fluke") —
  the right promise for exploratory work. Bonferroni controls "zero mistakes ever,"
  which is far too strict at n=50 and kills all signal. Actual and perceived are
  corrected in isolation because the *divergence between them* is the research
  question — pooling would blur it.
- **Caveat:** BH's clean guarantee assumes independence; the 88 features are
  correlated, so we treat BH as a principled filter, not a hard guarantee.
- **Status:** SETTLED.

### A3. Lead with effect size, not p-values
- **Decision:** Tables lead with rho + bootstrap 95% CI; p/q are support.
- **Rationale:** n=50 is underpowered. A non-significant feature means "couldn't
  confirm," NOT "no relationship" — reporting it as null would be wrong. The effect
  size and its CI carry the honest information.
- **Status:** SETTLED.

### A4. Bimodality of listener ratings → direct camp measures
- **Decision:** Measure listener split via `frac_low`=P(rating≤2), `frac_high`=
  P(rating≥4), `polarization`=2·min(low,high), `is_split`=both camps ≥0.25.
- **Rationale:** The bimodality *coefficient* and Hartigan's dip test both over-flag
  on 5-point discrete rating data (dip flagged 47/50, including strictly-decreasing
  distributions) because they assume continuous data. Direct camp proportions are
  interpretable and honest for ordinal ratings.
- **Alternatives:** bimodality coefficient, dip test — both REVISED out.
- **Status:** SETTLED.

### A5. Confirmatory vs exploratory kept structurally separate
- **Decision:** Pre-registered, directional, theory-driven tests are the
  confirmatory backbone; broad atheoretical feature sweeps live in a separate
  exploratory/classifier track and are never reported as confirmed.
- **Rationale:** The multiple-comparisons penalty only bites significance testing.
  A classifier judged by honest LOOCV + permutation null doesn't test one hypothesis
  per feature, so a rich feature set is an asset there, not a liability. Mixing the
  two would either dredge or destroy power.
- **Status:** SETTLED.

### A6. Data augmentation is a Phase-3 classifier technique only
- **Decision:** Volume/noise augmentation may expand *classifier training* data;
  it must NEVER enter inference/correlation, and requires speaker-grouped CV splits.
- **Rationale:** Augmentation makes copies of the same speaker. In inference that is
  pseudoreplication — it shrinks p-values by faking independent evidence (n stays 50
  speakers). In a classifier it's valid *only* if all copies of a speaker stay on the
  same side of every fold, else the model sees the test answer at train time.
- **Status:** SETTLED (deferred to Phase 3).

---

## B. Forced alignment (MFA)

### B1. Separate 16 kHz mono corpus for alignment; measure on original audio
- **Decision:** Align a purpose-built `mfa_corpus/` (mono, 16 kHz, 16-bit, one
  speaker per subdirectory). Take acoustic *measurements* on the original
  `clean_wavs/` (44.1/96 kHz).
- **Rationale:** Alignment doesn't need high frequencies, and 16 kHz mono sidesteps
  the stereo/mixed-rate feature-generation error MFA threw on the raw files.
  Measurement *does* need the high end — /s/ spectral energy extends well above
  8 kHz, and 16 kHz audio (8 kHz Nyquist) would truncate exactly the band that
  distinguishes sibilants. TextGrid boundaries are in seconds, so they transfer to
  the original audio unchanged.
- **Status:** SETTLED.

### B2. One speaker per subdirectory
- **Decision:** Each of the 50 speakers gets its own corpus subdirectory.
- **Rationale:** A flat folder made MFA treat all 50 as a single speaker, disabling
  per-speaker acoustic adaptation. Subdirectories → 50 speakers → better boundaries.
- **Status:** SETTLED.

### B3. Whole-file alignment (no manual utterance segmentation)
- **Decision:** Align each ~50–60 s passage as one utterance.
- **Rationale:** MFA's ~30 s "ideal" is a soft performance guideline aimed at
  multi-minute recordings. Sub-minute clean read speech has strong acoustic anchors
  throughout; alignment errors from deviations are *local and self-recovering* (it
  re-syncs at the next distinctive word), so one flub doesn't derail the file.
  Spot-check confirmed ~95% read accuracy — well within tolerance.
- **Alternatives:** two-pass sentence segmentation (held in reserve if a speaker's
  spot-check shows drift).
- **Status:** SETTLED.

---

## C. Segmental acoustic measurement

### C1. Sibilant descriptor set = COG, peak frequency, skew, duration
- **Decision:** Describe /s/ (and /ʃ/) by spectral moments + duration.
- **Rationale:** Sibilants are *noise* (no formants), so we describe the shape of the
  spectrum. COG (energy balance point) is the primary gay-voice correlate — a fronted
  tongue constriction shrinks the front cavity and raises resonance, giving the
  "sharper" /s/. Peak = the modal frequency (robust to low-freq junk). Skew = spectral
  asymmetry (fronted /s/ → negative skew). Duration links to the "careful
  articulation" stereotype. Standard set: Munson et al. 2006; Linville 1998.
- **Status:** SETTLED.

### C2. Measure the middle 50% of a fricative
- **Decision:** Compute sibilant moments over the central 50% of the interval.
- **Rationale:** The onset/offset are transitions — tongue moving in/out, neighboring
  sounds bleeding in (coarticulation). The middle is the steady-state frication, the
  most representative and least contaminated portion.
- **Status:** SETTLED.

### C3. Measure all /s/ tokens, flag singleton vs cluster
- **Decision:** Include every /s/; record whether it's singleton (*so, seat*) or in a
  cluster (*sky, store*). Primary analysis on all; robustness re-run on singletons.
- **Rationale:** Singletons are phonetically cleaner, but this passage has more cluster
  /s/, and dropping them shrinks the per-speaker token count → noisier means at n=50.
  Because everyone reads the *same* words, the singleton/cluster mix is identical
  across speakers, so cluster coarticulation is a shared constant, not a between-speaker
  confound. Keep the data, check robustness.
- **Status:** SETTLED.

### C4. Raw (unnormalized) sibilant moments
- **Decision:** Report /s/ COG etc. raw, not speaker-normalized.
- **Rationale:** The literature reports raw moments (comparability), and /s/ frication
  depends mostly on the small front cavity, which varies far less with vocal-tract size
  than voiced formants — so normalization matters much less here.
- **Caveat:** Not perfectly size-free; larger tract nudges COG down. Optional sanity
  check: correlate COG with a vocal-tract-size proxy.
- **Status:** SETTLED (with caveat).

### C5. Stressed vowels only
- **Decision:** Measure only stress-bearing vowels (TextGrid stress digit 1/2), not
  reduced (0).
- **Rationale:** Unstressed vowels reduce toward schwa — shorter, centralized, blurred
  formant targets. Stressed vowels are fully articulated → cleanest targets.
- **Status:** SETTLED.

### C6. Diphthong dynamics via F1/F2 at 20/50/80% → trajectory length + rate
- **Decision:** Sample formants at 20/50/80% of each diphthong (AY, EY, OW, AW);
  derive trajectory length (distance travelled in F1×F2) and rate of change.
- **Rationale:** A diphthong's identity *is* its movement; a midpoint snapshot discards
  it. The gay-voice hypothesis is fuller, less-monophthongized glides → longer
  trajectory. 20/80 (not 0/100) trims the coarticulated edges so we capture the vowel's
  own motion.
- **Status:** SETTLED.

### C7. Lobanov normalization for vowels/diphthongs
- **Decision:** Z-score each speaker's formants across their own vowels before
  cross-speaker comparison.
- **Rationale:** Vowel formants depend heavily on vocal-tract *length* — raw F1/F2 would
  partly measure body size, not style. Lobanov re-expresses each vowel as "where it sits
  in this speaker's own space," removing anatomy while preserving the social variation
  (best-performing normalization in Adank et al. 2004).
- **Status:** SETTLED.

### C8. Minimum-token filter per phone per speaker → N = 5
- **Decision:** A speaker needs ≥ 5 clean tokens of a phone to get a value for it
  (else NaN). Set from the phone census (`segmental_census.py`).
- **Rationale:** The census (fixed script → near-constant availability, min≈median)
  showed every confirmatory target clears 5 for all 50 speakers (worst: IY/SH/AW at
  6), so N=5 never touches the backbone. It cleanly drops the genuinely rare phones
  where a per-speaker value would be alignment jitter (TH=3, ER=3, UH=2, CH=3 → 0/50
  at ≥5). Net measurable inventory: 12 stressed vowels (AA AE AH AO AW AY EH EY IH IY
  OW UW) + 7 fricatives (S SH Z F V DH HH; TH dropped, HH borderline at 46/50). Stops
  plentiful but deferred (C9).
- **Alternatives:** N=3 (keeps TH/ER but those are too sparse to trust); N=10 (would
  cost IY/SH/AW/OW targets — too aggressive).
- **Status:** SETTLED (N=5).

### C9. Extract broad, analyze narrow
- **Decision:** `extract_segmental.py` measures *all* cleanly-measurable phones
  (vowels: F1/F2/F3 + duration; fricatives: moments + duration) at the token level, and
  emits both a narrow theory table (for confirmatory tests) and the full broad table
  (for exploration + the classifier). Stops (VOT) and nasals deferred.
- **Rationale:** Extraction machinery is identical regardless of scope, so breadth is
  nearly free; the confirmatory/exploratory separation happens at the *analysis* layer
  (see A5). Stops/nasals need separate fiddly logic for weak theoretical payoff.
- **Status:** SETTLED (stops/nasals deferred).

### C10. Per-speaker alignment diagnostic
- **Decision:** Emit `n_tokens` / `n_dropped` per speaker from the extractor.
- **Rationale:** MFA forces alignment silently and never reports deviations. A high drop
  rate surfaces speakers who likely departed from the script — turning silent failures
  into a visible table.
- **Status:** SETTLED.
