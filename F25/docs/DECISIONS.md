# F25 Methodological Decision Record

Living record of *why* each analysis choice was made — the justification behind the
numbers, so the reasoning survives past memory and can seed the paper's methods
section. Format per entry: **Decision · Rationale · Alternatives considered · Status.**

Status legend: **SETTLED** (in use) · **PENDING** (awaiting data/decision) ·
**REVISED** (superseded, kept for the paper trail).

---

## D. Pre-registration — confirmatory segmental hypotheses (`04_segmental.py`)

Directions committed **before** testing (from the lit review), so `04` uses ONE-TAILED
tests in the predicted direction over a small family — the power fix for 02's null.
Key insight: the literature splits into **production** studies (what gay men actually
produce → predicts vs `true_kinsey`) and **perception** studies (what listeners respond
to → predicts vs `perceived_mean`). They AGREE for sibilants/vowels but DIVERGE for
pitch/breathiness — testing that divergence is the paper's core claim.

Sign = predicted correlation direction (higher target = gayer / higher Kinsey).
Tested vs BOTH targets, BH-FDR (q=0.10) within each target's family separately; effect
size + bootstrap CI lead (decision A3); sensitivity re-run excluding transmen.

**Group A — production & perception agree (same sign both targets):**
| Feature | Perceived | Actual | Source |
|---|---|---|---|
| S_cog (/s/ fronting) | + | + | Munson 2006; Rogers & Smyth 2003 |
| S_skew | − | − | Munson 2006 (top differentiator) |
| S_dur | + | + | Rogers & Smyth 2003; Smyth 2003 |
| vowel_space_area | + | + | Pierrehumbert 2004; Rendall 2007 |
| front_f2 | + | + | Munson 2006; Rendall 2007 |
| diphthong dynamism (mean z-trajlen) | + | + | Panfili 2011 |
| diphthong duration | + | + | Panfili 2011 |

**Group B — production & perception DIVERGE (opposite/null — the payoff):**
| Feature | Perceived | Actual | Source |
|---|---|---|---|
| F0 range (pctlrange0-2) | + | − | Gaudio 1994; Rogers & Smyth 2003; Holmes 2024 |
| F0 stddevNorm (dynamism) | + | − | Gaudio 1994; Holmes 2024 |
| F0 mean | + | − | Holmes 2024 (gay men lower pitch) |
| v_cpps | − | + | Holmes 2024 (gay men *less* breathy) |
| HNR | − | + | Holmes 2024 |
| v_h1h2 | + | − | Holmes 2024 |

- **S_skew sign caveat:** the 750 Hz high-pass (C13) shifts absolute skew positive, but
  the *correlation direction* (more-fronted /s/ → more-negative skew) is preserved, so −
  stands.
- **diphthong duration caveat:** partly confounded with speech rate (Pierrehumbert 2004
  argued vowel expansion reflects precision, not slow rate); flagged, revisit with a rate
  covariate if it's the lone Group A hit.
- **Group B actual = directional per Holmes production study**, not merely "null" — Holmes
  found gay men lower-pitched and less-breathy, so we pre-register those signs; a true null
  simply won't reject.
- Individual low-front vowel formants (AE/EH F1) held EXPLORATORY (two-tailed) — the lit
  flags them as differentiators but doesn't pin a direction.
- **Status:** SETTLED (pre-registered 2026-07-10).

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
  OW UW) + 7 fricatives (S SH Z F V DH HH; TH dropped). HH retained despite being
  borderline (46/50 clear 5; the other 4 get NaN) for its breathiness relevance —
  see C11. Stops plentiful but deferred (C9).
- **Alternatives:** N=3 (keeps TH/ER but those are too sparse to trust); N=10 (would
  cost IY/SH/AW/OW targets — too aggressive).
- **Status:** SETTLED (N=5).

### C11. Breathiness voice-quality block on stressed vowels
- **Decision:** Add CPPS (smoothed cepstral peak prominence; primary) and H1–H2
  (secondary) measured on stressed vowels, as a confirmatory breathiness target.
  Predicted direction: breathier → perceived gayer, i.e. **lower CPPS** and **higher
  H1–H2**.
- **Rationale:** Breathiness is a documented correlate of perceived male sexuality,
  but it is a *voice-quality* property of voiced sounds — so it's measured on vowels
  via cepstral/spectral-tilt cues, NOT via the /h/ (HH) segment. The two are distinct;
  keeping HH (C8) captures the segment, this captures the phonation. CPPS is the robust
  modern standard (aperiodic voice → lower CPPS); H1–H2 is the classic harmonic measure
  (breathy voice → dominant fundamental → higher H1–H2). Cheap, since we already open
  every stressed-vowel interval.
- **Status:** SETTLED.

### C13. High-pass filter before sibilant spectral moments (750 Hz)
- **Decision:** High-pass the fricative (pass-Hann band, 750 Hz–Nyquist) before
  computing spectral moments.
- **Rationale:** Raw fricative spectra are contaminated by low-frequency energy (room
  rumble, voicing bleed) that drags COG down and puts the spectral peak near DC — a
  probe gave a nonsensical peak of 32 Hz and COG ~3.5 kHz. High-passing at 750 Hz (well
  below /s/ frication energy, which starts ~2–3 kHz) restores plausible values
  (COG ~5.1 kHz, peak ~3.9 kHz). The cutoff is not sensitive (500/750/1000 Hz give
  COG within ~80 Hz); 750 Hz is a common literature choice.
- **Consequence for skew:** removing the low-frequency tail flips spectral *skew* to
  positive (our /s/ skew mean ≈ +1.0), opposite the classic negative-skew literature
  computed on the full band. Skew stays a valid *within-study* measure (all speakers
  filtered identically) but its **sign is not comparable to unfiltered literature** —
  so let the data set skew's predicted direction, and treat COG/peak as the robust
  primary sibilant cues.
- **Status:** SETTLED.

### C12. Formant analysis settings
- **Decision:** Burg LPC formants, max 5 formants, ceiling 5000 Hz, 25 ms window,
  50 Hz pre-emphasis; report F1–F3.
- **Rationale:** Standard adult-male settings; a fixed ceiling is reproducible and the
  speakers are (nearly) all male. Per-token ceiling optimization (FastTrack-style, as
  F24 uses) is heavier and deferred; the plausibility filters (C10-adjacent) catch gross
  tracking errors.
- **Caveat:** 2 transmen and any higher-pitched voices may be slightly better served by
  a higher ceiling; revisit if the plausibility filter drops many of their tokens.
- **Status:** SETTLED (revisit if drop rate is high).

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

---

## E. SSL representations & attribution (`10`–`10e`)

### E1. WavLM is FROZEN; only a small head is fit
- **Decision:** `microsoft/wavlm-base-plus` runs in eval mode with `requires_grad_(False)`
  throughout. The only thing fit on our data is a linear head (~769 numbers).
- **Rationale:** The encoder has ~95M parameters and we have 50 speakers. Fine-tuning would
  memorize the sample and generalize to nobody. All the representational learning already
  happened on 94k hours; we borrow the representation and fit only the readout. This is the
  standard frozen-SSL + lightweight-probe protocol (SUPERB, Yang et al. Interspeech 2021).
- **Alternatives:** fine-tuning (rejected: catastrophic overfit at n=50); a deeper MLP probe
  (rejected: more capacity to fit noise, and probe capacity is known to change conclusions —
  Zaiem et al., Interspeech 2023).
- **Status:** SETTLED.

### E2. Cache all 13 layers; choose the layer INSIDE cross-validation, with a max-statistic null
- **Decision:** `10_wavlm_extract.py` stores every hidden state (13, 768) per speaker.
  `10b` evaluates all 13 and reports a family-wise p from a **max-statistic permutation
  null** — each permutation shuffles the label once and records all 13 layers together, so
  the null is the null of *the best of 13*.
- **Rationale:** Which layer carries a property is a hyperparameter, and picking the winner
  post-hoc is double-dipping. The max-stat null is the correct family-wise control. Layer
  specialization is real and documented (Pasad, Chou & Livescu, ASRU 2021), so the choice
  cannot simply be fixed a priori either.
- **Result:** perceived ρ=0.73 at layer 6 of 12 — the middle of the stack, where
  paralinguistic content is expected; a small independent sanity check. Actual orientation
  does not beat the null at any layer.
- **Status:** SETTLED.

### E3. Mean-pool over time → one vector per (speaker, layer)
- **Decision:** Average the per-frame hidden states across the whole utterance. Long clips
  are chunked at 20 s purely to bound self-attention memory, with each chunk's sum
  accumulated and divided by the *total* frame count, so the result equals a single global
  mean-pool.
- **Rationale:** Keeps n honestly 50 (one row per speaker) and matches the standard probing
  setup. Chunking does not distort the average, and WavLM's positional encoding is
  convolutional/relative, so there is no absolute-position information to disrupt.
- **Caveat that matters for interpretation:** mean-pooling is **duration-weighting**. Vowels
  are ~40% of frames, /s/ ~2%, so /s/ can only move the pooled vector by ~2% of its budget
  regardless of how informative it is. Any "the model reads X not Y" claim from pooled
  representations is conditional on this. A duration-insensitive pooling (attention, max)
  is a different question that could answer differently.
- **Status:** SETTLED (with caveat).

### E4. Two heads — ridge and elastic net — on the same layer
- **Decision:** Every prediction and attribution step is run through both an L2 (ridge,
  dense) and an L1+L2 (elastic net, sparse) head, fit at the same layer.
- **Rationale:** Both minimize squared error and differ only in how coefficients are
  penalized, so the pair is a controlled test of whether a signal is distributed or carried
  by a sparse subset. Ridge keeps all 768 dims; enet keeps ~30. Mirrors the `07`/`07c` twin.
- **Status:** SETTLED — and see E6, where the pair's *disagreement* became the finding.

### E5. Frame attribution by exact linear decomposition, not occlusion (`10c`)
- **Decision:** Decompose the prediction into per-frame contributions algebraically:
  `pred = const + mean_t (w_eff · h_t)`, `w_eff = coef / scale`.
- **Rationale:** The head is linear on a mean-pooled embedding, so this is exact — no
  perturbation, no re-fit, no leakage, one extra forward pass. Occlusion is the wrong tool
  here: silencing a ~2%-of-frames /s/ barely moves a whole-utterance mean (E3), so occlusion
  would measure phone *duration*, not phone *importance*.
- **Status:** SETTLED as a method; its conclusions are not head-invariant (E6).

### E6. Weight-space attribution is PROBE-DEPENDENT — this is a finding, not a bug
- **Decision:** Report the ridge/enet disagreement rather than picking the more convenient
  head.
- **Finding:** at the same layer with comparable predictive performance (ρ=0.73 / 0.67), the
  two heads produce frame attributions correlating only r=0.45 and **opposite** per-phone
  signs (ridge: /s/ contrast −2.44; enet: +0.30).
- **Rationale:** The 768 dimensions are collinear, so many weight vectors predict the pooled
  mean equally well and coefficient-reading is not identifiable. This is the textbook
  multicollinearity result, and lasso's arbitrary selection from a correlated group is the
  original motivation for elastic net (Zou & Hastie) — i.e. the disagreement is *predicted*,
  not anomalous. Cf. Antverg & Belinkov, ICLR 2022, on probe weights as a poor basis for
  ranking dimensions.
- **Status:** SETTLED.

### E7. Input-space integrated gradients (`10d`) — and what it does NOT fix
- **Decision:** Attribute in input space by integrating gradients along a straight path from
  a silence baseline to the clip (20 midpoint steps, first 10 s per speaker), then bin to
  phones via the MFA TextGrids.
- **Rationale:** Weight-space attribution credits *dimensions*, which are collinear.
  Gradients w.r.t. the waveform credit *moments*, which are not interchangeable in the same
  way. Empirically this helped: head agreement rose to r=0.61 and the per-phone signs
  reconciled (vowels positive, /s/ negative, both heads).
- **REVISION — the original rationale was too strong.** "Input space escapes the probe
  arbitrariness" is false. By IG's own Linearity axiom,
  `IG_i(f) = Σ_j w_j · IG_i(pooled_j)` — the attribution map is a *linear function of w*, so
  a different equally-good probe can give a different map. Implementation Invariance does not
  help (it covers re-parameterising one function, not selecting among many that fit equally
  well). See Bilodeau, Jaques, Koh & Kim, *PNAS* 121(2), 2024, which proves complete+linear
  attribution methods — naming IG and SHAP — can fail to beat random guessing for inferring
  model behaviour. **The observed ridge/enet agreement is an empirical observation, not a
  guarantee.**
- **Window caveat:** the first 10 s is the *same text* for every speaker (read passage), so it
  is well controlled; coverage is adequate (7–8 /s/, 32–52 vowels, 4–6 /aɪ/ per speaker). What
  it does not control is reading-onset style, and it captures only ~24% of each speaker's /s/.
- **Status:** REVISED (method retained; the "IG resolves it" interpretation withdrawn).

### E8. IG completeness is checked, not assumed
- **Decision:** Compute `f(x)` and `f(x₀)` explicitly and report
  `|Σ IG − (f(x) − f(x₀))| / |f(x) − f(x₀)|` per head, plus the full `g(α)` path curve.
- **Rationale:** The path integral is approximated by a midpoint rule over linearly spaced α.
  Linear spacing is perceptually lopsided (α=0.025 is −32 dB, α=0.975 is −0.2 dB), so the
  quiet end — where the encoder's response plausibly moves fastest — is sampled once. Whether
  that under-resolves the integral is empirical, not a matter for a priori argument:
  completeness holds iff the discretization captured the path. Costs 2 extra forwards per
  speaker (~3%). Sundararajan et al. (2017) recommend exactly this check; no speech paper
  found reports it.
- **Status:** PENDING — implemented, **never run**. Per-phone numbers are provisional until
  it passes.

### E9. Attribution claims must survive the probe's equivalence class (planned)
- **Decision:** Replace single-probe attribution with a Rashomon-set protocol — sample
  K≈20 equally-performing probes (bootstrap refits / regularization within the
  statistically-indistinguishable range), run IG for each, and report the *range* per phone
  plus sign agreement across K. Claim only what survives.
- **Rationale:** Follows directly from E6 and E7: if attribution is a function of an
  arbitrary choice among equally-good probes, the defensible object is the consensus across
  that set, not any one fit. This is the established remedy in the model-multiplicity
  literature (Laberge et al., *JMLR* 24(364), 2023; Fisher, Rudin & Dominici; Breiman 2001),
  where attribution spread provably widens along near-collinear directions.
- **Cost note:** the forward pass is already shared across heads (`retain_graph=True`), so K
  probes cost `1 forward + K backwards` per α. Because IG is linear in w (E7), the K probes
  can also be PCA'd to a few principal directions and the rest reconstructed by linear
  combination.
- **Supporting practices:** group collinear dimensions before attributing; report baseline
  sensitivity (silence vs matched-spectrum noise) as its own axis; judge stability
  numerically on **signed** values — Adebayo et al. found IG passes parameter-randomisation
  only in signed rank correlation while the rendered map stays deceptively intact.
- **Status:** PENDING.
