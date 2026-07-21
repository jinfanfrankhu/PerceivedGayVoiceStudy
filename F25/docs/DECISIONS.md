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
- **Status:** REVISED — implemented and **now run** (2026-07-20). Under the original *linear*
  α spacing it **FAILED**: rel.err 213% (max 416%) at 20 steps, 117% (max 159%) at 50 steps,
  both heads, tol 5%. The `g(α)` curve shows why — ~100%+ of `f(x)−f(x₀)` accrues in α∈[0,0.01]
  (violent silence→onset response, then saturation), so uniform α samples the spike ~once, and
  the convergence rate (rel.err ∝ steps^−0.65) projects ~7000 steps to reach tol — impractical.
  This motivated the change-of-variables quadrature in **E10**. Per-phone numbers remain
  provisional until completeness passes under the new grid.

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

### E10. Non-uniform (power) α-quadrature for the IG path integral (`10d`)
- **Decision:** Approximate `∫₀¹ ∂f/∂x(x₀+α(x−x₀)) dα` on a **change-of-variables** grid
  `α = uᵖ` (uniform-`u` midpoints, default `p=3`, `IG_STEPS=32`), with quadrature weights
  `wⱼ = p·uⱼ^{p−1}/steps`, instead of the uniform-α midpoint rule. Env: `IG_ALPHA` (power|
  linear), `IG_ALPHA_POWER` (3.0). `p=1` recovers the original linear rule exactly.
- **Rationale:** The uniform rule was **run and failed completeness by 78–213%** (E8) because,
  for the silence baseline, essentially all of `f(x)−f(x₀)` accrues in `α∈[0,0.01]` — WavLM
  responds violently to the silence→signal onset and then saturates. Uniform α lands ~1 sample
  in that spike; the observed convergence rate projects ~7000 steps to reach the 5% tolerance.
  The power substitution is not a fudge: it is an **exact reparameterisation of the same
  integral**, so `Σᵢ IGᵢ = f(x)−f(x₀)` still holds in the limit and the completeness relative
  error remains the *independent* test that the discretisation succeeded — the estimator stays
  unbiased, only the sample allocation changes to match where the integrand has mass.
- **Alternatives considered:** (a) brute-force more uniform steps — rejected, ~7000 needed,
  weeks of CPU; (b) change the baseline to linearise `g(α)` (E11).
- **REFUTED BY EXPERIMENT (2026-07-20).** The power grid did **not** rescue the silence
  baseline — it made completeness *worse* (ridge 222%, enet 264% at p=3/32 steps, vs 117%/78%
  for linear/50). The premise that the onset is a *resolvable* spike is wrong: the silence→
  signal transition is effectively **near-singular** (the encoder's first response to signal
  onset jumps by orders of magnitude across an interval no practical grid subdivides), so
  concentrating α-points there piles them on a near-vertical wall and the midpoint rule
  diverges rather than converges. The real lever was the **baseline** (E11): with matched-
  spectrum noise, `g(α)` is near-linear and completeness passes at 32 *linear* steps
  (ridge 2.3%), with the α-schedule then nearly irrelevant. The power grid is kept as an
  available, unbiased option (`IG_ALPHA=power`) but is **not** the fix and is not the default
  choice once the baseline is well-posed.
- **Status:** REVISED — power-α retained as a harmless option; it does not solve the silence
  completeness failure. See E11 for what did.

### E11. IG baseline: silence primary, matched-spectrum noise as a sensitivity axis (`10d`)
- **Decision:** Keep **silence (zeros)** as the primary IG baseline (E7); add an optional
  **phase-randomized matched-spectrum noise** baseline (`IG_BASELINE=noise`) — the speaker's
  own clip with magnitudes preserved and phases randomized (fixed `IG_SEED`), destroying
  temporal/phonetic structure while matching the long-term spectral envelope. Both baselines
  are run through the same feature extractor as the input so they share its per-clip
  normalization. The noise result is reported as the planned baseline-sensitivity axis (E9),
  not as a replacement for silence.
- **Rationale:** IG attributes `f(x)−f(x₀)`, so the baseline defines the reference of
  "absence". Silence has zero free parameters and answers "what in the voice drives the score
  vs. nothing"; it is the standard default and keeps the primary interpretation clean. A
  matched-spectrum noise baseline answers a *different, narrower* question — "what drives the
  score **beyond** the gross spectral envelope" — because it credits the envelope to the
  baseline. That is exactly why it is informative as a **robustness check** (does the
  vowels-over-/s/ story survive removing the broadband/spectral-tilt component that the docs
  flag as a possible /s/-negativity artifact?) and exactly why it must not silently become the
  primary: it is entangled with the substantive claim and adds researcher degrees of freedom
  (which noise? matched to what?) that silence does not have.
- **Alternatives considered:** additive Gaussian/white noise (rejected: does not match the
  speaker's spectral envelope, so it removes a different, less interpretable thing); averaging
  IG over a *distribution* of baselines à la Expected Gradients (deferred: more compute; the
  two-baseline contrast is the reportable sensitivity axis for now).
- **EXPERIMENT (2026-07-20) forces the primacy question open.** The intended "silence primary"
  stance assumed silence-IG was merely under-resolved. It is not: silence-IG **fails
  completeness at every practical grid** (E8, E10) — the method is effectively ill-posed for
  this encoder, so it cannot be the primary result as written. The **noise** baseline, by
  contrast, is well-posed (ridge completeness 2.3% at 32 linear steps), yields *more* head-
  robust attributions (ridge/enet time-space r=0.66 vs 0.61 for silence, 0.45 in weight space),
  and both heads agree /s/ is **not** a positive cue (0% speakers positive, both). This is a
  reportable finding in its own right and *strengthens* the ICASSP thesis: even completeness —
  IG's most basic correctness axiom — is unsatisfiable under the standard baseline, and the
  baseline choice is itself an arbitrary, answer-changing degree of freedom (feeds E9).
- **DECISION (2026-07-20): NOISE is the primary baseline.** Silence's completeness failure is
  reported as a finding (it supports the reliability thesis), and silence-IG is kept runnable
  (`IG_BASELINE=silence`) only to reproduce that failure. Default config is therefore
  `IG_BASELINE=noise`, `IG_ALPHA=linear`, `IG_STEPS=128` (E12). Downstream: 10d's per-phone IG
  numbers are regenerated under noise. 10e is **IG-independent** (it correlates the head's vowel-region
  readout with hand vowel features and never uses a baseline), so it does not need re-running
  for this change — but its /aɪ/ interpretation is cross-referenced against the noise-baseline
  10d phone profile, so keep the two consistent when writing.
- **Prior art / grounding (added 2026-07-21).** The baseline is a *synthesis of two
  established techniques*, not an ad-hoc construction — each half is textbook:
  1. **Non-zero / distributional IG baselines are standard practice.** Sturmfels, Lundberg &
     Lee, *"Visualizing the Impact of Feature Attribution Baselines,"* Distill 2020
     (distill.pub/2020/attribution-baselines) shows the baseline is a *consequential*
     hyperparameter and analyses Gaussian-noise and other non-zero baselines; the
     Gaussian-centred-on-input idea traces to Smilkov et al. (SmoothGrad, 2017), and the
     principled distribution-averaged form is Expected Gradients (Erion, Janizek, Sturmfels,
     Lundberg & Lee, *Nat. Mach. Intell.* 2021). So "don't use a zero/silence baseline" is a
     mainstream position, and a noise baseline is one of the named alternatives.
  2. **Phase-randomised, power-spectrum-preserving surrogates are the standard "surrogate
     data" null.** Keep the FFT magnitude, randomise the phase, inverse-FFT: Theiler, Eubank,
     Longtin, Galdrikian & Farmer, *"Testing for Nonlinearity in Time Series: The Method of
     Surrogate Data,"* Physica D 58 (1992) 77–94 (doi:10.1016/0167-2789(92)90102-S). By the
     Wiener–Khinchin theorem, preserving the magnitude spectrum preserves the autocorrelation/
     power spectrum (the "linear" structure), while phase carries the temporal/nonlinear
     structure that randomisation destroys — exactly the "same spectral colour, no phonetic
     structure" property we want.
  Our baseline uses (2) as the reference distribution for (1), per speaker. A targeted search
  found IG applied to audio/speech (e.g. Spectral IG, arXiv:2605.19607; temporal-detection
  evaluation on sound classifiers, arXiv:2605.23293) but **no prior use of a phase-randomised
  matched-spectrum baseline for speech/SSL attribution specifically** — so the *combination* is,
  as far as we checked, new to this setting (a minor methods contribution for ICASSP), while
  neither ingredient is novel or exotic on its own.
- **Status:** SETTLED (noise primary; silence retained as the reported failure case).

### E12. Completeness verdict uses a scale-floored relative error (`10d`)
- **Decision:** Judge IG completeness per speaker by `|Σ IG − Δf| / max(|Δf|, median_j|Δf_j|)`
  (Δf = f(x)−f(x₀)), i.e. the ordinary per-speaker relative error but with the denominator
  **floored at the head's median |Δf|**. `completeness_ok` = max of this over speakers ≤ 5%.
  The raw per-speaker relative error, the absolute error, and a `small_signal` flag are still
  emitted per (speaker, head) in `wavlm_ig_completeness.csv`.
- **Rationale:** The raw relative error `|Σ IG − Δf|/|Δf|` is **ill-defined as Δf→0**. In the
  full noise-baseline run (64 steps) one speaker (Randall_Larry_McGaren) had |Δf|=0.005 for
  enet — ~1% of the median (0.502) — because the model's perceived-gayness readout on his real
  clip nearly equals its readout on his phase-randomized spectral surrogate. His *absolute*
  completeness error (0.0053) was smaller than ridge's typical absolute error, yet his raw
  relative error read 105%, single-handedly failing the `max ≤ tol` verdict. This is a
  substantive edge case (a voice whose readout is carried entirely by its spectral envelope, so
  Δf and its attribution are both ~0), not an integration failure — flooring the denominator at
  the typical signal judges such speakers on absolute closeness while leaving on-scale speakers'
  relative errors unchanged. It does **not** loosen the test for genuinely under-resolved runs:
  there the absolute errors are large for everyone, so the floored metric fails too.
- **Alternatives considered:** raw max relative error (rejected: one near-zero-Δf speaker makes
  the whole-run verdict meaningless); global normalisation by a single head-wide scale (rejected:
  unfairly penalises large-|Δf| speakers whose absolute error scales with their signal); median
  or 90th-percentile relative error as the verdict (rejected: weaker than a per-speaker
  guarantee, and hides which speakers are small-signal). Reporting the audit CSV keeps the raw
  numbers and the flag visible so nothing is swept under the floor.
- **Step count (why 128, not 64):** the floored metric revealed a *second*, non-metric issue —
  6/47 on-scale enet speakers had 5–13% completeness at 64 steps because the **sparse enet head
  produces strongly non-monotonic paths** (worst: Keith_Tadeo_Muna, g(α) swings −0.88→+1.08→
  −0.36 while net Δf=+0.52, so the quadrature must resolve a large mid-path hump to land a modest
  net). Ridge's paths are near-monotonic and clean at 64. Doubling to **128 steps** converged
  both heads (ridge max 0.87%, enet max 5.00%) — the authoritative run. So the default is 128;
  ridge alone would pass at 64. This head-dependent path complexity is itself worth a sentence in
  the paper (it is another face of the sparse-vs-dense probe difference in E6).
- **Status:** SETTLED (scale-floored metric; authoritative config noise / linear / 128 steps,
  both heads completeness_ok=True).

### E13. Attribute in the physical (phone/time) basis, not a reduced embedding subspace (`10c`/`10d`; `src/svd_rashomon_diagnostic.py`)
- **Decision:** Do **not** attack the attribution multiplicity by dimensionality-reducing the
  768-dim embedding (PCA/PCR) or by projecting attribution onto a "stable identifiable subspace."
  Attribute in the **fixed physical basis** (per-phone / per-frame, as 10c/10d already do) and
  quantify the residual multiplicity with a **Model-Class-Reliance attribution range** over the
  linear Rashomon set. Dimensionality reduction is rejected as a *fix*; the SVD is retained only
  as a *diagnostic* of the Rashomon set's geometry.
- **Rationale (empirical — `svd_rashomon_diagnostic.py`, L6, n=50, 500 bootstraps):** the SVD of
  the standardized pooled embeddings shows the multiplicity is **structural**, not a tidy
  low-rank artifact:
  1. **The space is genuinely high-dimensional.** Effective rank (entropy) ≈ **31**, participation
     ratio ≈ 22; **30 of the 49 PCs** are needed for 90% variance — a flat spectrum. So there is
     no low-dim subspace to reduce *to* without discarding signal. PCA/PCR reduction would throw
     away predictive information, not concentrate it. (Consistent with the earlier 07 verdict that
     PCR ≈ hard-thresholded ridge, low value.)
  2. **The perceived signal is distributed, and not in the high-variance directions.** Strongest
     single correlate is **PC5 (Spearman ρ=+0.43, p=.002, 5% variance)**; the rest is scattered
     across PC9/10/15 and reaches into an **unstable tail (PC25, 1.2% variance)**. PCs 1–4 (top
     variance) barely correlate. The full probe's ρ≈0.73 therefore *aggregates ~a dozen weak
     directions* — one PC alone only reaches ρ≈0.43. (Per-PC p-values uncorrected across 49 tests:
     PC5 survives correction; PC9/15/25 are suggestive only.)
  3. **There is no clean stable subspace to attribute in.** Bootstrap subspace stability (mean
     cosine of principal angles) is only **0.62 for the single leading direction**, rising to
     ~0.81 by top-10 — i.e. the embedding basis itself does not replicate at n=50. This kills the
     "project attribution onto the stable top-k subspace" protocol arm: the stable subspace does
     not exist. (Subspaces are more stable than the individual PCs inside them — the span holds up
     better than any one axis — which is *why* the physical-basis attribution of 10d, r=0.61–0.66,
     beats the embedding-basis weight-space attribution of 10c, r=0.45.)
  4. **Ordinary collinearity is not the villain; the exact null space is.** Condition number of the
     retained 49-dim subspace is only **~64** (modest). The dominant non-identifiability is the
     **719-dim exact null space** (directions with zero between-speaker variance but nonzero
     within-speaker frame variance) — the data-covariance-vs-encoder-Jacobian mismatch named in E7.
     This is *why* attribution can move while predictions do not.
- **Alternatives considered:** (a) PCA/PCR reduction to k dims — **rejected**, the flat spectrum
  means k must be ≈ full rank to keep the signal, so it reduces nothing and only relocates the
  arbitrariness into "which k / which rotation." (b) Supervised reduction (PLS) — deferred; it
  would concentrate the signal but selection-by-y must live inside CV, and it does not remove the
  null-space freedom that actually drives the multiplicity. (c) "Attribute in the bootstrap-stable
  top-k subspace" — **rejected** on 3 above (no such stable subspace at n=50). (d) MCR attribution
  range over the Rashomon set — **adopted as the quantification**, PENDING implementation.
- **Consequence for the paper:** this pushes the study toward the *structurally-under-determined*
  reading, which **strengthens** the reliability thesis (the ρ.73 rests on a distributed, partly
  non-replicating signal inside a 719-dim free null space — single-model attribution cannot be
  trusted) while **retiring** the "clean subspace fix" framing. Constructive claim = attribute in
  the fixed physical basis + report the MCR multiplicity range, not dimensionality reduction.
- **Status:** SETTLED that dimensionality reduction is *not* the fix and attribution stays in the
  physical basis (diagnostic is reproducible via `src/svd_rashomon_diagnostic.py`, outputs
  `tables/wavlm_svd_rashomon.csv` + `figures/prediction/wavlm_svd_rashomon.png`). **PENDING:** the
  MCR attribution-range computation that quantifies the spread.
