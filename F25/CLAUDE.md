This is a research project investigating the "gay voice" phenomenon within Gen Z males.

See `README.md` for the full folder layout, run order, and method notes. Quick map:

- **Raw data** (immutable): `data/raw/` — `Master Spreadsheet.csv` (metadata + true
  Kinsey, canonical speaker names), `ratings.csv` (listener ratings from the Neon
  platform), `features.csv` (88 eGeMAPSv02 features).
- **Processed data** (regenerable): `data/processed/` — built by `src/build_dataset.py`.
- **Code**: `src/` (`common.py` paths, `build_dataset.py` first, then numbered analysis
  scripts). All paths resolve relative to the script, so cwd doesn't matter.
- **Outputs**: `outputs/figures/`, `outputs/tables/`.
- **Archive**: `archive/` holds the superseded exploratory pipeline (old
  `run_classifiers.py` config-driven experiments, `classifier_logs/`,
  `feature_rankings/`). Reference only — not the active path.

Use the **`gayvoice` conda env** (Python 3.13): `conda activate gayvoice`. The system
`python` (3.14) lacks the scientific stack, and `py -3.13` no longer resolves. Deps are
locked in the repo-root `requirements*.txt` — `requirements.txt` covers `01`–`09`,
`requirements-all.txt` adds the extraction + WavLM stacks.

## Kinsey Scale
Modified scale: 1 = complete attraction to women, 2 = preference to women, 3 = even bi,
4 = preference to men, 5 = complete men. Ratings use the same 1–5 scale
(1 = sounds straight ... 5 = sounds gay).

## Stimulus
All 50 speakers read the **same scripted passage** (opens "so this morning i woke up late
and..."), ~42–73 s each. Content is therefore constant across speakers and only voice varies —
which is why fixed-window analyses (e.g. `10d_wavlm_ig.py`'s first-10 s slice) are comparable
speaker-to-speaker rather than confounded by what was said.

## Three data layers (same 50 speakers)
Acoustics (features.csv) · Actual orientation (Kinsey + self-ID) · Perceived gayness
(listener ratings). Central questions: acoustics→actual, acoustics→perceived,
perceived-vs-actual accuracy.

## Data-integrity gotcha
Speaker names are canonicalized to the Master Spreadsheet. The SAME three speakers were
originally misspelled in two independent places and both were corrected to the Master
canonical spelling (Chandan→Chandani, Steven→Samuel, Gallway→Galway):
  1. `data/raw/ratings.csv` (listener export) — guarded by `build_dataset.py check_join()`.
  2. The **audio filenames** in `clean_wavs/`, and hence `mfa_corpus/` + `mfa_textgrids/`
     folder names — which feed `file_id` in the segmental pipeline. These were renamed
     to canonical; `04_segmental.py` / `05_segmental_explore.py` now warn loudly if any
     segmental `file_id` fails to match Master (this bug silently ran the segmental
     analyses on n=47 before it was caught). No MFA re-run was needed — TextGrid content
     is filename-independent.
Join everything through `data/processed/crosswalk.csv`.

## ICASSP framing (researched 2026-07-20)

**The contribution must be methodological, not applied.** IEEE SPS reviewer guidelines
reject "trivial application of an existing technique in a different context." Frozen-SSL +
linear probe is a fully established paradigm (SUPERB, Yang et al. Interspeech 2021), so
"we ran WavLM on a new dataset" is not a contribution.

> ⚠️ **The "10c → 10d arc is the ICASSP contribution" plan is WEAK — two adversarial
> novelty searches on 2026-07-20 (one outside speech, one inside).** Net: the *idea* is
> not novel (extensive NLP/vision/statistics prior art), but the *specific speech
> instantiation* is unoccupied. It can only be written as "this known NLP failure mode is
> live in speech SSL probing, and here is what it costs" — never as a discovery. Read
> both *Novelty search* sections below before reviving it.

### Novelty search result (2026-07-20) — prior art on probe-dependent attribution
The proposed claim was: *weight-space attribution over frozen SSL embeddings is
probe-dependent because the dimensions are collinear; input-space IG is stable.* Each
component is already published:
- **Antverg & Belinkov, ICLR 2022**, "On the Pitfalls of Analyzing Individual Neurons in
  Language Models" (arXiv:2110.07483) — criticises *exactly* the practice of ranking
  dimensions by probe weights. The single most dangerous citation.
- **Gairola et al., ICLR 2025**, "How to Probe: Simple Yet Effective Techniques for
  Improving Post-hoc Explanations" (arXiv:2503.00641) — frozen DINO/CLIP + probes +
  attribution; shows probe *design* changes the attributions. Same experimental frame,
  different modality.
- **Krishna et al. 2022**, "The Disagreement Problem in Explainable ML" (arXiv:2202.01602)
  — attribution methods contradict each other; IG is one of the six studied.
- **Hewitt & Liang EMNLP 2019; Belinkov, *CL* 48(1) 2022** — probe design determines
  interpretability conclusions. Genus-level prior art.
- **Breiman 2001 (Rashomon effect); Fisher/Rudin/Dominici; Semenova & Rudin ICML 2024;
  Laberge et al. JMLR 2023** — "equally good models give contradictory importances" is a
  mature literature whose *established remedy is Rashomon-set analysis*, not "switch to
  input space." A reviewer may ask why we did not do the standard thing.
- **Multicollinearity → non-identifiable coefficients** is textbook regression theory, and
  lasso's arbitrary pick-one-of-a-correlated-group is the original motivation for elastic
  net (Zou & Hastie). Our ridge-vs-enet disagreement is the *textbook prediction*.

**⚠️ Unverified but high-risk:** arXiv:2605.21492 (2026) reportedly proves no feature
ranking is simultaneously faithful, stable and complete under collinearity, and reportedly
claims gradient/input-space attribution does **not** escape it. If correct this undercuts
10d's positive claim directly. Low-confidence extraction — **verify before citing or being
blindsided.**

**What could still survive** (per the search): not the phenomenon, but a speech-specific
consequence — e.g. showing the collinearity structure of speech SSL embeddings is
qualitatively different (temporal smoothness, phone-identity redundancy), or concretely
demonstrating that *specific published speech-phonetics probing conclusions are artifacts*.
"This known phenomenon also occurs in WavLM" is not a contribution.

### Novelty search #2 (2026-07-20) — the SPEECH domain specifically
Complementary search inside eess.AS / cs.SD. **No paper fully scoops it.** An arXiv abstract
search for `"integrated gradients" AND "probe"` in eess.AS ∪ cs.SD returns **zero** results,
and the Interspeech 2025 speech-interpretability tutorial taxonomises probing and
feature-attribution as *separate, never-compared* families. Specifically unoccupied: no
speech paper compares ridge/lasso/enet probe coefficients as interpretations, reports a
frame-level agreement r between weight-space and input-space attribution, or backprops IG
through a **frozen** encoder for a **linear probe** (all IG-on-frozen-SSL work found
attaches a trained backend).

**Two exposures to manage:**
- **arXiv:2605.01381** (Edinburgh CSTR, May 2026), "A framework for analyzing concept
  representations in neural models" — five concept-subspace estimators on **HuBERT
  frame-level features with phone labels**; states "high-retention subspaces may not be
  unique… there can be redundancy in the representation space." **Already published the
  redundancy/non-uniqueness half, in speech, at phone level.** Must cite and position against.
- **Sajjad, Durrani & Dalvi, TACL 2022** (arXiv:2108.13138) — survey stating outright that
  probe regularisation determines neuron ranking (L1 spiky, L2 distributed, elastic net
  chosen *because* of correlated neurons, per Dalvi et al. AAAI 2019). The mechanism is
  textbook in NLP probing.

**⚠️ The arbiter claim is the weak point, and this is a SCIENCE problem, not just a venue
one.** Shen et al. (Interspeech 2025) report inter-seed agreement **below 0.6** for most
speech attribution conditions and call that regime unreliable. Our cross-head r = **0.61**
sits right at that line. Do not lead with r=0.61 as evidence that "IG resolves it." The
defensible evidence is the **sign consistency** (enet /s/ flips from +0.30 in weight space
to −6.5e−6 in input space; both heads agree on vowels-positive / /s/-negative). Also note
the quantities differ — Shen varies *random seeds on a fine-tuned model* (the model itself
changes), we vary *the probe on a frozen model* (model identical). Ours is arguably the
cleaner comparison, but that argument has to be made explicitly, not assumed.

Useful nearby precedents for phone-binned attribution (cite, don't re-derive): Fucci et al.,
Interspeech 2025 (arXiv:2506.02181, "Echoes of Phonetics"); arXiv:2406.10422 (phoneme-binned
saliency); Wu, Bell & Rajan, ICASSP 2024 (arXiv:2305.18011).

**Must-cite, all directly on point:**
- **Shen et al., Interspeech 2025**, "On the reliability of feature attribution methods for
  speech classification" (arXiv:2505.16406) — wav2vec2, concludes attribution methods are
  "generally unreliable" for speech. Our 10c/10d arc is a *constructive* answer to this.
- **Sigurgeirsson & Ungless, Interspeech 2024**, "Just Because We Camp, Doesn't Mean We
  Should: The Ethics of Modelling Queer Voices" (arXiv:2406.07504) — the ethics precedent.
- **Kachel et al., Interspeech 2023** — closest published analogue (n=72, 35 raters),
  pre-registered on OSF, and withheld stimuli from public release on ethical grounds.
- **Zaiem et al.** (arXiv:2308.14456) — probe-capacity changes SSL model rankings. A
  reviewer may raise it; pre-empt.
- **Pasad, Chou & Livescu, ASRU 2021** (arXiv:2107.04734) — layer-wise probing precedent.

**Two places we exceed venue norms — say so explicitly, they are free credit:**
1. **IG completeness/convergence is reported by ~nobody.** Zero speech papers found doing
   it, including papers whose whole purpose is evaluating IG-family methods. Sundararajan
   et al. (2017) explicitly recommend it. See the completeness TODO below.
2. **No permutation/label-shuffling test was found in any ICASSP/Interspeech paper.** Our
   max-statistic null across 13 layers is well beyond convention.

**Ethics section is mandatory** — a named "Compliance with Ethical Standards" statement,
required whether or not approval was needed, allowed on the 5th page. ICASSP has **no**
policy specific to protected attributes or sensitive-category inference, so the framing is
ours to get right: the null on *actual* orientation is the protective result and should be
stated deliberately, not left implicit.

### Novelty search #3 (2026-07-20) — the perceived-vs-actual DISSOCIATION framing
Alternative ICASSP framing: *"a frozen SSL probe predicts listener-PERCEIVED gayness
(ρ=0.73) but ACTUAL orientation at chance — when benchmark labels are human annotations,
the probe may learn annotator perception, not the attribute."* Verdict: **nothing fully
scoops it, but neither half of the claim is new.**

**The two papers that must be cited or the paper looks unread:**
- **Kachel, Simpson & Steffens (2018)**, "Do I Sound Straight?", *JSLHR* 61(7) — 25 gay +
  26 straight German men, self-reported orientation *and* 74 listeners' perceptions on the
  same speakers. Perceived tracked f0, /s/ CoG, F2; **actual produced essentially one weak
  difference (F1)**. Concluded speech stereotypes "do not contain a kernel of truth."
  **Covers ~70% of our empirical claim in classical acoustics.** Our delta: foundation-model
  hypothesis space (so a *stronger* null), continuous ratings + Kinsey, 3,450 ratings, a
  real permutation null instead of "n.s.", and the benchmark-validity framing they omit.
- **El-Tawil, Sampath & Mower Provost, ICASSP 2026**, "What You Feel Is Not What They See:
  On Predicting Self-Reported Emotion from Third-Party Observer Labels" (arXiv:2601.21130)
  — observer-trained models, **near-zero CCC on self-report**. Same logical structure, in
  speech, at ICASSP, four months ago, from a lab reviewers know. **The generalized claim
  cannot be presented as new.** But note it is *also proof the venue accepts this paper
  shape* — position as a second domain extending it, not as a discovery.

Also cite: **Wong et al. 2026** (arXiv:2604.25776) — an explicit benchmark-validity critique
of SER; **Jacobs & Wallach, FAccT 2021**, "Measurement and Fairness" — the construct-validity
frame; **Liao, Song & Gunes**, *IEEE TAFFC* (arXiv:2210.09138) — apparent vs self-reported
personality, paired design; **Tomašev et al., AIES 2021** (arXiv:2102.04257) — orientation as
a prototypically *unobserved* characteristic; **Agüera y Arcas, Todorov & Mitchell (2018)** —
the Wang & Kosinski rebuttal (classifier reads self-presentation, not the attribute; our
paired design is methodologically cleaner than theirs); **Sanchez, Ross & Markl, SLT 2024**
(arXiv:2409.13335) — gender-label underspecification across 107 Interspeech papers.

**What is genuinely unoccupied (searched, not found):**
1. Any SSL/foundation-model probe on paired perceived-vs-actual sexual orientation, any modality.
2. Any construct-validity critique aimed at **SSL probing benchmarks' labels** — the probing
   self-critique literature (Belinkov *CL* 2022) is entirely about probe capacity/selectivity.
3. Any speech paper reporting **chance-level ML prediction of actual orientation**. No
   published speech "gaydar null" exists.
4. Any speech paper whose headline is "attribute X cannot be inferred, therefore do not build
   this." Precedent exists outside speech (Bowyer et al., *criminality-from-face*, T-TBIOM
   2020; Cox et al. 2016 on the gaydar myth) but not inside it.

**Recommended positioning:** a second domain replicating El-Tawil et al. (2026) where the
target attribute is *protected* and the correct conclusion is **abstention rather than better
modelling** — citing Kachel et al. (2018) as the classical-acoustics result we extend to
foundation models. Do not claim the generalized insight as novel.

### ⛔ CRITICAL — "input-space IG escapes the probe arbitrariness" IS FALSE (verified 2026-07-20)
**Do not write this claim. It is refuted by IG's own axioms.** Sundararajan et al.'s
**Linearity axiom** gives, for `f = Σ_j w_j · pooled_j`:

    IG_i(f; x, x0)  =  Σ_j  w_j · IG_i(pooled_j; x, x0)

The per-sample attribution map is a **linear function of w**. Perturb w by δ and the map
changes by `Σ_j δ_j IG_i(pooled_j)` — zero only if δ lies in the null space of the
path-averaged encoder Jacobian, which is a *different* object from the data-covariance null
space that made δ unidentifiable in the first place. **The probe's arbitrariness passes
straight through into the IG map.** Our ridge-vs-enet pair are simply two points in the same
Rashomon set; a third equally-good probe can give a third IG story.

*Implementation Invariance does not rescue it* — that axiom covers re-parameterising **one**
function ("outputs equal for all inputs"), not choosing among **many** functions that merely
fit equally well. Wrong axiom for this worry. Cf. Black, Leino & Fredrikson
(arXiv:2111.08230): models with identical predictions can have arbitrarily different
gradients almost everywhere.

**The real impossibility result to cite:** **Bilodeau, Jaques, Koh & Kim, "Impossibility
Theorems for Feature Attribution," *PNAS* 121(2), 2024** (arXiv:2212.11870) — peer-reviewed,
and it names IG explicitly: *"any feature attribution method that is complete and linear —
for example, Integrated Gradients and SHAP — can provably fail to improve on random guessing
for inferring model behaviour."*

**Do NOT cite arXiv:2605.21492** (Caraker et al., "Attribution Impossibility"). Verified: the
paper is real, but IG appears **once**, in a parenthetical list of ways to instantiate a
global scalar. Zero occurrences of "saliency" or "input space". It is about ranking flips
across *retrained* models and explicitly says "a fixed pipeline with a fixed seed produces a
deterministic, reproducible ranking." Unreviewed preprint, unaffiliated authors
(gmail addresses), no citations, 2-star repo, four-line pigeonhole proof. Irrelevant here.

**Our r=0.61 is NOT comparable to Shen et al.'s numbers.** Verified from their full text:
their metric is **ISA = inter-seed top-20% index overlap** across **nine re-fine-tuned
models**, chance floor ~0.20, reported as boxplots with no results table. Ours is a Pearson
correlation of frame-level attributions across **two probes on a frozen model**. Different
metric, different source of variation. Do not match the numbers in either direction. (Their
substantive findings still matter: IG was their *most* reliable method yet "does not surpass
50% inter-seed agreement for most tasks," and they diagnose the same cause we do — "high
resolution and highly correlated redundant features.")

**THE SALVAGE — and it is a better paper.** Stop claiming IG is the stable arbiter. Instead:
1. **Sample the probe's equivalence class** — bootstrap refits / seed sweep / an ε-loss ball
   of equally-good probes (K≈20 costs ~10× the current IG run; feasible as a weekend job).
2. **Compute IG for each sampled probe**, and report the **range** per phone/region rather
   than a point estimate from one fit.
3. **Claim only the consensus** that survives across the set. Single-w claims are indefensible.
This is the established Rashomon-set remedy (Laberge et al., *JMLR* 24(364), 2023 — attribution
spread is `aᵀω_S ± √(δ·aᵀ(HᵀH/N)⁻¹a)`, and that inverse Gram **blows up along near-collinear
directions**, i.e. collinearity sets the width). Also group collinear dims before attributing
(φ_ij = φ_i + φ_j), report **baseline sensitivity** as its own axis (silence vs matched-spectrum
noise), and judge stability **numerically on signed values, never visually** — Adebayo et al.
found IG passes parameter-randomisation only in signed rank correlation while the *picture*
stays deceptively intact.

### ICASSP calibration + DECISION (2026-07-20)
Estimated acceptance odds, from surveying what ICASSP actually accepts (not what the
guidelines say): **methods framing ~20–30%; dissociation/fairness framing ~10–15%.** Both
below even odds. Area acceptance rate is **~46%** (IEEE SPS S&L TC: ~1,200 submissions,
~550 accepted for ICASSP 2025) — ICASSP is not a lottery, it rewards competent incremental
engineering.

**"Known method, first demonstration in speech" is a proven ICASSP shape**, on small/narrow
data: Wu, Bell & Rajan ICASSP 2024 (LIME → TIMIT phonemes); Pasad, Shi & Livescu ICASSP 2023
(CCA → speech SSL); Cho et al. ICASSP 2023 (linear probing → articulation); Mariotte et al.
ICASSP 2026 (sparse autoencoders → audio models, singing-technique case study). **n=50 is
not disqualifying for this paper type.**

**Why the dissociation framing loses at ICASSP specifically:** there is **no fairness/ethics
topic area** in the ICASSP 2027 CFP (verified against the verbatim topic list), so it routes
to Speech & Language or ML&GenAI and gets judged as engineering with no method and no
baseline. No ICASSP paper with a clean null headline was found. Its natural homes are
Interspeech or Lavender, not ICASSP.

**⚠️ A claim currently in our framing is FALSE:** "probe-dependence has not been shown in
speech." **Zaiem et al., Interspeech 2023** (*Speech SSL Representation Benchmarking: Are We
Doing it Right?*) already showed model rankings flip with the probing head. Narrow the claim
to **attribution-level** (which input features the probe credits) and cite Zaiem prominently
as motivation, not as a gap.

**Highest-leverage 8-week action: a SECOND corpus.** One dataset is the real weakness, not
n=50 — a methods paper demonstrating instability on exactly one 50-speaker corpus is a case
report. TIMIT is the obvious choice (gold hand-labelled phone boundaries, no MFA re-run, and
it is what the closest ICASSP precedent used). Check LDC licensing/institutional access early.

**DECISION: framing (A) with (B)'s data as the case study.** Contribution = weight-space
attribution over frozen SSL embeddings is probe-dependent while input-space IG is stable,
demonstrated on a perceptual-rating task *and* a standard corpus. Keeps the methods
deliverable ICASSP rewards, keeps the interesting dataset as motivation, demotes the null to
a secondary observation. Ethics statement still required.

Not viable: Special Session (proposal deadline 24 Jul 2026, needs six committed papers);
Show & Tell (27 Jan 2027, after notification, nothing to demo). Watch
2027.ieeeicassp.org for a Trustworthy-Speech-Processing-style satellite workshop — if one
appears, framing (B) belongs there and its odds roughly double.

## NWAV framing (researched 2026-07-20)
Different audience, opposite emphasis. **No NWAV abstract using wav2vec2/WavLM/HuBERT for
social meaning could be found** (NWAV 53 full program read end-to-end) — SSL models appear
there only as an *object of critique* (ASR bias, "accent translation"). The established way
to justify ML to variationists is the **auto-coding / measurement-instrument** framing
validated against human coding: Villarreal et al. 2020 (*Lab Phon* 11(6), random forests),
Kendall et al. 2021 (*Frontiers in AI*), Tagliamonte & Baayen 2012 (*LVC* 24(2), the paper
that legitimised trees/forests at this venue), plus fairness auditing (Villarreal 2024).

Perception, by contrast, is **well established**: NWAV 53 had dedicated *Perception*,
*Experimental*, *Persona* and *LGBTQIA+* sessions. Near-neighbour papers: Sulkin,
"Acoustic correlates of gender presentation and (perceived) sexuality"; Brown, Sumner &
Podesva, "Gender Identity and Ideology Shape Perceptions of Masculinity in Male Speech."
Campbell-Kibler (2009, *LVC*) is the theory citation and chaired the LGBTQIA+ session.

## Open TODOs

### Verify IG completeness in `10d_wavlm_ig.py` (code ready, NOT yet run)
The completeness check is now implemented but has **never been executed** — the numbers
currently in `outputs/tables/wavlm_ig_summary.csv` predate it and have no
`completeness_*` columns. Until it runs, 10d's per-phone attributions (and the
"vowels drive perceived, /s/ does not" conclusion that 10e builds on) are unverified.

*Why it matters.* IG approximates a path integral with a midpoint rule over `IG_STEPS=20`
**linearly** spaced points. Linear spacing is perceptually lopsided in amplitude —
alpha=0.025 is −32 dB, alpha=0.975 is −0.2 dB, so half the steps sit within 6 dB of full
volume while the quiet end is sampled once. If WavLM's response moves fast at low
amplitude, 20 steps may under-resolve the integral. Completeness
(`sum_i IG_i == f(x) − f(x0)`) is the diagnostic that settles it empirically.

Two diagnostics are emitted: `completeness_relerr_*` (pass/fail) and the new
`outputs/tables/wavlm_ig_path.csv`, which records `g(alpha) = f(alpha*x)` at every path point
per (speaker, head). Plot `g_alpha` vs `alpha` — near-linear means 20 steps is plenty; sharply
curved near alpha=0 means the quiet end is under-sampled. `path_frac_first_half` in the
summary is the one-number version (0.5 = perfectly linear).

*Run order.*
```powershell
conda activate gayvoice
$env:IG_NSPK=3; python src/10d_wavlm_ig.py    # ~5 min smoke test, reads the verdict line
```
- `completeness_ok = True` (rel.err <= `COMP_TOL` = 5%) → 20 steps is enough; do the full
  50-speaker run (~1.5–2 h) to regenerate the tables with the new columns.
- `completeness_ok = False` → raise `IG_STEPS` (try 50, then 100) until it converges, THEN
  do the full run. The published per-phone numbers would need regenerating.

*Blocker (observed 2026-07-20) — RESOLVED same day.* `py -3.13` stopped resolving: the old
3.13 install was gone (`py -0` listed only 3.14 at `C:\Python314`) and is not recoverable.
The `gayvoice` conda env had been created empty on 2026-07-08 (`conda create --name
gayvoice` with no package specs installs no interpreter at all), then given an unpinned
`python` — which resolved to 3.14, not 3.13. It now runs **Python 3.13.14** with the core
stack installed; use `conda activate gayvoice`.

Still outstanding for this run: **torch is not installed** in that env, so `10*` cannot
execute yet. `pip install -r requirements-all.txt` (repo root) pulls it — ~2 GB. Note that
lockfile resolves **numpy 2.4.6**, not the 2.5.1 in `requirements.txt`, because numba (via
librosa) caps it; install the combined lock rather than layering the tiers.

### Second-window robustness check for 10d (low priority — coverage already cleared)
10d attributes only the **first** `IG_SECONDS=10` s of each ~52 s clip
(`10d_wavlm_ig.py` line ~182, a plain `y_audio[:n]` slice).

*Coverage is NOT the problem* (checked 2026-07-20 against `mfa_textgrids/`): the window holds
7–8 /s/ tokens (~0.79 s), 32–52 vowels and 4–6 /aɪ/ for **every** speaker; zero speakers lack
/s/ or vowels. Because the stimulus is a read passage, the first 10 s is the *same text* for
all 50 — so this slice is better controlled than a random or centred window would be, and
that is worth stating as a methods strength rather than defending as a limitation.

*What remains worth testing:* (a) precision — the window captures only ~24% of each speaker's
/s/ tokens, which adds within-speaker noise (attenuating, so conservative); (b) reading-onset
style — speakers may be slightly more careful in the opening sentences than in habitual
speech. Re-running a subset on a later window would settle both. Needs an offset parameter
alongside `IG_SECONDS`.
