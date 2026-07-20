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
"we ran WavLM on a new dataset" is not a contribution. The defensible novelty is the
**10c → 10d arc**: weight-space attribution over frozen SSL embeddings is *not*
head-invariant (ridge and enet predict comparably, ρ=0.73/0.67, yet disagree at r=0.45 and
give opposite per-phone stories), and input-space IG resolves it (r=0.61, consistent signs).

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
