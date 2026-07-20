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
`requirements-all.txt` adds the extraction + WavLM stacks. Verified 2026-07-20 on the work
machine: Python 3.13.14, torch 2.13.0+cpu, transformers 5.14.1, librosa 0.11.0, no CUDA.

> **Shell note:** examples here use **bash** (Git Bash / MINGW64), so env vars are
> `IG_NSPK=3 python src/...`. The PowerShell form `$env:IG_NSPK=3` will not work in bash.

## ⚠️ Every analysis choice must be justified in `docs/DECISIONS.md`
That file is the methodological decision record — *why* each choice was made, in the format
**Decision · Rationale · Alternatives considered · Status**, with statuses **SETTLED** /
**PENDING** / **REVISED**. It is the source the paper's methods section is written from.

Working rules, and they are not optional:
1. **Read the relevant section before touching analysis code.** Sections: A statistics ·
   B forced alignment · C segmental measurement · D pre-registration · E SSL & attribution.
   Many "obvious improvements" were already considered and rejected there for stated reasons
   (e.g. why occlusion is wrong for `10c`, why BH-FDR is applied within-target, why
   augmentation may never touch inference).
2. **Any new or changed methodological choice gets an entry**, in the same format, in the
   same commit as the code. A parameter, threshold, filter, normalization, estimator, or
   statistical test with no entry is not finished work.
3. **Never silently overturn a SETTLED entry.** Mark it **REVISED**, keep the original text,
   and state the evidence that changed it — the paper trail is the point. See E7 for a worked
   example: the "input-space IG escapes probe arbitrariness" rationale was withdrawn on a
   proof, and both the claim and its refutation are preserved.
4. **Record what is unverified.** If a result depends on something not yet run, the entry
   says PENDING and says so plainly (see E8: the IG completeness check is implemented but has
   never been executed, so the per-phone numbers are provisional).

This repo is **public**. `DECISIONS.md` is research output and should read as such; venue
positioning and strategy belong in the gitignored `*.local.md` files, never here.

## ⇒ Active work: ICASSP 2027 paper (deadline 16 Sept 2026)
The working plan lives in **`docs/ICASSP_2027_PLAN.local.md`** and the venue notes in
**`docs/NOTES.local.md`** — both gitignored (`*.local.md`) and copied between machines by
hand, because this repo is public and they are planning artifacts, not research output. If
they are absent on this machine, ask for them rather than assuming no plan exists.

One-line version: *attribution over frozen SSL speech probes is unreliable in both weight
space and input space; we quantify the spread and give a Rashomon-set protocol that yields
attributions surviving it.* The gay-voice data is the case study.

**Next action:** `IG_NSPK=3 python src/10d_wavlm_ig.py` — the IG completeness check is
written but has **never been run**, and it gates everything downstream. See the
`10d_wavlm_ig.py` module docstring for the method caveats, including why IG does **not**
escape the probe arbitrariness.

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

## Sensitive data
`clean_wavs/` is 611 MB of identifiable voice recordings joinable to self-reported Kinsey
scores — a protected attribute on a small, re-identifiable population. **Do not upload it to
peer-to-peer GPU marketplaces** (Vast.ai and similar). Institutional HPC or a major cloud
provider only. IRB/consent paperwork is **not in this repo** and needs locating; see
`docs/ICASSP_2027_PLAN.md` §8.

## NWAV framing (researched 2026-07-20)
NWAV 54 (Montréal, 22–24 Oct 2026) submission is pending; notification is not separately
published, the CFP says only that the program appears Aug/Sept 2026.

Different audience from ICASSP, opposite emphasis. **No NWAV abstract using
wav2vec2/WavLM/HuBERT for social meaning could be found** (NWAV 53 full program read
end-to-end) — SSL models appear there only as an *object of critique* (ASR bias, "accent
translation"). The established way to justify ML to variationists is the **auto-coding /
measurement-instrument** framing validated against human coding: Villarreal et al. 2020
(*Lab Phon* 11(6), random forests), Kendall et al. 2021 (*Frontiers in AI*), Tagliamonte &
Baayen 2012 (*LVC* 24(2), the paper that legitimised trees/forests at this venue), plus
fairness auditing (Villarreal 2024).

Perception, by contrast, is **well established**: NWAV 53 had dedicated *Perception*,
*Experimental*, *Persona* and *LGBTQIA+* sessions. Near-neighbour papers: Sulkin,
"Acoustic correlates of gender presentation and (perceived) sexuality"; Brown, Sumner &
Podesva, "Gender Identity and Ideology Shape Perceptions of Masculinity in Male Speech."
Campbell-Kibler (2009, *LVC*) is the theory citation and chaired the LGBTQIA+ session.

The perceived-vs-actual dissociation is the primary result for the NWAV and Lavender
write-ups; in the ICASSP paper it is a secondary observation (see `docs/ICASSP_2027_PLAN.md`
§3). Prior art specific to that framing — Kachel et al. 2018, El-Tawil et al. 2026, and the
construct-validity literature — is in `docs/NOTES.local.md`, which is gitignored and copied
between machines by hand.
