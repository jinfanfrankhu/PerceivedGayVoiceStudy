# GayStudy — "Gay Voice" Research Program

Research on the perception and acoustics of "gay voice" in Gen Z males, aimed at
the **Lavender Languages and Linguistics Conference**. This repo combines two
studies that began as separate repos:

- **F24/** — *Listener perception study.* 85 listeners rated 36 speakers on a 1–5
  perceived-sexuality scale. Sociolinguistic angle: listener accuracy and how
  listener demographics shape perception. Toolchain: Python + R, MFA forced
  alignment, FastTrack/Praat formant tracking.
  Docs: `F24/PATRICK_BRIEFING.md`, `F24/analysis_summary.md`.

- **F25/** — *Acoustic + perceptual classification study.* 50 speakers with three
  data layers on the same people: 88 eGeMAPS acoustic features, actual orientation
  (Kinsey + self-ID), and 3,450 listener ratings (perceived gayness) from its own
  web platform. Toolchain: Python (pandas/scikit-learn/matplotlib).
  Docs: `F25/README.md`, `F25/CLAUDE.md`.

## Paper framing
F24 (perception) is the focus for the paper; F25 (acoustics) is supporting
evidence for *why* listeners judge as they do. Both feed one narrative: how
voice drives perceived sexuality, and how well that perception tracks reality.

## Documentation system
Hierarchical: this root file always loads and orients across both studies;
each sub-project's own `CLAUDE.md`/`README.md` carries the working detail and
loads when you work in that folder. Keep this root file thin — put project-
specific instructions in the relevant sub-project doc.

## Environment
Use **`py -3.13`** for the scientific stack (pandas, scipy, scikit-learn,
matplotlib, seaborn); the default `python` (3.14) lacks it. F25 scripts resolve
paths relative to themselves, so they run from any working directory.
