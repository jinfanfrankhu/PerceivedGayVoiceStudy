# GayStudy — "Gay Voice" Research Program

Research on the perception and acoustics of "gay voice" in Gen Z males. This repo
combines two studies that began as separate repos, each with its own venue track
(see **Venue status** below):

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

## Venue status (as of 2026-07-20)
- **F24** — Lavender Languages and Linguistics Conference: **accepted**.
- **F25** — NWAV 54 (Montréal, 22–24 Oct 2026): **pending**. Notification is not
  separately published; the CFP says only that the program appears **Aug/Sept 2026**.
- **F25** — **ICASSP 2027: full paper deadline 16 Sept 2026** (AoE), Toronto 16–21 May
  2027. 4 pages content + a 5th page for references/ethics only. Single-anonymous
  (include author list). Notification 13 Jan 2027.
  See `F25/CLAUDE.md` → *ICASSP framing* for the contribution/ethics requirements.

The two tracks want different headline framings from the same results, so check
which one you are writing for before shaping a claim:
- *NWAV (variationist sociolinguistics)* — the finding is the perceived-vs-actual
  dissociation: voice predicts what listeners hear, not what speakers are.
- *ICASSP (signal processing)* — the finding is methodological: weight-space
  attribution over frozen SSL embeddings is head-dependent (`10c`), and
  input-space integrated gradients resolves it (`10d`). The "gay voice" data is
  the case study that exposes it.

## Paper framing
F24 (perception) and F25 (acoustics) feed one narrative: how voice drives
perceived sexuality, and how well that perception tracks reality.

## Documentation system
Hierarchical: this root file always loads and orients across both studies;
each sub-project's own `CLAUDE.md`/`README.md` carries the working detail and
loads when you work in that folder. Keep this root file thin — put project-
specific instructions in the relevant sub-project doc.

**Analysis choices must be justified, not just made.** F25 keeps a methodological
decision record at `F25/docs/DECISIONS.md` (**Decision · Rationale · Alternatives ·
Status**). Read the relevant section before changing analysis code, and add an entry
in the same commit as any new or changed methodological choice. Full rules in
`F25/CLAUDE.md`. This repo is **public** — decision records are research output;
venue strategy belongs in the gitignored `*.local.md` files.

## Environment
Run the scientific stack from a **conda env built from the repo-root requirements**
(Python 3.11+). The env name varies by machine — check which conda envs exist rather
than assuming a name, and don't rely on the system `python`, which lacks the stack.
Dependencies are declared in `requirements*.in` at the repo root and locked in the
generated `requirements*.txt`; install **`requirements-all.txt`** — the single
consistent set for one env that runs everything — rather than layering the tiers.
Compile/install commands are in `requirements.in`.

F25 scripts resolve paths relative to themselves, so they run from any working
directory.
