# Briefing: "Gay Voice" Perception Study — for Patrick (presenter)

## The One Story (memorize this)

**Listeners strongly agree with each other about who "sounds gay" — but that shared
perception barely tracks who actually is gay.**

The "gay voice" is real as a *shared perceptual/social object*, and weak as a *detector of
actual orientation*. It's a high-consensus, low-validity stereotype.

The rhetorical arc of the talk mirrors how the analysis unfolded:
1. **First glance: nothing.** Accuracy is weak and not statistically significant.
2. **Then: the twist.** Listeners agree *with each other* remarkably well.
3. **PCA names the thing.** A single perceptual dimension (PC1) essentially *is* "perceived
   gayness" — but it only weakly reflects real orientation.
4. **Payoff.** The gay voice lives in listener *consensus*, not in listener *accuracy*.

---

## What makes this study different (say this early)

- **Attraction scale, not a binary.** Speakers and the perception task use a 1–5
  Kinsey-style attraction scale (1 = fully attracted to women … 5 = fully attracted to men),
  not a gay/straight label. Prior perception work collapses everyone to two boxes and throws
  away the middle. We keep it.
- **We separate consensus from accuracy.** Most perception studies conflate "did listeners
  agree" with "were listeners right." Our design + PCA pulls those apart — and they turn out
  to tell *opposite* stories.
- **Framing is queer-linguistic, not forensic.** This is about the social construction of a
  linguistic stereotype and how it misfires — NOT "can we detect gay people from audio." Keep
  the framing there; it's both more honest and the right fit for this venue.

---

## The Numbers (have these ready)

Sample: **85 listeners, 36 speakers** (Gen Z men). Missing data trivial (0.4%).

| Claim | Stat | Plot |
|---|---|---|
| Overall accuracy is weak / n.s. | Spearman ρ = **0.28**, p = 0.10 | 1A |
| Listeners barely detect above chance | d′ ≈ **0.11** | 8A |
| Pooled consensus is highly reliable | ICC(2,k) = **0.96** | 3A |
| …but individual agreement is modest | ICC(2,1) = **0.24** | 3A |
| A single dimension = "perceived gayness" | PC1 = **31%** var; PC1 vs mean rating **r=0.998** | 5D2 |
| …that dimension weakly reflects reality | PC1 vs actual orientation **r=0.27**, R²=0.07 | 5D1 |
| No demographic rescues accuracy | gender, orientation, familiarity, religiosity all n.s. | 1B–1E, 2B |

**Misperception examples (the stereotype misfiring — powerful on a slide):**
- Actually gay, heard as straight: **XX34** (actual 5.0, rated 1.95), WCL31, QI36.
- Actually straight, heard as gay: **PP3** (actual 2.2, rated 4.5), PL23 (actual 1.4, rated 3.6).

---

## Slide-by-Slide Talk (~15 min)

1. **Title / framing.** "How well does listener perception of the 'gay voice' track actual
   self-reported attraction — and what does the gap tell us?"
2. **Background & gap.** Prior work = binary labels, conflates agreement with accuracy. We use
   an attraction continuum and separate the two.
3. **Methods.** 85 listeners rate 36 male speakers on 1–5 attraction; speakers have
   self-reported attraction. Web-based (gayvoice.github.io). Note anonymization/consent.
4. **Result 1 — accuracy is weak.** Show **1A** scatter (ρ=0.28, n.s.). Beat: "So — case
   closed? People can't hear it?" *(pause)*
5. **Result 2 — but they agree.** Show **3A** inter-rater heatmap. ICC average-measures 0.96,
   single 0.24. "They're consistently perceiving *something*."
6. **Result 3 — PCA names it.** Show **5D2** (PC1 vs mean rating, r=0.998): PC1 *is* the
   consensus "perceived gayness" axis. Then **5D1** (PC1 vs actual, r=0.27): but it only
   weakly maps onto real orientation.
7. **The misfires.** A couple of speakers who break the stereotype (XX34, PP3). This is the
   emotional/analytic core — the stereotype is applied consistently and wrongly.
8. **No demographic saves it.** Briefly: listener gender/orientation/familiarity/religiosity
   don't significantly change accuracy (1B–1E, 2B).
9. **Interpretation.** The gay voice is a shared social-perceptual construct with high
   inter-subjective reality and low criterion validity. Consensus ≠ accuracy.
10. **Limitations & future.** (see below) + thank you.

*If it's a poster, not a talk: same spine — Accuracy panel (1A) → Agreement panel (3A) →
PCA panel (5D1/5D2) → Misfires table → Interpretation box. The two PCA plots are the poster's
centerpiece.*

---

## Plots worth putting on screen (all in F24/Dataplots/)

- **1A** overall accuracy scatter — the "weak accuracy" beat
- **3A** inter-rater reliability heatmap — the "but they agree" beat
- **5D2** PC1 vs mean rating (r=0.998) — "PC1 IS perceived gayness"
- **5D1** PC1 vs actual orientation (r=0.27) — "but it's not accurate"
- **5B / speaker readability** — which speakers are (mis)read
- Optional: **8A/8B** signal detection, **1E** accuracy across listener groups

---

## Landmines (do NOT overstate these)

1. **ICC.** Don't say "listeners agreed 96%." Say: *aggregate* perceptual signal is highly
   reliable (ICC average-measures 0.96); *individual* agreement is modest (0.24). The 0.96 is
   high partly *because* it pools 85 raters.
2. **"Accuracy" wording.** ρ=0.28 is **not significant** (p=0.10, n=36 speakers). Frame as
   "weak and not statistically reliable," never "listeners were accurate."
3. **Small n / power.** 36 speakers is modest. The non-significant accuracy could be
   low power. Be candid: our *positive* claim (high consensus) is robust; our accuracy claim
   is "we did not find reliable accuracy," not "we proved people can't."
4. **Causation / acoustics.** This study is perception only. We do NOT here show *which*
   acoustic features drive the judgment. (There's a separate F25 acoustic dataset pointing at
   pitch/F0 range and voicing, but it's a *different* 50-speaker sample — only mention if
   asked, and flag that it's a separate study.)
5. **Generalizability.** All Gen Z men; listeners skew a convenience sample. Don't generalize
   to all speakers/listeners.

---

## Anticipated Q&A

- **"Isn't ρ=0.28 actually a real effect, just underpowered?"** Possibly — that's why we
  frame it as "no reliable accuracy," not "no accuracy." The headline result doesn't depend on
  it; it depends on the consensus/accuracy *gap*, which holds regardless.
- **"What acoustic features drive perception?"** Not tested in this study. A companion acoustic
  analysis (separate sample) points to F0/pitch range and voicing measures, consistent with the
  literature, but we're not claiming it here.
- **"Why an attraction scale instead of gay/straight?"** Because real attraction is continuous
  and the binary discards the people who most test the stereotype (the middle of the scale).
- **"Could listeners be picking up on something other than orientation?"** Yes — that's
  arguably the point. They're picking up a *stereotyped voice profile* that only partly
  coincides with orientation. Hence high consensus, low validity.
- **"Did listener identity matter — e.g., are gay listeners better?"** No significant
  moderation by gender, orientation, familiarity, or religiosity (n's are small for some
  groups, esp. gay listeners N=2 — underpowered, don't lean on it).

---

## If you have 60 seconds to prep, read only this

85 listeners rated 36 men's voices on a 1–5 gay-attraction scale. **They agreed with each
other a lot (ICC avg 0.96) but were barely accurate (ρ=0.28 n.s., d′≈0.11).** PCA found one
dominant dimension that *is* the shared "perceived gayness" (r=0.998 with mean rating) but only
weakly matches real orientation (r=0.27). **Conclusion: the "gay voice" is a real, widely
shared perceptual stereotype that's a poor detector of who's actually gay.** Show plots 1A, 3A,
5D2, 5D1 in that order. Don't overclaim accuracy; don't say "96% agreement."
