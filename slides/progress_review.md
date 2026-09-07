# Review 1 — Progress Check

**9 slides, ~10 minutes.** A progress review is graded on *direction and evidence of real work*,
not on results. The strongest thing you have is that the hard part — the data — is done and tested,
and that you can already state the finding that makes the evaluation trustworthy.

Speaker notes are in italics. Figures are in `figures/` and were generated from real downloaded data.

---

## 1 — Title

**Computer vision for early wildfire detection: how many minutes, and what are they worth?**

Name · course · date · Review 1 of 3

*Lead with the reframing in one sentence: "This is not a fire-recognition project — it's a project
about how early an alert can fire, and what those minutes are worth in CO₂."*

---

## 2 — Why wildfire is a climate problem, not just a disaster

Warming → higher vapour-pressure deficit, longer fire seasons → drier fuels → larger, more complete
burns → CO₂ + black carbon → more warming.

One qualification up front: in vegetation that regrows, much of the carbon returns over decades. The
*durable* cost concentrates in peat, boreal soils, and permanent vegetation-type conversion.

*Saying the qualification unprompted signals you understand the system rather than reciting it. It
also pre-empts the most likely challenge from a climate examiner.*

---

## 3 — The gap in existing work

Most wildfire CV projects classify photos of **flames**.

By the time flames are photographable, the fire is not early — the information arrives after the
moment it could have changed the outcome. Fire lookouts watch for **smoke**.

**→ Figure 1** (`fig1_sequence_over_time.png`) — one real fire at t = −10, 0, +5, +20 min.

*Point at t = +5: a faint smudge on the ridge. Then t = +20: unmistakable. "Everything useful
happens in the gap between these two panels."*

---

## 4 — The research question

> **How many minutes after a plume first becomes visible does a detector fire, at a false-alarm rate
> low enough that an operator leaves it switched on?**

Not "what is the accuracy." Accuracy weights a frame at +1 min and one at +39 min equally; for early
detection they could not be less equal.

*This slide is the spine of the project. If they remember one slide, make it this one.*

---

## 5 — Data: why this dataset makes the question answerable

**FIgLib** (HPWREN, UC San Diego): 523 sequences from fixed fire-lookout cameras. Each is one real
fire — 40 frames before the plume appeared, 40 after, one per minute.

Every filename encodes its offset from that moment:

```
20180719_Skyline_sp-n-mobo-c/1465067280_+01680.jpg   →   +1680 s = +28 min
```

**Status: acquired.** Downloader written and tested; 523 sequences enumerated, download verified.

*"The ground-truth clock is free — that's why this dataset and not a Kaggle fire/no-fire set."*

---

## 6 — The finding that decides whether any result will mean anything

523 sequences, but only **338 distinct fires** — one fire is filmed by up to **9 cameras at once**,
and consecutive frames are one minute apart.

Split on frame or on sequence → near-identical views of the same fire land in both train and test →
AUROC ~0.99, meaning nothing.

**→ Everything is split by fire date, guarded by an assertion and a test that verifies the guard
catches injected leakage.**

*This is your strongest slide. It shows you found a non-obvious trap in the data by looking at it
rather than by downloading and training blindly. Expect a question; welcome it.*

---

## 7 — Method: why the standard recipe cannot be early, and the fix

**→ Figure 3** (`fig3_why_224_fails.png`) — the same frame at native / 448 / 224.

A 60 px plume survives as ~9 px at 448 and ~4 px at 224. It dies in the first pooling layer.

**The fix — the cameras never move, so use that.** For the frame at *t*, stack three channels:
`gray(t)` · `gray(t−4 min)` · `|difference|`. Drops into ImageNet weights unchanged.

**→ Figure 2** (`fig2_temporal_stack.png`)

Two arms, identical but for the input transform: **A** single-frame (expected late) vs **B**
temporal stack (hypothesis: earlier at equal false-alarm rate).

*Honest observation worth volunteering — see slide 9.*

---

## 8 — Where the climate number comes from

Area of a radially spreading fire ∝ **t²** → halving the detection delay quarters the fire.

Emissions via Seiler–Crutzen, the basis of inventories like GFED:

```
E  =  A × B × C × EF
      area × fuel load × combustion completeness × emission factor
```

Deliverable is a **sensitivity table** across spread rate and detection delay — not a single number.

**The bound I will state:** emissions are dominated by fires that escape initial attack, so early
detection shifts a *probability*, it does not bank the tonnage. The table is an upper bound per
incident.

*Stating the limit of your own headline number before anyone asks is the single most credible thing
you can do in a climate review.*

---

## 9 — Status, risk, and plan

**Done**
- Dataset acquired; downloader written and tested against the live archive
- Full pipeline built and validated — **17 automated checks pass** (leakage guard, frame pairing,
  operating-point threshold, detection-timing rule, emissions scaling)
- Method designed and justified; three method figures generated from real data

**Next (→ Review 2)**
- Train both arms, 3 seeds each; produce the detection-probability-vs-time curve
- Grad-CAM + failure gallery; cross-dataset false-alarm test on a different camera network

**Then (→ Review 3)**
- Emissions table with cited coefficients; full report and final presentation

**Open risk, stated honestly:** in the difference channel, early smoke is *not* the largest change in
the scene — drifting cloud is (visible in Figure 2). So arm B's advantage is a hypothesis, not a
foregone conclusion. If it doesn't hold, that is a reportable result and the comparison is still
sound.

*Ending on a named risk rather than a promise reads as competence. It also inoculates you: if arm B
underperforms at Review 2, you predicted it.*

---

## Likely questions

| Question | Answer |
|---|---|
| "Why not object detection / YOLO?" | Bounding boxes exist for only part of FIgLib, and the question is *when* an alert fires, not where the plume is. Classification plus a timing protocol answers it directly. |
| "Have you got results yet?" | Not yet — that's Review 2. The pipeline is built and tested; what remains is compute, not design. |
| "Why only ~30 sequences?" | Free-tier GPU budget. Split by fire, that is still ~30 distinct fires; the seed spread will show how much of the result is noise. |
| "Is 1% false positives good?" | Per frame, per camera, per minute — multiply it out for a 100-camera network and it's a lot. That's why there's a persistence rule, and it stays a stated limitation. |
| "Isn't this just SmokeyNet?" | SmokeyNet is the reference point and does it better with a bigger model. The contribution here is the evaluation framing — time-to-detection at a fixed false-alarm rate — and the emissions link. |
| "What if the deep model doesn't beat the classical baseline?" | Then I report that. It's a control precisely so the comparison is meaningful either way. |
