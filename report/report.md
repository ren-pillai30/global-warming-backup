# Early Wildfire Smoke Detection: How Many Minutes, and What Are They Worth?

*[Your name] — [Course] — [Date]*

> **How to use this file.** The argument is already laid out; the evidence is not. Every
> `⟨…⟩` is a slot to fill from `results_summary.csv` and the figures the notebook writes to
> `figures/`. Every **[CITE]** is a coefficient or claim that needs a real reference — those are the
> easiest marks to lose in a theory course and the easiest to defend once cited. Delete this box
> before submitting.

---

## Abstract

⟨150 words, written last.⟩ State the problem (early detection, not fire recognition), the method
(temporal stacking on fixed-camera imagery), the headline result (median ⟨X⟩ minutes to detection at
a 1% false-alarm rate, versus ⟨Y⟩ for a single-frame baseline), and the climate framing (what those
minutes are worth in tonnes of CO₂, and why that figure is an upper bound).

---

## 1. Motivation: fire as a climate feedback

Wildfire is one of the clearer self-reinforcing loops in the climate system. Warming raises
vapour-pressure deficit and lengthens fire seasons, drying fuels; drier fuels ignite more readily and
burn more completely; the resulting fires release CO₂ and black carbon, driving further warming.
**[CITE — Abatzoglou & Williams on anthropogenic fuel aridity, or equivalent.]**

One qualification, stated up front because it governs how much any of this counts: in temperate
forest and chaparral that regrows, much of the released carbon is reabsorbed over years to decades.
The *durable* climate cost concentrates in peat and boreal soils, and in fires that permanently
convert vegetation type. Treating every tonne of fire CO₂ as equivalent to a tonne of fossil CO₂
overstates the case, and this report does not. **[CITE]**

If the loop is real, the intervention with the best leverage per unit of effort is **time**: a fire
caught in its first minutes is a fire that may never become a large emitter.

## 2. Problem framing: why "early" means smoke

Most published student and hobbyist work on wildfire computer vision classifies photographs of
**flames**. By the time flames are large enough to photograph, the fire is not early — the
information arrives after the moment when it could have changed the outcome.

Fire lookouts watch for **smoke**, because the plume is visible minutes before anything else. So the
question this report asks is not whether a CNN can recognise a fire, but:

> **How many minutes after a plume first becomes visible does a detector fire, at a false-alarm rate
> low enough for an operator to leave it switched on?**

Fixed-camera networks that would host such a detector already exist — HPWREN and ALERTWildfire in
California, Pyronear in France — and are largely monitored by humans. The bottleneck is operator
attention across hundreds of simultaneous feeds, not sensor coverage. **[CITE]**

## 3. Data

**FIgLib** (Fire Ignition Library, HPWREN / UC San Diego): 523 sequences from fixed fire-lookout
cameras in Southern California. Each sequence is one real fire — 40 frames before the plume became
visible, the ignition frame, and 40 frames after, one per minute. Frame resolution is not uniform:
older sequences are 2048×1536, newer ones 3072×2048. The pipeline resizes both to a common input
size, so the only consequence is that a plume of a given physical size occupies fewer pixels in the
older cameras. **[CITE — Dewangan et al., arXiv:2112.08598.]**

The property that makes this project possible: **every filename encodes its offset in seconds from
the moment the plume became visible.**

```
20180719_Skyline_sp-n-mobo-c/1465067280_+01680.jpg   →   +1680 s = +28 min
```

That signed offset is a ground-truth clock, and it is what turns "accuracy" into
**time-to-detection**. This report used ⟨N⟩ sequences covering ⟨M⟩ distinct fires.

### 3.1 The split, and why it decides whether the numbers mean anything

Two properties of FIgLib will silently destroy an evaluation:

1. Consecutive frames are one minute apart from a fixed camera — they are near-duplicates.
2. **The 523 sequences cover only 338 distinct fires.** A single fire is often filmed by several
   cameras at once; one appears on nine.

Splitting at the frame level, or even at the sequence level, therefore scatters near-identical views
of the *same fire* across train and test. The model sees each test frame's neighbours during
training, AUROC climbs toward 0.99, and the number means nothing.

Every result here is split on the **fire date**, so all camera angles and all frames of a given fire
land entirely in one split. The split is guarded by an assertion in the code and by a test
(`test_pipeline.py`) that also verifies the guard rejects injected leakage.

*This is worth a paragraph in a report because it is the single most common way results like these
are inflated, and because the correction costs one line.*

## 4. Method

### 4.1 Why the obvious approach cannot be early

The default recipe — resize to 224×224, fine-tune a pretrained CNN — is *structurally* unable to be
early on this task. A FIgLib frame is 2048×1536 or 3072×2048; an early plume may span 60 px.
Downscaling to 224 leaves roughly 4–7 px of low-contrast haze, which does not survive the first
pooling layer. Such a model can learn "this scene is on fire" long after ignition, and score well on frame-level accuracy,
while being useless at the only time that matters.

### 4.2 Two fixes

**Resolution.** 448×448 rather than 224×224.

**Temporal stacking.** The cameras are *fixed*, so the background is free information. For the frame
at time *t* we construct a three-channel tensor:

| Channel | Content |
|---|---|
| 0 | `gray(t)` — the current view |
| 1 | `gray(t − 4 min)` — the same scene, before |
| 2 | `\|gray(t) − gray(t − 4 min)\|` — what changed |

This drops into ImageNet-pretrained weights unchanged: it is still a three-channel image tensor. A
plume nearly invisible in channel 0 is conspicuous in channel 2, because it is the only thing in the
scene that appeared. See Figure 2.

Two implementation notes worth stating, because both are places the result could have been quietly
wrong:

- All imagery is converted to **grayscale**. FIgLib mixes monochrome and colour cameras; grayscale
  makes them interchangeable at negligible cost, since smoke is essentially achromatic.
- The reference frame is matched **by timestamp, not by index**. Frame spacing is nominally 60 s but
  drifts between 58 s and 62 s, and some sequences drop frames outright (one observed gap was 121 s).
  Counting back four rows would silently pair the wrong frames wherever a frame is missing.

Stacking is not free of confounders — clouds drift, shadows move, cameras jitter — which is why a
learned model is still needed rather than a threshold on the difference image. Section 6.3 measures
exactly how much the learning is worth.

### 4.3 Arms

| Arm | Input | Role |
|---|---|---|
| A | `gray(t)` replicated ×3 | baseline; expected to be late |
| B | the temporal stack | hypothesis: earlier detection at equal false-alarm rate |
| C | difference → blur → threshold → largest blob | no-learning control |

Arms A and B share backbone (ResNet-18, ImageNet-pretrained), schedule, data and seeds. **The only
difference is the input transform**, which is what makes the comparison a test of the idea rather
than of tuning effort. Three seeds per arm; every number below is mean ± std.

## 5. Evaluation protocol

A detector is only useful at a false-alarm rate an operator will tolerate; a network firing a few
false alerts per camera per day gets switched off by the humans receiving them. So the operating
point is fixed *first*:

> **Threshold = the value giving a 1% false-positive rate on pre-ignition frames of the validation
> fires.** Chosen on validation, applied unchanged to test, never tuned on test.

**Time to detection.** Walk each test fire forward from ignition; alert at the first minute where the
score clears threshold for **2 consecutive frames**. Real alerting systems use a persistence rule; a
single-frame trigger is a false-alarm machine. Fires not detected within 40 minutes are **censored
and reported separately** — silently dropping misses is the easiest way to flatter a detector.

**Why not just report accuracy or AUROC?** Both weight a frame at t = +1 min and one at t = +39 min
equally. For early detection they could not be less equal. Frame-level metrics appear below for
comparability with prior work, but the timing results are the answer to the question.

## 6. Results

### 6.1 Frame-level

⟨Table from `results_summary.csv`: AUROC, AP, realised test FPR and recall at the operating point,
for arms A and B, mean ± std over seeds.⟩

### 6.2 Time to detection — the headline

⟨Table: median time to detection and fraction of fires detected within 40 min, arms A and B.⟩

![Detection probability versus minutes since ignition](../figures/fig3_detection_vs_time.png)

⟨Interpret the *left edge* of the curve specifically. A method that only rises at +20 min reports
fires people already know about. State the median for each arm and whether the difference exceeds
the seed-to-seed spread — if it does not, say so.⟩

### 6.3 What the learned model is worth

⟨Compare arm B against the classical control. If the gap is small, report that plainly: a negative
result honestly reported is worth more than a flattering one, and it bears directly on the emissions
argument in §7.5, since the CNN is the part that needs a GPU.⟩

### 6.4 Where the model looks, and where it fails

![Grad-CAM overlays](../figures/fig4_gradcam.png)

⟨The top row should show attention on the plume. **The bottom row is the more valuable half**: the
pre-ignition frames the model scored highest. Name what they have in common — cloud bank, haze,
camera motion, sun glare — because that is this method's real failure mode, and naming it before the
examiner does is worth more than the top row.⟩

### 6.5 Does it generalise, or did it memorise these cameras?

⟨False-alarm rate on the `cloud` class of `sagecontinuum/smokedataset`, a different camera network,
at the unchanged FIgLib operating point. A high rate is the expected and reportable result: it is
the honest measure of how far this travels off its training distribution.⟩

A caveat that is itself a finding: **arm B cannot take this test at all.** It needs a temporal pair,
and a single photograph cannot provide one. Its accuracy therefore comes at a cost in deployability —
it suits fixed tower networks and cannot score a crowd-sourced image.

## 7. From minutes to tonnes of CO₂

### 7.1 Minutes to hectares

A fire spreading radially at rate ROS covers area `π(ROS·t)²`. The **t²** is the whole argument:
area grows with the square of elapsed time, so halving the detection delay quarters the fire that
has to be fought.

### 7.2 Hectares to tonnes

The Seiler–Crutzen formulation, still the basis of inventories such as GFED **[CITE]**:

```
E = A × B × C × EF
```

| Term | Meaning | Value used | Source |
|---|---|---|---|
| A | area burned (ha) | from §7.1 | — |
| B | fuel load (t dry matter/ha) | ⟨30⟩ | **[CITE — chaparral fuel loading]** |
| C | combustion completeness | ⟨0.8⟩ | **[CITE]** |
| EF | CO₂ emission factor (g/kg DM) | ⟨1650⟩ | **[CITE — Andreae & Merlet 2001; Akagi et al. 2011]** |

### 7.3 Sensitivity, not a headline number

⟨Table from the notebook: rows = spread rate (1 / 5 / 20 m·min⁻¹, calm to wind-driven), columns =
detection delay (10 / 30 / 60 min), cells = t CO₂.⟩

The spread across that table — roughly four orders of magnitude — **is** the finding. A single
headline number would be false precision. What survives is the *shape*: the 10→60 minute column is a
36× difference at every spread rate.

### 7.4 The caveat that should be the loudest sentence in this report

**The table is an upper bound on a per-incident basis, not an expected saving.** Two reasons:

1. **Emissions are dominated by the tail.** Most fire CO₂ comes from the minority of fires that
   escape initial attack and burn for days. Early detection does not deterministically bank the
   tonnage above; it raises the *probability* that initial attack succeeds, and the expected benefit
   is that probability shift multiplied by the avoided tail. Estimating the shift requires
   suppression-outcome data this project does not have. **[CITE — initial-attack success statistics.]**
2. **Detection is not response.** Minutes convert to hectares only if crews are available and can
   reach the fire. In simultaneous-ignition wind events — precisely when it matters most — often they
   cannot.

### 7.5 The method's own carbon cost

⟨Training energy and its CO₂, versus the emissions from a single hectare. The ratio is
overwhelmingly favourable; the point is that a project proposing compute as a climate intervention
should verify that rather than assume it.⟩

## 8. Limitations

- **Night.** FIgLib is daylight. A visible-band smoke detector is blind at night, when a meaningful
  share of ignitions occur. Operational networks pair it with IR.
- **Geography.** One camera network, Southern California chaparral. §6.5 measures how badly that
  assumption travels.
- **Arm B needs a fixed camera with history**, so it cannot score arbitrary imagery.
- **The label is "plume visible", not ignition.** True ignition precedes FIgLib's zero point by an
  unknown margin, so every latency here is measured from a generous starting line.
- **Test-set size.** ⟨K⟩ held-out fires: the median time-to-detection carries real uncertainty,
  which is why every number is reported with a spread over seeds.
- **Detection is not suppression.** §7.4.

## 9. Conclusion

⟨Three or four sentences. What was measured, what the timing result was, and — most importantly —
what the honest bound on the climate claim is. Resist ending on the biggest number in the report.⟩

## References

⟨Fill in. At minimum: Dewangan et al. 2021 (FIgLib/SmokeyNet); Seiler & Crutzen 1980; Andreae &
Merlet 2001 or Akagi et al. 2011; an attribution paper for fire–climate feedback; a source for
chaparral fuel loading and combustion completeness; the `sagecontinuum/smokedataset` card.⟩
