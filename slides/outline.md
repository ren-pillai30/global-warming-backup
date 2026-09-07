# Presentation outline — 12 slides

One message per slide, stated as the slide title. If a slide needs two sentences of explanation to
land, it is two slides.

---

**1 — Title**
*Early Wildfire Smoke Detection: how many minutes, and what are they worth?*
Name, course, date. Background: one FIgLib frame with a visible plume.

**2 — Fire is a climate feedback loop, not just a disaster**
Warming → drier fuels → larger fires → CO₂ + black carbon → warming.
One qualification you will return to: regrowth reabsorbs much of it, so the durable cost is
concentrated in peat, boreal, and vegetation-type conversion.

**3 — Most wildfire CV projects detect the wrong thing**
Side by side: a stock "flames" image versus a real plume at t = +3 min.
*By the time there are flames to photograph, it isn't early.* Lookouts watch for smoke.

**4 — So the question is a timing question**
> How many minutes after a plume becomes visible does the detector fire, at a false-alarm rate low
> enough to leave switched on?

**5 — The dataset makes that answerable**
FIgLib: 523 sequences, fixed lookout cameras, 40 min before → 40 min after.
Show the filename decoding to `+1680 s = +28 min`. *The ground-truth clock is free.*

**6 — Why the obvious approach cannot be early**
Figure 1: the plume at t = −10, 0, +5, +20.
2048 px frame, 60 px plume, resize to 224 → ~7 px of haze. It dies in the first pooling layer.

**7 — The fix: the camera never moves, so use that**
Figure 2 (the temporal stack). Three channels: now, 4 minutes ago, the difference.
Drops into ImageNet weights unchanged. The plume is the only thing that appeared.

**8 — The one slide that makes the evaluation trustworthy**
523 sequences → **338 fires**; one fire on up to 9 cameras; frames 1 minute apart.
Split by fire, not by frame — otherwise AUROC → 0.99 and means nothing.
*Expect a question here. It is the slide that separates this from a tutorial.*

**9 — Results: detection probability vs. minutes since ignition**
Figure 3. Point at the **left edge**. Median time to detection, both arms, at 1% false alarms.
Say whether the gap exceeds the seed spread.

**10 — Where it looks, and where it fails**
Figure 4. Top row: attention on the plume. **Bottom row: its worst false alarms** — name the common
factor. Plus the cross-dataset cloud false-alarm rate: how far it travels off its training network.

**11 — What the minutes are worth**
Area ∝ t², `E = A × B × C × EF`, the sensitivity table.
The 10→60 min column is 36× at every spread rate.
*Show the whole table. A single number would be false precision.*

**12 — What this does not show**
Upper bound per incident, not an expected saving — emissions are dominated by fires that escape
initial attack, so early detection shifts a *probability*.
Detection ≠ response. Daylight only. One camera network.
*Close here, not on the biggest number. It is the most defensible slide in the deck.*

---

## Anticipated questions

| Question | Answer |
|---|---|
| "Why not YOLO / object detection?" | Bounding boxes exist for only part of FIgLib, and the question is *when* an alert fires, not where the plume is. Classification + a timing protocol answers it directly. |
| "Isn't 1% false positives still a lot?" | Yes — per frame, per camera, per minute. Multiply it out for a 100-camera network and say the number. That is why the persistence rule exists, and it is a genuine limitation. |
| "Why is arm B better?" | It sees what *changed*. Early smoke is faint in absolute terms but is the only new thing in a fixed scene. |
| "Could you just threshold the difference image?" | That is arm C, the control. Quote the measured gap. |
| "Why not train on more data?" | Compute budget. The seed spread indicates how much of the result is noise. |
| "Is the CO₂ number real?" | No — it is an upper bound per incident. Give the §7.4 answer. Do not oversell it. |
