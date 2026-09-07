# Early Wildfire Smoke Detection — and What Minutes Are Worth in Tonnes of CO₂

A computer-vision method for **early** wildfire detection, evaluated on the metric that actually
matters for climate impact — *how many minutes after ignition it fires* — and connected to an
emissions estimate.

Coursework project for a global warming theory course.

---

## The idea in one paragraph

Most wildfire-detection projects classify photographs of **flames**. By the time flames are large
enough to photograph, the fire is not early. Fire lookouts watch for **smoke**, which is visible
minutes before anything else. So the question is not *"can a CNN recognise a fire?"* — it obviously
can — but **how many minutes after a plume first becomes visible does the detector fire, at a
false-alarm rate low enough to deploy?** We can answer that because
[FIgLib](http://hpwren.ucsd.edu/HPWREN-FIgLib/) encodes, in every filename, that image's offset in
seconds from the moment the plume became visible.

## Method

Early smoke is small and faint: FIgLib frames are 2048×1536 or 3072×2048 depending on the camera,
and an early plume may span 60 px. Resize to the customary 224×224 and it becomes ~4-7 px of
low-contrast haze that does not survive the first pooling layer. A single-frame 224 px classifier
can learn *"this scene is on fire"* long after ignition, but it cannot be early. Two cheap fixes:

1. **448×448 input** instead of 224×224.
2. **Temporal stacking.** These are *fixed* cameras, so the background is free information. For the
   frame at time *t* we build a 3-channel tensor — `gray(t)`, `gray(t−4 min)`,
   `|gray(t) − gray(t−4 min)|` — which drops straight into ImageNet-pretrained weights with no
   architecture change. A plume nearly invisible in channel 0 is obvious in channel 2, because it is
   the only thing in the scene that appeared.

Two arms, identical in every respect but the input transform, so the comparison tests the idea
rather than tuning effort:

| Arm | Input | Role |
|---|---|---|
| A | `gray(t)` ×3 | baseline — expected to be late |
| B | the temporal stack | hypothesis: detects earlier at the same false-alarm rate |

A classical no-learning control (difference → blur → threshold → largest blob) measures what the
learned model is actually worth.

## Evaluation

Reported at a **fixed 1% false-alarm operating point**, with the threshold chosen on validation
fires and applied unchanged to test:

- **Time to detection** — walk each test fire forward from ignition, alert at the first minute where
  the score clears threshold for 2 consecutive frames (a persistence rule; single-frame triggers are
  false-alarm machines). Fires never detected within 40 minutes are **censored and reported
  separately**, not dropped.
- **Detection probability vs. minutes since ignition** — the headline figure.
- **Grad-CAM** overlays plus the model's worst false alarms, so its real failure mode is named.
- **Cross-dataset false-alarm rate** on the `cloud` class of a *different* camera network
  ([`sagecontinuum/smokedataset`](https://huggingface.co/datasets/sagecontinuum/smokedataset)).

Every number is the mean ± std over 3 seeds.

### The one thing that decides whether any of it means anything

FIgLib's 523 sequences cover only **338 distinct fires** — one fire is often filmed by up to 9
cameras simultaneously, and consecutive frames are a minute apart. Splitting on sequence, or on
frame, scatters near-identical views of the same fire across train and test and sends AUROC to ~0.99
for no reason. Everything here is split on the **fire date**, and `test_pipeline.py` asserts it.

## Running it

```bash
python download_figlib.py --sequences 30    # ~2.6 GB, cached, resumable
python test_pipeline.py                     # CPU, seconds — validates the pipeline before you train
# then run wildfire_smoke_detection.ipynb top to bottom on a Colab T4 (~45 min)
```

`download_figlib.py` is stdlib-only and deterministic. `--sequences 4` is enough for
`test_pipeline.py`; 30 is enough for a stable result.

## Layout

```
download_figlib.py                 fetch + extract FIgLib sequences (stdlib only)
test_pipeline.py                   CPU checks on the notebook's own functions — run before training
wildfire_smoke_detection.ipynb     data, training, evaluation, Grad-CAM, emissions
figures/                           generated; referenced by the report
report/report.md                   the graded artefact
slides/                            presentation
```

## Climate link

Area burned grows as **t²** for a radially spreading fire, so halving detection delay quarters the
fire you have to fight. Converted to emissions via the Seiler–Crutzen form `E = A × B × C × EF`
(the basis of inventories like GFED), the notebook produces a sensitivity table across spread rates
and detection delays.

Two caveats the report states plainly, because they matter more than the headline number:

- **The table is an upper bound per incident, not an expected saving.** Fire emissions are dominated
  by the minority of fires that escape initial attack. Early detection raises the *probability* that
  initial attack succeeds; it does not deterministically bank the tonnage. Estimating that shift
  needs suppression-outcome data this project does not have.
- **Detection is not response.** Minutes convert to hectares only if crews are available and can
  reach the fire — and in the wind events where it matters most, often they cannot.

> The emission coefficients in the notebook (`FUEL_LOAD_T_HA`, `COMBUSTION_COMPLETE`,
> `EF_CO2_G_PER_KG`) are the right order of magnitude but ship **uncited**. Replace each with a
> value from a paper you have read and cite it inline before submitting.

## Known limitations

Daylight only (FIgLib has no night sequences; real networks pair visible-band with IR) · one camera
network in Southern California chaparral · arm B requires a fixed camera with history, so it cannot
score an arbitrary photograph · the label is *"plume visible"*, not true ignition, so every latency
is measured from a generous starting line · a handful of held-out fires, hence real uncertainty on
the median.

## Data

- [FIgLib](http://hpwren.ucsd.edu/HPWREN-FIgLib/) — HPWREN, UC San Diego. 523 sequences, free, no login.
- [`sagecontinuum/smokedataset`](https://huggingface.co/datasets/sagecontinuum/smokedataset) — MIT, for the cross-dataset test.
- Dewangan et al., *FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real-Time Wildland Fire
  Smoke Detection*, [arXiv:2112.08598](https://arxiv.org/abs/2112.08598) — the reference point for this task.
