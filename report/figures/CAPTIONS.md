# Figures: captions, claims and provenance

Every figure is drawn by `make_figures.py` from constants recorded in
`FOVEANET_LOG.txt`. Nothing is recomputed at draw time. The **produced by** row
gives the command and archive that generated the numbers, so any figure can be
traced back to the run behind it and regenerated.

Palette: validated categorical default, slots 1–3 (blue `#2a78d6`, orange
`#eb6834`, aqua `#1baf7a`). Clears the all-pairs CVD floor (ΔE 9.2) and the
normal-vision floor (ΔE 24.0) in light mode. Aqua falls below 3:1 contrast on
the surface, so any series carrying it is directly labelled — the documented
relief rather than legend-only identity.

---

## Figure 1 — `fig10_pipeline`


**Caption.** The method as a pipeline. The event stream is cut into 10 ms
frames with no accumulation across them; a three-component Gaussian mixture is
fitted to each, initialised from the previous frame's parameters rather than
from a fresh seed; each component accumulates a persistence score as an
exponential moving average of its share of the frame's events; and the mixture
density weighted by that score is the saliency map from which the fovea box is
taken. The loop is the contribution: without it a component in one frame has no
relationship to any component in the last, and a per-component accumulated
score cannot be defined at all.

**Supports.** The method, and specifically why warm-starting is structural
rather than an optimisation.

**Section.** Method — opening figure.

**Produced by.** `make_figures.py`, hand-drawn schematic.

---

## Figure 2 — `fig06_warm_vs_cold`


**Caption.** Fovea dwell against frame length, for a mixture carried forward
between frames and one fitted independently on each. The advantage widens as the
window shortens: roughly sevenfold at 100 ms and unbounded at 1 ms, where a
per-frame fit changes fovea identity 563 times a second and holds no component
long enough for an accumulated score to mean anything.

**Supports.** Warm-starting is not an optimisation but the thing that makes a
per-component persistence score computable at all, and what puts short frames
within reach.

**Section.** Results — frame length. Also referenced from Method.

**Produced by.** Notebook 03 section 8, recorded at A1.2.

---

## Figure 3 — `fig11_persistence_traces`


**Caption.** Accumulated persistence for the three mixture components over one
recording. The traces are continuous and cross repeatedly, and one component
decays to zero as it stops holding activity. Vertical rules mark the frames on
which the fovea changed component. Two earlier definitions of persistence were
abandoned because they saturated: a within-frame version reached 1.0 for every
cluster over a 50 ms window, and a binary alive-or-not flag reached 1.0 for two
components of three. The exponential moving average of event share is the form
that discriminates.

**Supports.** Persistence is a usable ranking signal rather than a saturating
one, and the choice of definition was forced by two measured failures rather
than assumed.

**Section.** Method — persistence.

**Produced by.** `make_figures_qual.py`, one DVS128 Gesture recording at a 10 ms
window.

---

## Figure 4 — `fig05_saliency_stages`


**Caption.** One 10 ms frame through the method: the raw events; the mixture
components carried forward from the previous frame, annotated with each
component's accumulated persistence; the mixture density weighted by that
persistence, which is the saliency map; and the resulting fovea box. The frame
shown is representative in the sense that the saliency peak falls inside the
chosen box, which holds on 88% of frames under the persistence rule and 98%
under persistence-per-area.

**Supports.** The method section. Shows that the saliency map is not a separate
construction but the mixture itself, and that persistence is a per-component
quantity only because components keep their identity between frames.

**Section.** Method — the saliency map.

**Produced by.** `make_figures_qual.py`.

---

## Figure 5 — `fig07_all_conditions`


**Caption.** Every condition on DVS128 Gesture under the official
subject-independent protocol, five seeds, final epoch, with the percentage of
the sensor each retains. The dashed line is the full sensor. A fovea crop at 3%
of the sensor reaches 0.837 against 0.511 for a fixed centre crop of identical
size and 0.327 for a randomly placed one.

**Supports.** Placement, with the two controls that make it a measurement rather
than an observation.

**Section.** Results — placement.

**Caveat to carry in the text.** `recentred` leads here, but that result is
DVS128 Gesture only: on SL-Animals it loses to the full sensor by 0.112 and on
DVS-Lip by 0.168. It should not be presented as a general benefit.

**Produced by.** `classify.py --crops data/crops_ablation_10ms.npz --epochs 40
--seeds 5 --val-subjects 0`.

---

## Figure 6 — `fig01_sensor_fraction_sweep`


**Caption.** Accuracy against the fraction of the sensor retained, with the crop
placed two ways: on the fovea (blue) and at the centre of the frame (orange).
Both curves must meet at 100%, where the two conditions are the same full frame,
which makes that point a check on the sweep rather than a result. The shaded
region is the placement premium. The fovea curve peaks above the full-sensor
baseline at 43.6% of the sensor and remains level with it down to 23.7%, while
the centre curve only reaches the baseline by keeping everything. Fractions are
*achieved* rather than nominal: a large fovea box is clipped at the frame edge,
so a nominal 75% and 50% achieve 0.595 and 0.436. DVS128 Gesture, official
subject-independent split, five seeds, final epoch.

**Supports.** Foveation is not a fixed trade — there is a range of retained
fractions over which a well-placed crop matches or beats the full sensor, and
the losses reported at 3% were a consequence of cropping an order of magnitude
past that range.

**Section.** Results — sensor-fraction sweep.

**Produced by.** `build_crops_sweep.py --out data/crops_sweep_10ms.npz --window
10000 --split both`, then `classify.py --crops data/crops_sweep_10ms.npz
--epochs 40 --seeds 5 --val-subjects 0`. Peak margin verified separately on ten
seeds (`--seed-start 5 --conditions full,fov50`): +0.0305 at 7.4 se.

---

## Figure 7 — `fig02_diagnostic_predicts_outcome`


**Caption.** The blind-box ratio, measured on a sample of recordings before any
training, against the placement premium the classifier subsequently measured
(fovea crop minus a fixed crop of identical size). The relationship is monotone
across four datasets and changes sign: where activity is localised the fovea is
worth a great deal, and where it is diffuse a fixed crop is preferable. DVS-Lip
was measured and its failure recorded in `PREDICTIONS.txt` before its classifier
was run, making that point a forward prediction rather than a fitted one. The
connecting line indicates ordering only; four datasets do not define a curve.

**Supports.** Whether foveation helps is predictable in advance from a cheap
property of the recording, rather than being a fact about a method that has to
be discovered by training.

**Section.** Results — boundary condition.

**Produced by.** `blind_box.py --dataset {gesture,ucf,dvslip,slanimals} --limit
120` for the ratios; `classify.py` on `crops_official_10ms_v2.npz`,
`crops_slanimals_10ms.npz`, `crops_dvslip_10ms.npz` and `crops_ucf.npz` for the
premiums.

**Caveat to carry in the text.** The UCF-50 premium comes from the earlier
improvised-split protocol at three seeds, not the official-split protocol used
for the other three. The ordering is unaffected but the value is not directly
comparable.

---

## Figure 8 — `fig03_four_datasets`


**Caption.** The same four conditions on four datasets, ordered left to right by
blind-box ratio. The fovea crop (blue) leads the centre crop (orange) by 0.326
on DVS128 Gesture, by 0.059 on SL-Animals, and trails it on UCF-50 and DVS-Lip.
On DVS-Lip the fovea is statistically indistinguishable from a randomly placed
box of the same size (0.334 against 0.354, 0.4 se): the recordings are already
cropped to the mouth, so there is nothing left for a saliency mechanism to
select. Read DVS-Lip's ordering rather than its absolute accuracies — the full
condition reaches only 0.516 on 20 words against a chance rate of 0.05, because
word-level lip reading is hard for a 71k-parameter network on 32×32 input.

**Supports.** The sign change in Figure 7, shown condition by condition, with the
full-sensor and random-crop references that make each panel interpretable on its
own.

**Section.** Results — boundary condition.

**Produced by.** `classify.py --epochs 40 --seeds 5 --val-subjects 0` on
`crops_official_10ms_v2.npz`, `crops_slanimals_10ms.npz` and
`crops_dvslip_10ms.npz`; UCF-50 from the earlier run recorded in A5.4.

---

## Figure 9 — `fig04_where_the_fovea_lands`


**Caption.** The fovea box on five frames of one recording from each of three
datasets, ordered top to bottom by blind-box ratio. On DVS128 Gesture the box is
small and stays on the moving hand. On SL-Animals it wanders over the signer's
body without settling, and is twice the size. On DVS-Lip, whose recordings are
already cropped to the mouth, it covers a quarter of the frame and by the later
frames most of it. Frames are sampled across each recording but restricted to
those with an above-median event count: below that the mixture has too little to
fit and the box inflates toward the whole frame, which is a real behaviour but
not what this figure is about.

**Supports.** The boundary condition, shown rather than tabulated. The mean box
area rises with the ratio — 7%, 15%, 26% of the frame — which is the mechanism
behind the sign change in Figure 7.

**Section.** Results — boundary condition. Consider placing beside Figure 7.

**Produced by.** `make_figures_qual.py`, running `foveanet.PersistentGMM` with
`rule="persistence+hyst"` at a 10 ms window, exactly as `build_crops.py` does.

---

## Figure 10 — `fig08_direction_coherence`


**Caption.** The cosine between a tracker's step and the target's, against the
same quantity computed on the target's own successive steps. The targets move
smoothly and predictably, at 0.667; every selection rule tried recovers between
0.022 and 0.064 of that. The box is not failing to keep up — its motion is
close to unrelated to the target's.

**Supports.** The tracking failure is one of identity rather than of motion, and
that rules out the whole family of fixes aimed at motion: better damping, a
velocity model, a different selection rule.

**Section.** Results — tracking.

**Produced by.** `tracker.py` variants over all 60 VOT sequences, recorded at
9.7, 9.8 and 9.13.

---

## Figure 11 — `fig09_appearance_controls`


**Caption.** Normalised correlation at the target's true position, and the share
of candidate positions scoring above it, for three template sources. A template
from the same frame recovers the target almost exactly, confirming the geometry.
A template one frame old still ranks it in the top 6% of positions, so event
appearance does carry identity over short horizons. A template from the first
frame of the sequence ranks it at chance.

**Supports.** The appearance tracker's failure is template drift and the
inability to bootstrap from the tracker's own estimates, not an absence of
identity information in the data. This is what makes the negative result
bounded rather than bare.

**Section.** Results — tracking.

**Produced by.** The controls in 9.10, over 10 VOT sequences at 2 px smoothing.

---

## Complete

All eleven figures are produced. `make_figures.py` draws the ones that come from
recorded constants; `make_figures_qual.py` draws the four that run the model on
real recordings and therefore need the datasets present.
