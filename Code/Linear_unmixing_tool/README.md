# Linear unmixing tool

Nuance-style linear spectral unmixing of the processed `.im3` cubes — with the
part Nuance has not got: **the same regions are unmixed on every version of the
leg at once**, and each result is scored against the leg's label mask, so the
ways we have of preparing a cube are compared on one page instead of by eye,
one file at a time.

```bash
cd "Code/Linear_unmixing_tool"
../../venv_for_HSI/bin/python unmix.py
```

opens a browser on `Data/Processed`. Point it somewhere else with a positional
argument; `--out DIR` moves where results are written.

## What it compares

A leg exists in two aligned trees and, in each, as four concatenated cubes plus
the seven single-excitation cubes they were built from — **22 versions**, all on
one pixel grid, all covered by one label mask:

| version id | what it is | ch |
|---|---|---|
| `all__concat__raw` | the 7 cubes stacked, raw sensor counts | 217 |
| `all__concat__per_chan` | z-normalised per channel | 217 |
| `all__concat__per_exc` | z-normalised per excitation cube | 217 |
| `all__concat__whole` | z-normalised over the whole 4D cube | 217 |
| `all__ex__365` | the 365 nm cube on its own | 31 |
| `oe__…` | the same, with the overexposed leading bands dropped | 210 / 26-31 |

`--tree oe` narrows the whole tool — browser and batch alike — to
`Alligned_cropped_overexposed_channels_removed`, `--tree all` to
`Alligned_Cropped_All_channels_kept`. One tree is **11 versions per leg and no
duplicates at all** (see below), which is the cleaner comparison of the two;
both trees together is the one that also answers "does dropping the overexposed
bands help?".

Only the **aligned, cropped** trees are in here. The two full-frame trees
(`ALL_channels_kept`, `Overexposed_channels_removed`) are left out on purpose:
their seven excitations are not on a common pixel grid, so one region set and
one mask cannot be correct across them, and no mask was ever painted for them.
Adding them would mean drawing regions seven times per leg and scoring nothing.

## The workflow

One leg at a time. **Pick → paint → unmix → read.**

1. **Pick a leg** and, in *Display*, the version you want to *look at* while you
   work. Which one you look at does not change the result — the regions are
   pixel coordinates, and every version is measured at those same pixels.
2. **Paint a region per structure.** Brush, box, right-drag to erase. The
   *Endmembers* plot under the display shows the spectrum each region currently
   gives, split at the excitation boundaries, and settles as you add pixels.
   **Seed from mask** fills in a starting set sampled out of the label mask —
   a few small blobs per class, ~2% of it, not the whole class.
3. **Unmix this leg** (or *all legs*). Every version is unmixed and scored;
   about 3 s for a small leg, 25 s for all six.
4. **Results** tab, **this leg**: the class maps side by side, sorted by
   accuracy, with a table underneath. Click any card for the abundances, the residual, the
   per-class numbers and the confusion matrix. `report ↗` is the same thing
   across all legs as a standalone page, opening with a ranking chart — one row
   per version, one dot per leg, coloured by whether the cube is a
   concatenation or a single excitation. Dots rather than bars on purpose: the
   accuracies sit between about 0.45 and 0.97, and a bar chart would either
   flatten them all against a zero baseline or lie about its own length by not
   having one.

The report opens with two charts:

- **Every leg across every cube version** — the versions along x in build order
  (310 nm through white light, then the concatenations, with a rule marking
  where one ends and the other begins), one line per leg, and a **metric picker**:
  accuracy, mean IoU, IoU per class, or the residual RMSE of the fit. This is the
  chart that shows what an average hides — whether all six legs agree that
  concatenating helps, or whether one leg is carrying the mean. Hover a leg in
  the legend to pick its line out of the crowd; hover a point for its value.
- **Accuracy by cube version** — the ranking, one dot per leg plus the mean.

Under them, the **across-all-legs table** carries the same per-structure detail
the per-leg tables do — mean accuracy and its spread, mean IoU, the IoU of each
structure averaged over the legs, and the residual fraction — because a version
that wins on accuracy can be losing on the two small classes, and an average
over classes hides exactly that. A line under the table gives each structure's
IoU averaged over every version, and which version did best on it.

**Every table sorts.** Click a column heading — in the app and in the report —
to order by it; click again to reverse. Numbers start high-to-low, names A-to-Z,
and in the report a leg's thumbnails re-order with its table so the pictures and
the numbers never disagree. This is worth using rather than reading the default
ranking: accuracy is dominated by muscle, the largest class, so sorting by
**IoU vessel** or **IoU nerve** usually names a different winner.

5. **Results** tab, **all legs**: the same table the report opens with — one
   row per version, every number averaged over the legs, with each structure's
   IoU as its own column and the residual fraction. Sorts on any heading.

Regions and settings are saved as you work, so closing the tab loses nothing.

## The maths

**Endmembers.** One spectrum per structure, the mean (or median) of its region's
pixels in *that version's* cube. That is the whole comparison: identical pixels,
different preparation, different spectra.

**Unmixing.** Every pixel `b` is written as `E a` with `a ≥ 0` (NNLS — what
Nuance does), or without the constraint (plain least squares). The NNLS here is
**exact, not iterative**: the solution is the unconstrained fit on some subset
of the endmembers, and it is the feasible subset with the smallest residual, so
with a handful of components all subsets are enumerated and every pixel of every
subset is solved at once in numpy. Verified against the KKT conditions to
machine precision.

**Extra component.** Optional, and off by default. `constant` adds a flat
component that soaks up an offset equal in every channel; `dark` adds the
cube's own per-channel 1st percentile. The z-normalised cubes carry a large
per-channel additive offset by construction — `concat_lib` maps z into uint16
through a shared window — so `dark` is the component that can absorb it, and
turning it on is a fair thing to compare.

**Prediction.** The class of the structure with the largest abundance, among the
structures that map to a mask class. A component with no class (an extra, or one
you added yourself) still competes for signal in the fit but never wins a pixel
— it is not something the mask distinguishes.

## Accuracy, and why there are two numbers

Each leg's mask is fully painted (2 = vessel, 3 = nerve, 4 = muscle, no
unlabeled pixels), so accuracy is defined over the whole frame.

Every version is scored twice:

- **all pixels** — the plain number.
- **outside the regions** — the same, with the endmember pixels removed.

The second is the one to trust and the one everything is sorted by. The
endmembers *are* the mean spectra of the region pixels, so those pixels are
close to fitted by construction; scoring on them flatters the model by however
much of the frame they cover. With the seeded ~2% regions the gap is small
(≈0.001), which is itself the useful check — if it ever gets large, the regions
have grown into a training set.

Reported per version: overall accuracy, mean IoU, per-class precision / recall /
IoU / dice, the confusion matrix, and two residuals.

## The residual

Unmixing writes every pixel's spectrum `b` as a mixture `E a` of the structure
spectra. Whatever is left over, `b − E a`, is the **residual**: the part of that
pixel the model could not account for. Two numbers summarise it:

- **`residual_rmse`** — the root-mean-square of `b − E a` across the channels,
  averaged over the pixels, **in the cube's own units**. Comparable between legs
  of the same version and *not* between versions: a raw cube's values sit under
  4040 counts while a z-normalised cube is mapped across the whole uint16 range,
  so the same quality of fit shows up about sixteen times larger.
- **`residual_rel`** — `sqrt(Σ‖b − Ea‖² / Σ‖b‖²)`, the fraction of the signal
  the model failed to explain. Scale-free, so this is the one to compare, and
  the one the charts and tables show.

It is a different question from accuracy. Accuracy asks *did the right structure
win this pixel*; the residual asks *does a mixture of these three spectra
describe this pixel at all*. A high residual means the spectrum carries
structure the endmembers cannot represent — a component nobody drew a region
for, saturation, or just noise — and it is worth reading beside accuracy: the
single excitations fit their 31 channels well (3.5–7.2% on this data) simply
because three endmembers have less to explain there, while the 210-channel
concatenations sit at 8.9–10.7% and still classify better.

Runs are reproducible: mask seeding is deterministic, so the same command twice
gives byte-identical regions and identical metrics. If you want to be sure a
table is current, delete its `--out` folder and re-run rather than topping it up.

## Duplicates

Only an issue when comparing **both** trees. Trimming the overexposed bands
touches the 385 and 400 nm cubes and nothing else, so most single-excitation
cubes are the *same file* in both trees: 5 of the 22 versions are a cube already
counted. Those are found by content hash and marked `duplicate_of` — the results
grid hides them by default and the report greys them out, because two identical
rows are one result, not two. With `--tree` there is only one of each and
nothing is marked.

## What it writes

Nothing is ever written into `Data/Processed`. Everything lands under
`Data/Derived/Linear_unmixing/`:

```
metrics_all.csv                one row per leg × version — the table to analyse
report.html                    every leg, every version, ranked, with pictures
<leg>/selection.json           structures, reference, solver settings
<leg>/rois.png                 the regions, uint8, 0 = unassigned, i+1 = structure i
<leg>/truth.png                the label mask in the class colours
<leg>/rois_preview.png         the regions, drawn on the reference cube
<leg>/metrics.csv              this leg's versions
<leg>/metrics.json             the same plus confusion matrices and pixel counts
<leg>/<version>/classmap.png   argmax of the scored abundances
<leg>/<version>/abundance_*.png  one per structure
<leg>/<version>/residual.png   how badly the model fitted, per pixel
<leg>/<version>/endmembers.csv the spectra this version measured, per channel
<leg>/<version>/metrics.json   this version alone
```

`--save-npy` additionally writes the raw `float32` abundance stacks,
`(K, Y, X)`, if you want the numbers rather than the pictures.

## Batch

```bash
../../venv_for_HSI/bin/python unmix.py --batch --report
```

runs every leg that already has regions saved and writes the comparison page,
with no browser involved. `--tree oe` restricts it to one tree and `--out DIR`
keeps that comparison in its own folder:

```bash
../../venv_for_HSI/bin/python unmix.py --batch --tree oe --seed-missing --report \
    --out ../../Data/Derived/Linear_unmixing_OE_removed
```

`--leg NAME` (repeatable) limits it; `--seed-missing`
gives any leg without regions a set sampled out of its label mask, which is the
quickest way to get a first full table:

```bash
../../venv_for_HSI/bin/python unmix.py --batch --seed-missing --report
```

Reading the results back is plain pandas:

```python
import pandas as pd
df = pd.read_csv("Data/Derived/Linear_unmixing/metrics_all.csv")
df = df[df.duplicate_of.isna()]                       # one row per distinct cube
print(df.groupby("label").accuracy_outside_rois.agg(["mean", "std"])
        .sort_values("mean", ascending=False))
```

## Keys

| | |
|---|---|
| `1`…`9` | pick the structure to paint |
| `B` / `R` / `E` | brush / box / eraser |
| right-drag | erase, whatever tool is active |
| `[` `]` | brush size |
| `X` / `V` | show regions / show the label mask |
| `F` | fit · wheel zooms at the cursor · shift-drag or middle-drag pans |
| `,` `.` | previous / next leg |
| `⌘Z` / `⇧⌘Z` | undo / redo · `⌘S` save now |
| `esc` | close the detail sheet |

## Verifying the browser half

```bash
../../venv_for_HSI/bin/python selftest/run.py
```

Appends `selftest/probe.js` to a throwaway copy of `index.html`, starts the app
against a temporary `--out` directory so nothing real is touched, drives it in
headless Chrome on a real wall clock, and reports 40 checks covering boot,
painting, undo, structure editing, saving, the endmember plot, the display
modes, seeding, a full unmixing run, the results grid, the detail sheet and leg
switching. Do **not** reach for `--virtual-time-budget`: it fast-forwards the
polling and races the real fetches.

## Notes

- Cubes stay memory-mapped; a row block at a time is converted to float, so a
  100 MB cube never becomes a 400 MB array. One cube is kept open at a time for
  the display, because scrubbing the band slider asks for band after band of
  one version.
- Display defaults to a 1–99 percentile stretch. The sensor clips at 4040
  counts, so a min-max stretch is usually worse.
- **Nothing the server sends is cached.** The result images are the trap: their
  URLs are stable (`/out/<leg>/<version>/classmap.png`) but their contents change
  every run, so a cached copy shows the *previous* run's regions and class maps
  while the numbers beside them are current. The app additionally stamps every
  image URL with the run time. If a picture ever looks like it disagrees with
  its numbers, that is the bug to suspect first.
- The regions preview is drawn **on the reference cube**, dimmed, rather than on
  black — a few hundred coloured pixels floating on a black field cannot be
  checked against anything, and the point of the picture is to see whether the
  vessel region is actually on the vessel. Its caption gives the pixel count per
  structure, so the picture can be checked against a number.
- The region map travels with the requests that use it (the seed call sends the
  structures on screen, not the ones last saved), because saving is debounced —
  reading state back off disk mid-edit is how the annotation tool once produced
  silently wrong masks.
- `.im3` reading is `Anotation_tool/im3.py` and `concat_lib.read_cube_info`;
  nothing here re-implements the format.
