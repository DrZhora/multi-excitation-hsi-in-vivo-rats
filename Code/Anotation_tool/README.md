# .im3 annotation tool

Per-pixel class labelling for hyperspectral `.im3` cubes, for training a UNet.

```bash
cd "Code/Anotation_tool"
../../venv_for_HSI/bin/python annotate.py \
    "../../Data/Raw/Rat_Cubes" \
    --export-to "../../Data/Processed/Alligned_Cropped_All_channels_kept" \
    --exclude 10.09 --exclude 09.09
```

`--exclude` skips any cube whose path contains that text. Point the tool at the
top of the tree and filter, rather than at `12.09/` directly — the leg names in
the export are built from the path below the input folder, so moving the input
down a level would rename `12.09_Rat_1_Left_leg` to `Rat_1_Left_leg`.

Here `10.09` is excluded by choice, and `09.09` because it is an exploratory
session: different scenes, exposures and zoom levels in one folder, where a
shared frame means nothing. Drop either flag to get them back.

You work one leg at a time. **Align → box → crop → label**, then the next leg:

1. **Auto-align** — every cube is registered against every other and one offset
   per cube is solved for.
2. **Check it** — pick a cube that moved and switch to **Diff** (black where
   they agree), **Blink** or **Checker**. Different excitations never score
   1.00; the picture is the judge. **Nudge** (`A`) what the solver got slightly
   wrong: drag, or arrows, `⇧` for 10 px.
3. **Draw the box** (`R`, then drag). It starts as everything all the cubes
   cover and cannot leave it. The button shows what it will produce:
   `Crop to box 500×400`.
4. **Crop to box** — writes every cube shifted onto the reference and cut to
   your box, then **switches the tool onto those files**. The view becomes the
   cropped image and the cubes now share a pixel grid.
5. **Paint, then Save.** No box, no alignment, no warping left — one mask, and
   it is written for every cube of the leg. Then `.` for the next leg.

**Re-crop** throws the cropped cubes and their labels away and puts you back on
the raw ones, if the box was wrong.

Because every cube is cut to the same tissue, the exported cubes share a pixel
grid — so one label image is correct for all of them, and concatenating them is
finally meaningful. Reopening the tool later puts you straight back on the
cropped cubes for any leg you have already cropped.

Nothing is written into `Data/Raw`. Work in progress (alignments, the
full-frame mask) lives in `<destination>/_work/` and is picked up again when
you reopen a leg, so you can stop mid-leg.

Options: `--port N`, `--no-browser`, `--no-recursive`, `--no-group` (annotate
every cube separately), `--no-previews`, `--only TEXT` / `--exclude TEXT`
(repeatable path filters), `--labels-dir DIR` (move the working files), and
without `--export-to` the tool is just an annotator that writes masks to
`<input>_labels/`.

`export_aligned.py` does the same crop-and-align for every leg in one batch,
choosing each box automatically as the full overlap — useful if you do not want
to pick boxes by hand:

```bash
../../venv_for_HSI/bin/python export_aligned.py          # add --dry-run first
```

## What you get

```
Alligned_Cropped_All_channels_kept/
    Distinct_cubes/<leg>/
        310_20000exp.im3             aligned + cropped to your box
        ... one per cube ...
        _alignment.json              reference, per-cube dx/dy/score, the box, provenance
    Distinct_cubes_labels/
        classes.json                 ids -> names -> colours
        <leg>/310_20000exp.png       uint8, exactly the cube's H x W, class id, 255 = unlabeled
        ... one per cube (identical: the cubes share a grid) ...
    _work/                           in-progress masks and alignments, full frame
```

Each mask PNG is exactly its cube's height and width, single channel, no
palette, no scaling — `np.array(Image.open(path))` gives class ids directly.
**Training needs no alignment logic at all**: open a cube, open its PNG, they
line up. The seven masks of a leg are byte-identical, because after the crop
the seven cubes are the same tissue on the same grid.

**255 is the ignore index.** Anything you never painted stays 255, and so does
anything that fell outside a cube after shifting, so a partially labelled leg
still trains: `CrossEntropyLoss(ignore_index=255)` skips those pixels.

### align.json

```json
{
  "reference": "365_3000exp.im3",
  "convention": "dx, dy: a feature at (x, y) in this cube is at (x + dx, y + dy) in the reference cube's frame",
  "cubes": {
    "310_20000exp.im3": {"dx": -51, "dy": 27, "score": 0.3691},
    "365_3000exp.im3":  {"dx": 0,   "dy": 0,  "score": 1.0}
  },
  "overlap_box": [8, 27, 1339, 1039],
  "roi": null
}
```

`overlap_box` is the reference-frame rectangle every cube can see; the tool
draws it as a dashed outline. Painting outside it is allowed, but those labels
fall off the edge of at least one cube and are saved as 255 there.

Nothing is resampled and no cube is rewritten — the offsets are whole pixels in
a sidecar, so warping a mask into a cube's frame and back is lossless.

## Using it for training

```python
import json
import numpy as np, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

import sys; sys.path.insert(0, "Code/Anotation_tool")
from im3 import Im3Cube

ROOT = Path("Data/Processed/Alligned_Cropped_All_channels_kept")

class Im3Segmentation(Dataset):
    """Aligned, cropped cubes and their masks.

    Returns (C, H, W) float32 and (H, W) int64. A cube and its mask share a
    pixel grid, so there is nothing to align or resample here.
    """

    def __init__(self, root=ROOT, bands=None, legs=None):
        root = Path(root)
        meta = json.loads((root / "Distinct_cubes_labels" / "classes.json").read_text())
        self.names, self.ignore = meta["labels"], meta["unlabeled"]
        self.bands = bands                       # None = all 31

        self.items = []
        for cube in sorted((root / "Distinct_cubes").rglob("*.im3")):
            leg = cube.parent.name
            if legs and leg not in legs:
                continue
            mask = root / "Distinct_cubes_labels" / leg / f"{cube.stem}.png"
            if mask.exists():
                self.items.append((cube, mask, leg))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cube_path, mask_path, _ = self.items[i]
        with Im3Cube(cube_path) as cube:
            bands = self.bands if self.bands is not None else range(cube.bands)
            x = np.stack([cube.channel(c) for c in bands]).astype(np.float32)
        # a palette PNG: plain np.array gives the class ids. Never .convert("L").
        y = np.array(Image.open(mask_path), dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
```

Then `nn.CrossEntropyLoss(ignore_index=255)` and
`out_channels = max(int(k) for k in ds.names) + 1`.

Note the class ids are **2, 3, 4** (vessel, nerve, muscle) — `background` is
defined but never painted. That means `out_channels` has to be 5 with channels
0 and 1 dead, unless you remap the ids to 0, 1, 2 when loading.

The seven cubes of a leg share one annotation, so they are seven samples of the
same labels under different excitation — useful as augmentation, but **split
train/val by leg, not by cube** (`Im3Segmentation(legs=[...])`), or the same
painted pixels land on both sides.

Two things worth doing before you train, neither of which the tool does for
you, because both are decisions about the experiment rather than about labels:

- **Normalise.** Raw counts run 0-4040 and clip at the ceiling, and exposure
  differs per excitation, so a raw cube is not comparable across bands. Either
  annotate the already-normalised cubes under `Data/Processed/`, or divide by
  the per-channel percentiles their sidecars record.
- **Tile.** 1392x1040x217 is far too large for one forward pass. Cut patches
  (512x512 is a reasonable start) and drop any patch that is entirely 255.

## Keys

| | |
|---|---|
| `1`…`0` | pick class |
| `B` / `E` / `P` / `R` / `G` | brush / eraser / polygon / region box / fill |
| `A` | nudge alignment (arrows move the viewed cube, `⇧` = 10 px) |
| `[` `]` | brush size |
| right-drag | erase, whatever tool is active |
| `V` | flip between Normal and Diff |
| `C` | next cube in this leg |
| `,` `.` | previous / next leg |
| wheel | zoom at the cursor |
| space-drag, middle-drag | pan |
| `F` | fit · `X` hide the mask |
| `←` `→` | previous / next band (or nudge, when `A` is on) |
| `⌘Z` / `⇧⌘Z` | undo / redo · `⌘S` save |

Polygon: click the vertices, `⏎` fills, `⌫` drops the last point, `esc`
cancels. `esc` also drops the region box.

**Fill** (`G`) is a paint bucket that stops at anything already painted. Draw
the nerve across the field, pick muscle, click above it — every unlabelled
pixel up to that line becomes muscle, in about 20 ms. Right-clicking pours a
region back out to unlabelled. **Fill rest** does the whole image at once:
everything still unlabelled becomes the current class, which is the usual last
step when what is left over is all one thing.

The **Display** panel only changes what you see, never what is saved. *Band*
shows one emission band (its wavelength is printed next to the slider), *RGB*
maps three bands onto red/green/blue, *Mean* averages the spectral axis for a
quick anatomical view. Low/High are percentile clip points — the sensor
saturates at 4040, so a min-max stretch is usually worse than the 1-99%
default. Gamma is applied in the browser and is free to drag.

Work is saved when you hit Save, switch legs, tick *done*, close the tab, and
once a minute while the mask is dirty. The header says `unsaved changes` in
amber whenever something is not yet on disk.

## Notes

- **The reference cube defines the frame the mask lives in.** Clicking `ref` on
  another cube re-expresses the mask, the offsets and the ROI in that cube's
  frame — nothing is lost, but it does move every label, so it is an edit, not
  a view setting.
- Alignment is **translation only**. The stage does not rotate and the animal
  shifts, so that is what the data needs; rotation or scale would show up as a
  score no nudge can improve.
- The match score is a normalised cross-correlation of gradient images over the
  overlap. Across excitations a *correct* match lands around 0.3-0.9, not 1.0,
  because the excitations genuinely light up different tissue. Judge alignment
  by Diff/Blink; use the score to decide which cube to look at first.
- Registering each cube straight against the reference is not enough — on this
  data 310 nm against 325 nm scores 0.14 and lands 300 px out. Every pair is
  measured and one offset per cube is fitted by weighted least squares with
  outlier rejection, which routes a stubborn cube through the neighbours it
  does match. See register.py.
- A cube is never fully loaded: it stays memory-mapped and single channels are
  pulled out as needed, which keeps a 630 MB 217-band cube usable.
- Masks are written to a temp file and renamed, so an interrupted save cannot
  destroy the previous one.
- If you delete a class from the editor after painting with it, those pixels
  keep their id in the mask and simply stop being drawn. The per-cube `.json`
  still lists the count, under the name `class <id>`.
