#!/usr/bin/env python3
"""Write each leg's cubes aligned to one another and cropped to their overlap.

    python export_aligned.py                      # all legs, default paths
    python export_aligned.py --only 12.09/Rat_1   # just the matching legs

The cubes of a leg are the same scene under different excitations, but the
animal drifts between them — on this dataset five of eight legs are out by more
than 5 px, one by 130. Every downstream step that treats a pixel as one
spectrum across excitations (concatenation above all) is wrong where that is
true.

This fixes it once, on disk. Each leg is registered, and every cube is written
out cropped to the rectangle all of them can see, in its own frame. The result
is a folder of cubes that share a pixel grid: pixel (x, y) is the same piece of
tissue in all seven, so one annotation covers the leg and concatenation is
finally meaningful.

Cropping rather than padding is deliberate — a padded border is invented data,
and a border of zeros would quietly become a class of its own to a segmentation
model.

Only the shift is applied: whole-pixel translation, no resampling, so every
output pixel is a source pixel unchanged. Wavelengths, exposures and the rest
of the acquisition records are carried through untouched (see im3_write.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import register  # noqa: E402
from im3 import Im3Cube  # noqa: E402
from im3_write import write_cropped  # noqa: E402

ROOT = Path("/Users/zmkoyan/User Files/multi-excitation-hsi-in-vivo-rats")
SRC_DEFAULT = ROOT / "Data/Raw/Rat_Cubes"
DEST_DEFAULT = ROOT / "Data/Processed/Alligned_Cropped_All_channels_kept/Distinct_cubes"


def band_key(path):
    """Excitation ascending, white light last — the project's usual cube order."""
    m = re.match(r"\s*(\d{3})", path.stem)
    return (0, int(m.group(1)), path.stem.lower()) if m else (1, 0, path.stem.lower())


def find_legs(src, folder_name="Distinct_Cubes"):
    """Every folder of that name holding cubes, with its cubes in channel order.

    Matching on the folder name keeps exploratory sessions out: 09.09/RAT_1 is
    a pile of different scenes, exposures and zoom levels in one directory, and
    a shared frame means nothing there.
    """
    legs = []
    for folder in sorted(src.rglob(folder_name)):
        cubes = sorted(folder.glob("*.im3"), key=band_key)
        if len(cubes) >= 2:
            parts = folder.relative_to(src).parts
            legs.append({"dir": folder, "cubes": cubes,
                         "name": "_".join(parts[:-1]).replace(" ", "_")})
    return legs


def leg_geometry(leg, drop_bad=True):
    """Register one leg. Returns the reference, the offsets and the crop boxes."""
    shapes, means = [], []
    for path in leg["cubes"]:
        with Im3Cube(path) as c:
            shapes.append((c.bands, c.height, c.width))
            means.append(c.mean_image())
    if len({(h, w) for _, h, w in shapes}) != 1:
        raise ValueError(f"{leg['name']}: cubes differ in size {sorted(set(shapes))} — "
                         "they cannot share a frame")
    bands, height, width = shapes[0]

    offsets, ref, records = register.register(means)
    bad = [i for i, r in enumerate(records) if not r["alignable"]]

    # A cube no translation can place onto the reference must not set the crop,
    # and must not be exported as if it had been aligned — its labels would sit
    # on the wrong tissue.
    keep = [i for i in range(len(records)) if i not in bad] if drop_bad else list(range(len(records)))
    box = register.overlap_box([offsets[i] for i in keep], width, height)

    crops = []
    for (dx, dy) in offsets:
        # The overlap rectangle is stated in the reference's frame; each cube
        # sees it at its own offset.
        crops.append([box[0] - int(dx), box[1] - int(dy),
                      box[2] - int(dx), box[3] - int(dy)])
    return {"reference": int(ref), "offsets": [[int(a), int(b)] for a, b in offsets],
            "records": records, "overlap_box": box, "crops": crops, "keep": keep,
            "unalignable": bad,
            "bands": [b for b, _, _ in shapes], "height": height, "width": width}


def export_leg(leg, dest, force=False, unalignable="stop"):
    geo = leg_geometry(leg, drop_bad=(unalignable == "skip"))
    out_dir = dest / leg["name"]
    manifest_path = out_dir / "_alignment.json"

    if geo["unalignable"] and unalignable == "stop":
        names = ", ".join(leg["cubes"][i].stem for i in geo["unalignable"])
        stale = " — STALE OUTPUT ON DISK, DELETE IT" if manifest_path.exists() else ""
        # Refusing to write is only safe if nothing older is left sitting there
        # claiming to be this leg.
        return geo, out_dir, f"NOT EXPORTED — no translation aligns {names}{stale}"
    if unalignable != "skip":
        geo["keep"] = list(range(len(leg["cubes"])))

    if manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text())
        if old.get("overlap_box") == geo["overlap_box"]:
            return geo, out_dir, "already exported"
        return geo, out_dir, "EXISTS with a different crop — rerun with --force"

    x0, y0, x1, y1 = geo["overlap_box"]
    cubes_meta = []
    for i, path in enumerate(leg["cubes"]):
        if i not in geo["keep"]:
            continue
        dx, dy = geo["offsets"][i]
        comments = json.dumps({
            "aligned_cropped_by": "Code/Anotation_tool/export_aligned.py",
            "source": str(path.relative_to(SRC_DEFAULT.parent.parent)),
            "leg": leg["name"],
            "reference": leg["cubes"][geo["reference"]].name,
            "offset_to_reference": {"dx": dx, "dy": dy},
            "convention": ("dx, dy: a feature at (x, y) in this cube was at "
                           "(x + dx, y + dy) in the reference cube's frame"),
            "source_crop": geo["crops"][i],
            "overlap_box_in_reference": geo["overlap_box"],
        }, separators=(",", ":"))
        nx, ny, nc = write_cropped(path, out_dir / path.name, geo["crops"][i], comments)
        cubes_meta.append({"file": path.name, "source": str(path),
                           "dx": dx, "dy": dy,
                           "score": geo["records"][i]["score"],
                           "score_unshifted": geo["records"][i]["score_unshifted"],
                           "alignable": geo["records"][i]["alignable"],
                           "source_crop": geo["crops"][i],
                           "shape": {"C": nc, "Y": ny, "X": nx}})

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "leg": leg["name"],
        "source_dir": str(leg["dir"]),
        "reference": leg["cubes"][geo["reference"]].name,
        "overlap_box": geo["overlap_box"],
        "source_size": {"width": geo["width"], "height": geo["height"]},
        "output_size": {"width": x1 - x0, "height": y1 - y0},
        "note": ("Every cube here is cropped to the same piece of tissue, so pixel "
                 "(x, y) is the same location in all of them. Whole-pixel shifts "
                 "only — no resampling."),
        "cubes": cubes_meta,
        "excluded": [{"file": leg["cubes"][i].name,
                      "reason": "no translation aligns it with the others",
                      "best_offset": geo["offsets"][i],
                      "best_score": geo["records"][i]["score"]}
                     for i in geo["unalignable"] if i not in geo["keep"]],
        "written": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }, indent=2))
    return geo, out_dir, "written"


def verify(out_dir):
    """Re-register what was written: the residual shift should now be ~0."""
    cubes = sorted(out_dir.glob("*.im3"), key=band_key)
    means = []
    for p in cubes:
        with Im3Cube(p) as c:
            means.append(c.mean_image())
    offsets, ref, records = register.register(means)
    worst = max(max(abs(r["dx"]), abs(r["dy"])) for r in records)
    return worst, [r["score"] for r in records]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--dest", type=Path, default=DEST_DEFAULT)
    ap.add_argument("--folder-name", default="Distinct_Cubes",
                    help="only folders with this name are treated as legs")
    ap.add_argument("--only", help="substring: export just the legs whose path contains it")
    ap.add_argument("--force", action="store_true", help="re-export legs already written")
    ap.add_argument("--unalignable", choices=["stop", "skip", "keep"], default="stop",
                    help="a cube no translation can align with the rest: stop and "
                         "report the leg (default), skip just that cube, or keep it "
                         "anyway at its best-effort offset")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args(argv)

    legs = find_legs(args.src, args.folder_name)
    if args.only:
        legs = [l for l in legs if args.only in str(l["dir"])]
    if not legs:
        raise SystemExit(f"no '{args.folder_name}' folders with cubes under {args.src}")

    print(f"  source  {args.src}")
    print(f"  dest    {args.dest}")
    print(f"  legs    {len(legs)}\n")
    print(f"{'leg':26s} {'reference':16s} {'in':>11s} {'out':>11s} {'lost':>6s} cubes  status")
    print("-" * 104)

    total = 0
    for leg in legs:
        t = time.time()
        if args.dry_run:
            geo = leg_geometry(leg, drop_bad=(args.unalignable == "skip"))
            x0, y0, x1, y1 = geo["overlap_box"]
            status = "dry run"
            out_dir = args.dest / leg["name"]
        else:
            geo, out_dir, status = export_leg(leg, args.dest, args.force, args.unalignable)
            x0, y0, x1, y1 = geo["overlap_box"]
        w, h = x1 - x0, y1 - y0
        lost = 1 - (w * h) / (geo["width"] * geo["height"])
        n = len(geo.get("keep", leg["cubes"]))
        print(f"{leg['name']:26s} {leg['cubes'][geo['reference']].stem:16s} "
              f"{geo['width']}x{geo['height']:<5d} {w}x{h:<5d} {lost * 100:5.1f}% "
              f"{n}/{len(leg['cubes'])}  {status} ({time.time() - t:.0f}s)")
        if status == "written" and not args.no_verify:
            worst, scores = verify(out_dir)
            flag = "ok" if worst <= 3 else "STILL OFF"   # 1-2 px is measurement noise
            print(f"{'':26s} {'':16s} verify: residual shift {worst} px, "
                  f"worst score {min(scores):.2f}  [{flag}]")
        if status == "written":
            total += sum(p.stat().st_size for p in out_dir.glob("*.im3"))

    if total:
        print(f"\n  wrote {total / 1e9:.1f} GB")
    print(f"\n  next: annotate them\n"
          f"    python annotate.py \"{args.dest}\"\n"
          f"  masks land in {args.dest.parent / (args.dest.name + '_labels')}")


if __name__ == "__main__":
    main()
