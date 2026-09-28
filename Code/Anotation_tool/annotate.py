#!/usr/bin/env python3
"""Semantic-segmentation annotator for .im3 hyperspectral cubes.

    python annotate.py /path/to/Distinct_Cubes

The cubes sitting in one folder are treated as one **group** — the same leg
under different excitations. You align them once, paint one mask, and that mask
is written out for every cube in the group, each in its own pixel frame.

Alignment is whole-pixel translation, estimated automatically and correctable
by hand; see register.py for why it is done the way it is. The offsets live in
a sidecar and nothing is ever resampled or rewritten, so the cubes on disk are
untouched.

Written to `<folder>_labels/`:

    classes.json                  ids -> names -> colours
    align.json                    reference cube, per-cube dx/dy/score, ROI
    <cube>.png                    uint8 H x W per cube, pixel = class id, 255 = unlabeled
    <cube>.json                   per-cube progress, including its alignment
    _previews/<cube>_overlay.png  a human-readable check image

255 is the ignore index, so a partially painted leg still trains:
`CrossEntropyLoss(ignore_index=255)` skips exactly what you never touched.
See README.md for a ready-made torch Dataset.

Why a local web app rather than a notebook: painting and nudging an alignment
need per-event feedback, and a kernel round trip per mouse move cannot give it.
The browser owns the interaction; Python reads cubes, registers them and writes
masks.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import socket
import sys
import threading
import traceback
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import register  # noqa: E402
from im3 import Im3Cube  # noqa: E402
from im3_write import write_cropped  # noqa: E402

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
UNLABELED = 255

DEFAULT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#00ced1",
]

ALIGN_CONVENTION = ("dx, dy: a feature at (x, y) in this cube is at "
                    "(x + dx, y + dy) in the reference cube's frame")


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _natural_key(text):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", text)]


def _band_key(stem):
    """Excitation ascending, white light last — the project's usual cube order."""
    m = re.match(r"\s*(\d{3})", stem)
    return (0, int(m.group(1)), stem.lower()) if m else (1, 0, stem.lower())


# --------------------------------------------------------------------------
# Cubes: opening, caching, and turning channels into 8-bit display images.
#
# A concatenated cube is 630 MB, so only single channels ever enter memory.
# Registration needs every cube of a group at once, but only as mean images
# (5.8 MB each), which are cached for the whole group.
# --------------------------------------------------------------------------
class CubeStore:
    def __init__(self, max_open=3, max_channels=12, max_means=16):
        self._lock = threading.RLock()
        self._open = {}                 # path -> Im3Cube, insertion-ordered LRU
        self._channels = {}             # (path, c) -> uint16 (Y, X)
        self._pcts = {}                 # (path, key, lo, hi) -> (float, float)
        self._means = {}                # path -> float32 (Y, X)
        self.max_open, self.max_channels, self.max_means = max_open, max_channels, max_means

    def cube(self, path):
        path = str(path)
        with self._lock:
            cube = self._open.pop(path, None)
            if cube is None:
                cube = Im3Cube(path)
            self._open[path] = cube
            while len(self._open) > self.max_open:
                old, victim = next(iter(self._open.items()))
                del self._open[old]
                for key in [k for k in self._channels if k[0] == old]:
                    del self._channels[key]
                victim.close()
            return cube

    def channel(self, path, c):
        path, key = str(path), (str(path), c)
        with self._lock:
            arr = self._channels.pop(key, None)
            if arr is None:
                arr = self.cube(path).channel(c)
            self._channels[key] = arr
            while len(self._channels) > self.max_channels:
                del self._channels[next(iter(self._channels))]
            return arr

    def mean(self, path):
        path = str(path)
        with self._lock:
            arr = self._means.pop(path, None)
            if arr is None:
                arr = self.cube(path).mean_image()
            self._means[path] = arr
            while len(self._means) > self.max_means:
                del self._means[next(iter(self._means))]
            return arr

    def bounds(self, path, arr, key, lo_pct, hi_pct):
        """Display bounds for one image, cached.

        Percentiles come off a stride-4 subsample: 90k pixels put the 1st/99th
        within a count or two of the exact answer and turn a 60 ms sort into a
        4 ms one, which matters while the band slider is being dragged.
        """
        ck = (str(path), key, round(lo_pct, 3), round(hi_pct, 3))
        with self._lock:
            hit = self._pcts.get(ck)
        if hit is not None:
            return hit
        sample = arr[::4, ::4]
        if lo_pct <= 0 and hi_pct >= 100:
            lo, hi = float(sample.min()), float(sample.max())
        else:
            lo, hi = (float(v) for v in np.percentile(sample, [lo_pct, hi_pct]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(sample.min()), float(max(sample.max(), sample.min() + 1))
        with self._lock:
            if len(self._pcts) > 512:
                self._pcts.clear()
            self._pcts[ck] = (lo, hi)
        return lo, hi

    def close_all(self):
        with self._lock:
            self._channels.clear()
            self._means.clear()
            for cube in self._open.values():
                cube.close()
            self._open.clear()



def mask_palette(classes):
    """A 256-entry PNG palette: each class its own colour, 255 magenta.

    Written as a palette ("P" mode) PNG, a mask shows its classes in colour in
    any image viewer while `np.array(Image.open(path))` still returns the raw
    class indices — the palette is metadata, not pixel data. An "L" mode mask
    is the same numbers but renders as near-black, because class 2 out of 255
    is black, which makes the files impossible to check by eye.

    255 is magenta rather than black so unlabeled area is obvious at a glance,
    and cannot be confused with a background class that is itself black.
    """
    pal = bytearray(768)
    for c in classes:
        i = int(c["id"]) * 3
        r, g, b = (int(c["color"][k:k + 2], 16) for k in (1, 3, 5))
        pal[i], pal[i + 1], pal[i + 2] = r, g, b
    pal[UNLABELED * 3:UNLABELED * 3 + 3] = bytes((255, 0, 255))
    return bytes(pal)


def save_mask_png(arr, path, classes):
    """Write a class-index mask, atomically, as a colour-palette PNG."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    im = Image.frombytes("P", (arr.shape[1], arr.shape[0]), arr.tobytes())
    im.putpalette(mask_palette(classes))
    tmp = path.with_suffix(".png.tmp")
    im.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, path)


def read_mask_png(path):
    """Class indices from a mask PNG, whether it is palette or greyscale."""
    with Image.open(path) as im:
        if im.mode == "P":
            return np.array(im, dtype=np.uint8)      # palette index == class id
        return np.array(im.convert("L"), dtype=np.uint8)


def _stretch(arr, lo, hi):
    """Linear window to uint8. Gamma is applied in the browser, not here."""
    out = (arr.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6))
    return np.clip(out, 0, 255).astype(np.uint8)


def render(store, path, mode, params):
    """An 8-bit PIL image of one cube, per the display settings.

    Always the cube's own unshifted pixels — the browser draws it at the
    cube's offset, so nudging an alignment costs no round trip.
    """
    lo_pct = float(params.get("lo", 1.0))
    hi_pct = float(params.get("hi", 99.0))

    if mode == "mean":
        arr = store.mean(path)
        lo, hi = store.bounds(path, arr, "mean", lo_pct, hi_pct)
        return Image.fromarray(_stretch(arr, lo, hi), "L")

    if mode == "rgb":
        planes = []
        for name in ("r", "g", "b"):
            c = int(params.get(name, 0))
            arr = store.channel(path, c)
            lo, hi = store.bounds(path, arr, c, lo_pct, hi_pct)
            planes.append(_stretch(arr, lo, hi))
        return Image.fromarray(np.dstack(planes), "RGB")

    c = int(params.get("c", 0))
    arr = store.channel(path, c)
    lo, hi = store.bounds(path, arr, c, lo_pct, hi_pct)
    return Image.fromarray(_stretch(arr, lo, hi), "L")


# --------------------------------------------------------------------------
# The workspace: groups of cubes, their alignment, their shared mask.
# --------------------------------------------------------------------------
class Workspace:
    def __init__(self, input_dir, recursive=True, previews=True, group=True,
                 export_to=None, labels_dir=None, only=None, exclude=None):
        self.input_dir = Path(input_dir).expanduser().resolve()
        if not self.input_dir.is_dir():
            raise SystemExit(f"not a folder: {self.input_dir}")
        self.export_to = Path(export_to).expanduser().resolve() if export_to else None
        # With an export destination the working files belong under it too, so
        # the source tree stays exactly as it was found.
        if labels_dir:
            self.labels_dir = Path(labels_dir).expanduser().resolve()
        elif self.export_to:
            self.labels_dir = self.export_to / "_work"
        else:
            self.labels_dir = self.input_dir.parent / f"{self.input_dir.name}_labels"
        self.previews = previews
        self.store = CubeStore()
        self._lock = threading.RLock()

        files = sorted(self.input_dir.glob("**/*.im3" if recursive else "*.im3"),
                       key=lambda p: (_natural_key(str(p.parent)), _band_key(p.stem)))
        # Filters match against the path relative to the input folder, so the
        # input can stay at the top of the tree — which is what keeps leg names
        # like "12.09_Rat_1_Left_leg" intact.
        found = len(files)
        if only:
            files = [f for f in files
                     if any(o in str(f.relative_to(self.input_dir)) for o in only)]
        if exclude:
            files = [f for f in files
                     if not any(x in str(f.relative_to(self.input_dir)) for x in exclude)]
        if found and not files:
            raise SystemExit(f"every one of the {found} cubes under {self.input_dir} was "
                             f"filtered out by --only/--exclude")
        if found != len(files):
            print(f"  filtered {found - len(files)} of {found} cubes out")
        if not files:
            where = "or its subfolders" if recursive else "(drop --no-recursive?)"
            raise SystemExit(f"no .im3 files in {self.input_dir} {where}")

        # Cubes are grouped by folder AND by shape: a shared mask only means
        # anything if the cubes share a pixel grid.
        buckets = {}
        for path in files:
            bands, height, width, waves = self._peek(path)
            key = (str(path.parent), height, width) if group else (str(path), 0, 0)
            buckets.setdefault(key, []).append(
                {"path": path, "name": path.stem, "bands": bands,
                 "rel": str(path.relative_to(self.input_dir)), "wavelengths": waves})

        self.groups = []
        for gid, (key, cubes) in enumerate(buckets.items()):
            rel_dir = Path(cubes[0]["rel"]).parent
            self.groups.append({
                "id": gid,
                "rel_dir": "" if str(rel_dir) == "." else str(rel_dir),
                "name": (str(rel_dir) if str(rel_dir) != "." else self.input_dir.name)
                        + (f"  [{cubes[0]['name']}]" if not group else ""),
                "cubes": cubes,
                "height": key[1] or cubes[0].get("height"),
                "width": key[2] or cubes[0].get("width"),
            })
        for g in self.groups:                       # shapes come from the peek above
            b, h, w, _ = self._peek(g["cubes"][0]["path"])
            g["height"], g["width"] = h, w
            self.adopt_cropped(g)                   # already cropped? use those cubes

        self.classes = self._load_classes()

    def _peek(self, path):
        cube = self.store.cube(path)
        return cube.bands, cube.height, cube.width, [round(x, 1) for x in cube.wavelengths]

    # -- classes -----------------------------------------------------------
    @property
    def classes_path(self):
        return self.labels_dir / "classes.json"

    def _load_classes(self):
        # The exported dataset carries its own copy, so the working folder can
        # be thrown away without losing the class list.
        if not self.classes_path.exists() and self.export_to:
            shipped = self.export_to / "Distinct_cubes_labels" / "classes.json"
            if shipped.exists():
                try:
                    blob = json.loads(shipped.read_text())
                    if blob.get("classes"):
                        return [{"id": int(c["id"]), "name": str(c["name"]),
                                 "color": str(c.get("color") or DEFAULT_COLORS[0])}
                                for c in blob["classes"]]
                except Exception:
                    pass
        if self.classes_path.exists():
            try:
                blob = json.loads(self.classes_path.read_text())
                if blob.get("classes"):
                    return [{"id": int(c["id"]), "name": str(c["name"]),
                             "color": str(c.get("color") or DEFAULT_COLORS[0])}
                            for c in blob["classes"]]
            except Exception:
                print(f"! could not read {self.classes_path}, starting fresh")
        return []

    def save_classes(self, classes):
        clean, seen = [], set()
        for c in classes:
            cid = int(c["id"])
            if not 0 <= cid < UNLABELED:
                raise ValueError(f"class id {cid} is outside 0..{UNLABELED - 1}")
            if cid in seen:
                raise ValueError(f"duplicate class id {cid}")
            seen.add(cid)
            clean.append({"id": cid, "name": str(c["name"]).strip() or f"class {cid}",
                          "color": str(c.get("color") or DEFAULT_COLORS[cid % len(DEFAULT_COLORS)])})
        clean.sort(key=lambda c: c["id"])
        with self._lock:
            self.classes = clean
            self.labels_dir.mkdir(parents=True, exist_ok=True)
            self.classes_path.write_text(json.dumps({
                "unlabeled": UNLABELED,
                "labels": {str(c["id"]): c["name"] for c in clean},
                "classes": clean,
                "source_folder": str(self.input_dir),
                "updated": _now(),
            }, indent=2))
        return clean

    # -- cropped legs ------------------------------------------------------
    def cropped_dir(self, g):
        return self.export_to / "Distinct_cubes" / self.leg_name(g) if self.export_to else None

    def adopt_cropped(self, g):
        """Once a leg has been cropped, work on the cropped cubes themselves.

        This is what makes the order sane: crop first, then label what you can
        actually see. After this the group's cubes ARE the files on disk, all
        on one grid with zero offsets, so a mask needs no warping at all.
        """
        g["cropped"] = False
        d = self.cropped_dir(g)
        if d is None or not (d / "_alignment.json").exists():
            return g
        try:
            man = json.loads((d / "_alignment.json").read_text())
        except Exception:
            return g
        cubes = []
        for c in g["cubes"]:
            path = d / Path(c["path"]).name
            if not path.exists():
                return g                       # half-written: stay on the raw cubes
            cubes.append({**c, "path": path})
        g["raw_cubes"] = g.get("raw_cubes", g["cubes"])
        g["raw_size"] = g.get("raw_size", (g["width"], g["height"]))
        g["cubes"] = cubes
        g["width"] = int(man["output_size"]["width"])
        g["height"] = int(man["output_size"]["height"])
        g["cropped"] = True
        g["crop_box"] = man["box_in_reference"]
        g["meta"] = None
        return g

    def uncrop(self, g):
        """Throw the cropped copy away and go back to the raw cubes."""
        d = self.cropped_dir(g)
        if d and d.exists():
            shutil.rmtree(d)
        labels = self.export_to / "Distinct_cubes_labels" / self.leg_name(g)
        if labels.exists():
            shutil.rmtree(labels)
        if g.get("raw_cubes"):
            g["cubes"] = g["raw_cubes"]
            g["width"], g["height"] = g["raw_size"]
        g["cropped"] = False
        g.pop("crop_box", None)
        g["meta"] = None
        return g

    # -- paths -------------------------------------------------------------
    def group_dir(self, g):
        # A cropped leg keeps its labels next to the cubes they belong to.
        if g.get("cropped") and self.export_to:
            return self.export_to / "Distinct_cubes_labels" / self.leg_name(g)
        return self.labels_dir / g["rel_dir"] if g["rel_dir"] else self.labels_dir

    def align_path(self, g):
        return self.group_dir(g) / "align.json"

    def mask_path(self, g, cube):
        return self.group_dir(g) / f"{cube['name']}.png"

    def meta_path(self, g, cube):
        # After cropping the cubes share a grid and therefore a mask, so seven
        # identical sidecars would be noise: one per leg says the same thing.
        if g.get("cropped"):
            return self.group_dir(g) / "labels.json"
        return self.group_dir(g) / f"{cube['name']}.json"

    def preview_path(self, g, cube):
        return self.group_dir(g) / "_previews" / f"{cube['name']}_overlay.png"

    # -- alignment ---------------------------------------------------------
    def align(self, g):
        """The group's stored alignment, or a fresh unaligned one."""
        names = [c["name"] for c in g["cubes"]]
        if g.get("cropped"):
            # Already resampled onto one grid — there is nothing left to shift.
            man = {}
            try:
                man = json.loads((self.cropped_dir(g) / "_alignment.json").read_text())
            except Exception:
                pass
            scores = {Path(c["file"]).stem: c.get("score", 1.0) for c in man.get("cubes", [])}
            ref_name = Path(man.get("reference", names[0])).stem
            done = False
            meta = self.meta_path(g, g["cubes"][0])
            if meta.exists():
                try:
                    done = bool(json.loads(meta.read_text()).get("done", False))
                except Exception:
                    pass
            return {"reference": names.index(ref_name) if ref_name in names else 0,
                    "offsets": [[0, 0] for _ in names],
                    "scores": [float(scores.get(n, 1.0)) for n in names],
                    "roi": None, "done": done, "auto": True}
        path = self.align_path(g)
        if path.exists():
            try:
                blob = json.loads(path.read_text())
                by_name = {Path(k).stem: v for k, v in blob.get("cubes", {}).items()}
                ref_name = Path(blob.get("reference", "")).stem
                if all(n in by_name for n in names) and ref_name in names:
                    return {
                        "reference": names.index(ref_name),
                        "offsets": [[int(by_name[n]["dx"]), int(by_name[n]["dy"])] for n in names],
                        "scores": [float(by_name[n].get("score", 0.0)) for n in names],
                        "roi": blob.get("roi"),
                        "done": bool(blob.get("done", False)),
                        "auto": bool(blob.get("auto", False)),
                    }
                print(f"! {path} does not match the cubes in {g['name']}, ignoring it")
            except Exception:
                print(f"! could not read {path}, ignoring it")
        return {"reference": 0, "offsets": [[0, 0] for _ in names],
                "scores": [0.0] * len(names), "roi": None, "done": False, "auto": False}

    def save_align(self, g, align):
        names = [c["name"] for c in g["cubes"]]
        ref = int(align["reference"])
        offsets = [[int(dx), int(dy)] for dx, dy in align["offsets"]]
        blob = {
            "reference": g["cubes"][ref]["rel"],
            "convention": ALIGN_CONVENTION,
            "cubes": {g["cubes"][i]["rel"]: {"dx": offsets[i][0], "dy": offsets[i][1],
                                             "score": round(float(align["scores"][i]), 4)}
                      for i in range(len(names))},
            "overlap_box": register.overlap_box(offsets, g["width"], g["height"]),
            "roi": align.get("roi"),
            "width": g["width"], "height": g["height"],
            "auto": bool(align.get("auto", False)),
            "done": bool(align.get("done", False)),
            "updated": _now(),
        }
        d = self.group_dir(g)
        d.mkdir(parents=True, exist_ok=True)
        self.align_path(g).write_text(json.dumps(blob, indent=2))
        return blob

    def auto_align(self, g, ref=None):
        """Register every cube of the group. Returns the new alignment."""
        images = [self.store.mean(c["path"]) for c in g["cubes"]]
        offsets, ref_idx, records = register.register(images, ref)
        old = self.align(g)
        return {"reference": int(ref_idx),
                "offsets": [[int(r["dx"]), int(r["dy"])] for r in records],
                "scores": [float(r["score"]) for r in records],
                "scores_unshifted": [float(r["score_unshifted"]) for r in records],
                "roi": old.get("roi"), "done": old.get("done", False), "auto": True}

    # -- masks -------------------------------------------------------------
    def read_mask(self, g):
        """The group's mask, in the frame of whichever cube is the reference.

        The reference cube's own PNG *is* the reference-frame mask (its offset
        is (0, 0) by definition), so no extra canonical file is needed.
        """
        align = self.align(g)
        cube = g["cubes"][align["reference"]]
        path = self.mask_path(g, cube)
        if not path.exists():
            return None
        arr = read_mask_png(path)
        if arr.shape != (g["height"], g["width"]):
            print(f"! {path.name} is {arr.shape}, group is {(g['height'], g['width'])} — ignoring")
            return None
        return arr

    def write_masks(self, g, mask_ref, align, done=None):
        """Warp the shared mask into every cube's frame and write them all."""
        if mask_ref.shape != (g["height"], g["width"]):
            raise ValueError(f"mask is {mask_ref.shape}, group is {(g['height'], g['width'])}")
        if done is not None:
            align = dict(align, done=bool(done))

        d = self.group_dir(g)
        d.mkdir(parents=True, exist_ok=True)
        blob = self.save_align(g, align)
        names = {c["id"]: c["name"] for c in self.classes}
        ref = int(align["reference"])
        out = []

        for i, cube in enumerate(g["cubes"]):
            dx, dy = align["offsets"][i]
            arr = mask_ref if i == ref else register.ref_to_cube(mask_ref, dx, dy, UNLABELED)

            # Written beside the target and renamed: a crash mid-save can then
            # never leave a half-written mask where a good one used to be.
            save_mask_png(arr, self.mask_path(g, cube), self.classes)

            ids, counts = np.unique(arr, return_counts=True)
            per_class = {int(k): int(n) for k, n in zip(ids, counts) if int(k) != UNLABELED}
            labeled = int(sum(per_class.values()))
            info = {
                "cube": cube["rel"],
                "mask": path.name,
                "width": g["width"], "height": g["height"], "bands": cube["bands"],
                "unlabeled": UNLABELED,
                "labeled_px": labeled,
                "labeled_fraction": round(labeled / (g["width"] * g["height"]), 4),
                "class_px": {str(k): v for k, v in sorted(per_class.items())},
                "class_names": {str(k): names.get(k, f"class {k}") for k in sorted(per_class)},
                "done": bool(align.get("done", False)),
                # the shared annotation this cube's mask came from
                "group": g["rel_dir"] or ".",
                "reference": g["cubes"][ref]["rel"],
                "dx": int(dx), "dy": int(dy),
                "align_score": round(float(align["scores"][i]), 4),
                "align_convention": ALIGN_CONVENTION,
                "roi": align.get("roi"),
                "updated": _now(),
            }
            if not g.get("cropped"):
                self.meta_path(g, cube).write_text(json.dumps(info, indent=2))
            out.append(info)

        if g.get("cropped"):
            ref_info = out[ref]
            self.meta_path(g, g["cubes"][ref]).write_text(json.dumps({
                "leg": self.leg_name(g),
                "masks": [c["name"] + ".png" for c in g["cubes"]],
                "note": "every cube of this leg shares this mask; the copies are byte-identical",
                "width": ref_info["width"], "height": ref_info["height"],
                "unlabeled": UNLABELED,
                "labeled_px": ref_info["labeled_px"],
                "labeled_fraction": ref_info["labeled_fraction"],
                "class_px": ref_info["class_px"], "class_names": ref_info["class_names"],
                "done": bool(align.get("done", False)),
                "updated": _now(),
            }, indent=2))

        # A palette mask is already legible in any viewer, so a cropped leg
        # does not need an overlay image as well.
        if self.previews and not g.get("cropped"):
            # Every cube only when the group is finished: a preview per cube is
            # the clearest proof the alignment is right, but it is not worth
            # rebuilding seven of them on a one-minute autosave.
            targets = g["cubes"] if align.get("done") else [g["cubes"][ref]]
            for cube in targets:
                i = g["cubes"].index(cube)
                dx, dy = align["offsets"][i]
                arr = mask_ref if i == ref else register.ref_to_cube(mask_ref, dx, dy, UNLABELED)
                try:
                    self._write_preview(g, cube, arr)
                except Exception:
                    traceback.print_exc()

        ref_info = out[ref]
        return {"group": {"labeled_px": ref_info["labeled_px"],
                          "labeled_fraction": ref_info["labeled_fraction"],
                          "class_px": ref_info["class_px"],
                          "done": bool(align.get("done", False))},
                "align": blob, "cubes": out}

    def _write_preview(self, g, cube, arr, max_width=800):
        """A small overlay image, purely so a mask can be checked in Finder."""
        base = self.store.mean(cube["path"])
        lo, hi = self.store.bounds(cube["path"], base, "mean", 1.0, 99.0)
        grey = _stretch(base, lo, hi)
        rgb = np.dstack([grey, grey, grey]).astype(np.float32)
        for c in self.classes:
            hit = arr == c["id"]
            if hit.any():
                colour = np.array([int(c["color"][i:i + 2], 16) for i in (1, 3, 5)], np.float32)
                rgb[hit] = rgb[hit] * 0.45 + colour * 0.55
        im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.BILINEAR)
        out = self.preview_path(g, cube)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out)


    # -- exporting a finished leg ------------------------------------------
    def leg_name(self, g):
        """12.09/Rat_2/Left_leg/Distinct_Cubes -> 12.09_Rat_2_Left_leg."""
        parts = [p for p in Path(g["rel_dir"]).parts if p not in (".", "")]
        if parts and parts[-1].lower() == "distinct_cubes":
            parts = parts[:-1]
        return "_".join(parts).replace(" ", "_") or self.input_dir.name

    def export_group(self, g, align, box=None):
        """Write this leg aligned and cropped, with its labels, to the destination.

        The box is chosen in the reference cube's frame, and every cube is cut
        at the same anatomical rectangle — its own box, offset by its own
        shift. The result is a folder of cubes on a common pixel grid, and one
        label image that is therefore correct for all of them.

        The box is clamped to the region every cube actually covers. Beyond
        that at least one cube has no pixels, and padding the gap would invent
        data that a segmentation model would happily learn.
        """
        if self.export_to is None:
            raise ValueError("no export destination; start with --export-to DIR")

        offsets = align["offsets"]
        overlap = register.overlap_box(offsets, g["width"], g["height"])
        box = [int(v) for v in (box or align.get("roi") or overlap)]
        clamped = [max(box[0], overlap[0]), max(box[1], overlap[1]),
                   min(box[2], overlap[2]), min(box[3], overlap[3])]
        if clamped[2] - clamped[0] < 8 or clamped[3] - clamped[1] < 8:
            raise ValueError("the box does not overlap the region all cubes cover")
        was_clamped = clamped != box
        box = clamped
        x0, y0, x1, y1 = box

        mask_ref = self.read_mask(g)
        if mask_ref is None:
            mask_ref = np.full((g["height"], g["width"]), UNLABELED, np.uint8)
        # The mask is in the reference frame and the box is too, so one crop
        # serves every cube — they are all the same tissue now.
        mask_crop = np.ascontiguousarray(mask_ref[y0:y1, x0:x1])

        name = self.leg_name(g)
        cube_dir = self.export_to / "Distinct_cubes" / name
        label_dir = self.export_to / "Distinct_cubes_labels" / name
        cube_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        ref = int(align["reference"])
        written = []
        for i, cube in enumerate(g["cubes"]):
            dx, dy = (int(v) for v in offsets[i])
            crop = [x0 - dx, y0 - dy, x1 - dx, y1 - dy]
            comments = json.dumps({
                "aligned_cropped_by": "Code/Anotation_tool/annotate.py",
                "leg": name, "source": str(cube["path"]),
                "reference": g["cubes"][ref]["path"].name,
                "offset_to_reference": {"dx": dx, "dy": dy},
                "convention": ALIGN_CONVENTION,
                "source_crop": crop,
                "box_in_reference": box,
            }, separators=(",", ":"))
            nx, ny, nc = write_cropped(cube["path"], cube_dir / cube["path"].name,
                                       crop, comments)

            save_mask_png(mask_crop, label_dir / f"{cube['name']}.png", self.classes)
            written.append({"file": cube["path"].name, "source": str(cube["path"]),
                            "dx": dx, "dy": dy, "source_crop": crop,
                            "score": round(float(align["scores"][i]), 4),
                            "shape": {"C": nc, "Y": ny, "X": nx}})

        ids, counts = np.unique(mask_crop, return_counts=True)
        per_class = {int(k): int(n) for k, n in zip(ids, counts) if int(k) != UNLABELED}
        names = {c["id"]: c["name"] for c in self.classes}
        manifest = {
            "leg": name,
            "source_dir": str((self.input_dir / g["rel_dir"]) if g["rel_dir"] else self.input_dir),
            "reference": g["cubes"][ref]["path"].name,
            "box_in_reference": box,
            "box_was_clamped_to_overlap": was_clamped,
            "overlap_box": overlap,
            "source_size": {"width": g["width"], "height": g["height"]},
            "output_size": {"width": x1 - x0, "height": y1 - y0},
            "convention": ALIGN_CONVENTION,
            "note": ("Every cube here is cut to the same tissue, so pixel (x, y) is the "
                     "same location in all of them and one label image fits them all. "
                     "Whole-pixel shifts only — no resampling."),
            "labels": {"dir": str(label_dir), "unlabeled": UNLABELED,
                       "labeled_px": int(sum(per_class.values())),
                       "class_px": {str(k): v for k, v in sorted(per_class.items())},
                       "class_names": {str(k): names.get(k, f"class {k}")
                                       for k in sorted(per_class)}},
            "cubes": written,
            "written": _now(),
        }
        (cube_dir / "_alignment.json").write_text(json.dumps(manifest, indent=2))
        if self.classes:
            (self.export_to / "Distinct_cubes_labels" / "classes.json").write_text(
                json.dumps({"unlabeled": UNLABELED,
                            "labels": {str(c["id"]): c["name"] for c in self.classes},
                            "classes": self.classes, "updated": _now()}, indent=2))
        return manifest

    # -- listing -----------------------------------------------------------
    def group_json(self, g, align=None):
        align = align or self.align(g)
        ref = align["reference"]
        stats = {}
        p = self.meta_path(g, g["cubes"][ref])
        if p.exists():
            try:
                stats = json.loads(p.read_text())
            except Exception:
                pass
        return {
            "id": g["id"], "name": g["name"], "rel_dir": g["rel_dir"],
            "width": g["width"], "height": g["height"],
            "reference": ref,
            "roi": align.get("roi"),
            "done": bool(align.get("done", False)),
            "auto": bool(align.get("auto", False)),
            "aligned": any(dx or dy for dx, dy in align["offsets"]) or bool(align.get("auto")),
            "labeled_px": stats.get("labeled_px", 0),
            "labeled_fraction": stats.get("labeled_fraction", 0.0),
            "overlap_box": register.overlap_box(align["offsets"], g["width"], g["height"]),
            "leg_name": self.leg_name(g),
            "can_export": self.export_to is not None,
            "cropped": bool(g.get("cropped")),
            "crop_box": g.get("crop_box"),
            "exported": (self.export_to / "Distinct_cubes" / self.leg_name(g) /
                         "_alignment.json").exists() if self.export_to else False,
            "cubes": [{"index": i, "name": c["name"], "rel": c["rel"], "bands": c["bands"],
                       "wavelengths": c["wavelengths"],
                       "dx": align["offsets"][i][0], "dy": align["offsets"][i][1],
                       "score": align["scores"][i]}
                      for i, c in enumerate(g["cubes"])],
        }

    def listing(self):
        return [self.group_json(g) for g in self.groups]


def _normalise(align):
    """Offsets are relative to the reference, so the reference sits at (0, 0).

    Anything else is a disagreement about which frame the mask is in, and a
    silent one — so it is corrected on the way in rather than trusted.
    """
    rx, ry = align["offsets"][align["reference"]]
    if rx or ry:
        align = dict(align, offsets=[[dx - rx, dy - ry] for dx, dy in align["offsets"]])
    return align


# --------------------------------------------------------------------------
# HTTP layer. Small, stdlib-only, bound to localhost.
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "im3-annotator"
    workspace: Workspace = None

    def log_message(self, fmt, *args):
        pass

    def log_error(self, fmt, *args):
        sys.stderr.write("! %s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _fail(self, code, message):
        self._json({"error": message}, code)

    def _group(self, gid):
        try:
            return self.workspace.groups[int(gid)]
        except (ValueError, IndexError):
            return None

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        data, left = b"", n
        while left > 0:
            chunk = self.rfile.read(min(left, 1 << 20))
            if not chunk:
                break
            data += chunk
            left -= len(chunk)
        return data

    def do_GET(self):
        try:
            self._route_get()
        except BrokenPipeError:
            pass
        except Exception as exc:
            traceback.print_exc()
            self._fail(500, f"{type(exc).__name__}: {exc}")

    do_HEAD = do_GET

    def _route_get(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        ws = self.workspace

        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])

        if path == "/api/state":
            return self._json({
                "input_dir": str(ws.input_dir), "labels_dir": str(ws.labels_dir),
                "unlabeled": UNLABELED, "palette": DEFAULT_COLORS,
                "classes": ws.classes, "groups": ws.listing(),
            })

        m = re.fullmatch(r"/api/group/(\d+)/(meta|image|mask)", path)
        if m:
            g = self._group(m.group(1))
            if g is None:
                return self._fail(404, "no such group")
            what = m.group(2)

            if what == "meta":
                return self._json(ws.group_json(g))

            if what == "image":
                try:
                    cube = g["cubes"][int(q.get("cube", 0))]
                except (ValueError, IndexError):
                    return self._fail(404, "no such cube in this group")
                img = render(ws.store, cube["path"], q.get("mode", "single"), q)
                buf = io.BytesIO()
                img.save(buf, "PNG", compress_level=1)    # speed over bytes on localhost
                return self._send(200, buf.getvalue(), "image/png")

            arr = ws.read_mask(g)
            if arr is None:
                return self._send(204, b"", "application/octet-stream")
            return self._send(200, arr.tobytes(), "application/octet-stream",
                              {"X-Width": str(arr.shape[1]), "X-Height": str(arr.shape[0])})

        self._fail(404, "not found")

    def do_POST(self):
        try:
            self._route_post()
        except BrokenPipeError:
            pass
        except Exception as exc:
            traceback.print_exc()
            self._fail(500, f"{type(exc).__name__}: {exc}")

    def _route_post(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        ws = self.workspace

        if path == "/api/classes":
            try:
                classes = ws.save_classes(json.loads(self._body())["classes"])
            except (ValueError, KeyError) as exc:
                return self._fail(400, str(exc))
            return self._json({"classes": classes})

        m = re.fullmatch(r"/api/group/(\d+)/(align|offsets|mask|done|export|uncrop)", path)
        if m:
            g = self._group(m.group(1))
            if g is None:
                return self._fail(404, "no such group")
            what = m.group(2)

            if what == "align":
                body = json.loads(self._body() or b"{}")
                ref = body.get("reference")
                align = ws.auto_align(g, None if ref is None else int(ref))
                ws.save_align(g, align)
                worst = min(align["scores"])
                print(f"  aligned {g['name']}  ref={g['cubes'][align['reference']]['name']}  "
                      f"worst score {worst:.2f}", flush=True)
                return self._json({"group": ws.group_json(g, align),
                                   "scores_unshifted": align.get("scores_unshifted")})

            if what == "offsets":
                body = json.loads(self._body())
                align = ws.align(g)
                if "reference" in body:
                    align["reference"] = int(body["reference"])
                if "offsets" in body:
                    if len(body["offsets"]) != len(g["cubes"]):
                        return self._fail(400, "offsets do not match the group's cubes")
                    align["offsets"] = [[int(o[0]), int(o[1])] for o in body["offsets"]]
                if "scores" in body:
                    align["scores"] = [float(s) for s in body["scores"]]
                if "roi" in body:
                    align["roi"] = body["roi"]
                align["auto"] = bool(body.get("auto", False))
                if not 0 <= align["reference"] < len(g["cubes"]):
                    return self._fail(400, "reference is not a cube of this group")
                align = _normalise(align)
                ws.save_align(g, align)
                return self._json({"group": ws.group_json(g, align)})

            if what == "export":
                body = json.loads(self._body() or b"{}")
                align = ws.align(g)
                if body.get("align"):
                    a = body["align"]
                    if "offsets" in a and len(a["offsets"]) != len(g["cubes"]):
                        return self._fail(400, "offsets do not match the group's cubes")
                    align.update({k: a[k] for k in ("reference", "offsets", "scores", "roi")
                                  if k in a})
                    align["reference"] = int(align["reference"])
                    align["offsets"] = [[int(o[0]), int(o[1])] for o in align["offsets"]]
                if not 0 <= align["reference"] < len(g["cubes"]):
                    return self._fail(400, "reference is not a cube of this group")
                align = _normalise(align)
                ws.save_align(g, align)
                try:
                    man = ws.export_group(g, align, body.get("box"))
                except ValueError as exc:
                    return self._fail(400, str(exc))
                ws.adopt_cropped(g)
                print(f"  cropped {man['leg']}: {len(man['cubes'])} cubes at "
                      f"{man['output_size']['width']}x{man['output_size']['height']}"
                      f" — now labelling the cropped cubes", flush=True)
                return self._json({"export": man, "group": ws.group_json(g)})

            if what == "uncrop":
                ws.uncrop(g)
                print(f"  un-cropped {ws.leg_name(g)} — back to the raw cubes", flush=True)
                return self._json({"group": ws.group_json(g)})

            if what == "done":
                align = ws.align(g)
                align["done"] = bool(json.loads(self._body()).get("done", True))
                ws.save_align(g, align)
                return self._json({"group": ws.group_json(g, align)})

            # mask: the body is the mask, in the frame of the reference named
            # by `align`. That alignment travels WITH the pixels rather than
            # being read back off disk: the mask only means anything alongside
            # the frame it was painted in, and a stored alignment can be one
            # debounced request behind the browser's.
            raw = self._body()
            if len(raw) != g["width"] * g["height"]:
                return self._fail(400, f"expected {g['width'] * g['height']} bytes, got {len(raw)}")
            align = ws.align(g)
            if q.get("align"):
                try:
                    blob = json.loads(q["align"])
                except ValueError:
                    return self._fail(400, "align is not valid JSON")
                if "offsets" in blob and len(blob["offsets"]) != len(g["cubes"]):
                    return self._fail(400, "offsets do not match the group's cubes")
                if "reference" in blob:
                    align["reference"] = int(blob["reference"])
                if "offsets" in blob:
                    align["offsets"] = [[int(o[0]), int(o[1])] for o in blob["offsets"]]
                if "scores" in blob:
                    align["scores"] = [float(x) for x in blob["scores"]]
                if "roi" in blob:
                    align["roi"] = blob["roi"]
            if not 0 <= align["reference"] < len(g["cubes"]):
                return self._fail(400, "reference is not a cube of this group")
            align = _normalise(align)
            mask = np.frombuffer(raw, np.uint8).reshape(g["height"], g["width"])
            done = {"1": True, "0": False}.get(q.get("done"))
            result = ws.write_masks(g, mask, align, done=done)
            print(f"  saved {g['name']}: {len(result['cubes'])} masks, "
                  f"{result['group']['labeled_px']:,} px "
                  f"({result['group']['labeled_fraction'] * 100:.1f}%)", flush=True)
            result["group_meta"] = ws.group_json(g, align)
            return self._json(result)

        self._fail(404, "not found")

    def _static(self, name):
        target = (STATIC / name).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            return self._fail(404, "not found")
        ctype = {".html": "text/html; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}.get(target.suffix, "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)


def _free_port(preferred):
    for port in [preferred] + list(range(preferred + 1, preferred + 40)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"no free port near {preferred}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_dir", help="folder of .im3 cubes to annotate")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-recursive", action="store_true",
                    help="only look at .im3 files directly in the folder")
    ap.add_argument("--no-group", action="store_true",
                    help="annotate every cube on its own instead of one mask per folder")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-previews", action="store_true",
                    help="skip writing the _previews overlay images")
    ap.add_argument("--export-to", type=Path, default=None,
                    help="destination for finished legs: writes Distinct_cubes/<leg>/*.im3 "
                         "aligned and cropped to your box, plus Distinct_cubes_labels/<leg>/. "
                         "Working files go to <dest>/_work.")
    ap.add_argument("--only", action="append", default=[], metavar="TEXT",
                    help="keep only cubes whose path contains TEXT (repeatable)")
    ap.add_argument("--exclude", action="append", default=[], metavar="TEXT",
                    help="skip cubes whose path contains TEXT (repeatable), "
                         "e.g. --exclude 10.09 --exclude 09.09")
    ap.add_argument("--labels-dir", type=Path, default=None,
                    help="where in-progress masks and alignments live "
                         "(default: <dest>/_work, or <input>_labels without a destination)")
    args = ap.parse_args(argv)

    ws = Workspace(args.input_dir, recursive=not args.no_recursive,
                   previews=not args.no_previews, group=not args.no_group,
                   export_to=args.export_to, labels_dir=args.labels_dir,
                   only=args.only, exclude=args.exclude)
    Handler.workspace = ws

    port = _free_port(args.port)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    n_cubes = sum(len(g["cubes"]) for g in ws.groups)

    print(f"  cubes   {n_cubes} in {len(ws.groups)} group(s) under {ws.input_dir}")
    print(f"  work    {ws.labels_dir}")
    if ws.export_to:
        print(f"  export  {ws.export_to}/Distinct_cubes[_labels]/<leg>/")
    print(f"  open    {url}")
    print("  ctrl-c to stop\n", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()
        ws.store.close_all()


if __name__ == "__main__":
    main()
