"""Linear spectral unmixing of the processed .im3 cubes, Nuance-style.

What Nuance does, and what this does too: you pick a region for each structure
on one image, it averages the spectra in those regions into one *endmember* per
structure, and then every pixel of the cube is written as a non-negative
combination of those endmembers. The output is one abundance image per
structure plus a residual image saying how badly the model fitted.

What Nuance cannot do, and this is the whole point of the tool: the same
regions are then re-measured and re-unmixed on *every other version of the same
leg*, so the sixteen-odd ways we have of preparing a leg are compared on one
picture and one number rather than by eye, one file at a time.

A leg under `Data/Processed` exists in two aligned trees (all 217 channels
kept, or the overexposed leading bands removed, 210 channels), and in each tree
as four concatenated cubes (raw, and three z-normalisations) plus the seven
single-excitation cubes it was concatenated from. That is 22 versions of one
leg, and because the aligned/cropped export put every cube of a leg on the same
pixel grid, one set of regions and one label mask are correct for all 22.

    version id             what it is
    all__concat__raw       217 ch, the 7 cubes stacked, raw sensor counts
    all__concat__whole     217 ch, z-normed over the whole 4D cube
    oe__concat__per_chan   210 ch, overexposed bands dropped, z per channel
    oe__ex__365            31 ch, the 365 nm cube alone, overexposed dropped
    ...

Accuracy: each leg carries a hand-painted mask (2 = vessel, 3 = nerve,
4 = muscle, every pixel painted). A version's predicted class is the argmax
over the abundances of the structures that map to a mask class, and that is
scored against the mask twice — over every pixel, and again with the pixels
that defined the endmembers excluded, because those pixels are the ones the
model was fitted on and they flatter it.

Nothing here writes into `Data/Processed`. Regions, abundances, metrics and the
report all land under `Data/Derived/Linear_unmixing/`.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "Anotation_tool"))

from concat_lib import read_cube_info  # noqa: E402

PROCESSED_ROOT = PROJECT / "Data" / "Processed"
DERIVED_ROOT = PROJECT / "Data" / "Derived" / "Linear_unmixing"

# =============================================================================
# The version matrix: which cubes of a leg we unmix and compare.
#
# Only the aligned/cropped trees are in here. The two full-frame trees are
# deliberately left out: their seven excitations are not on a common pixel grid
# (see the cube-misalignment note in the project memory), so one region set and
# one mask cannot be correct across them, and no mask was ever painted for
# them. Adding them would mean drawing regions seven times per leg and scoring
# nothing.
# =============================================================================
TREES = [
    {"key": "all", "dir": "Alligned_Cropped_All_channels_kept",
     "label": "all channels", "short": "all"},
    {"key": "oe", "dir": "Alligned_cropped_overexposed_channels_removed",
     "label": "overexposed removed", "short": "OE-rm"},
]

# Each concatenated cube lives in its own folder, one per normalisation "way"
# built by concat_lib. The key is what appears in a version id.
NORMS = [
    {"key": "raw", "folder": "Raw_Concatenated", "label": "raw counts"},
    {"key": "per_chan", "folder": "Per_Chanell_Norm_1_99_Concat", "label": "z per channel"},
    {"key": "per_exc", "folder": "Per_Cube_Norm_1_99_Concat", "label": "z per excitation"},
    {"key": "whole", "folder": "Overall_4D_Norm_1_99_Concat", "label": "z whole cube"},
]

# The order single-excitation cubes are listed in; a leg's files carry their own
# exposure in the name (325_5000exp.im3), so cubes are matched by this prefix.
EXCITATIONS = ["310", "325", "340", "365", "385", "400", "white"]

IGNORE = 255                       # the masks' unlabeled id, skipped when scoring


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================
# Discovery
# =============================================================================
def _excitation_key(stem):
    """'325_15000exp' -> '325'; 'white_10exp' and the 'whte_8exp' typo -> 'white'."""
    m = re.match(r"\s*(\d{3})", stem)
    return m.group(1) if m else "white"


def _concat_cube(tree_dir, leg, norm):
    """The one concatenated .im3 of a leg for one normalisation, or None.

    Matched by prefix rather than by a table of suffixes, because the suffix is
    whatever concat_lib chose for that way and only the folder is contractual.
    """
    folder = tree_dir / "Concated_cubes" / norm["folder"]
    hits = sorted(folder.glob(f"{leg}_concat*.im3"))
    return hits[0] if hits else None


def _versions_of_leg(root, leg, trees=None):
    """Every cube of one leg across the given trees, in display order."""
    versions = []
    for tree in (trees or TREES):
        tree_dir = root / tree["dir"]
        for norm in NORMS:
            path = _concat_cube(tree_dir, leg, norm)
            if path is None:
                continue
            versions.append({
                "id": f"{tree['key']}__concat__{norm['key']}",
                "tree": tree["key"], "kind": "concat", "key": norm["key"],
                "label": f"{tree['short']} · concat · {norm['label']}",
                "group": f"{tree['label']} — concatenated",
                "path": path,
                "mask": tree_dir / "Concated_cubes_labels" / f"{leg}_concat.png",
            })

        cubes = {_excitation_key(p.stem): p
                 for p in sorted((tree_dir / "Distinct_cubes" / leg).glob("*.im3"))}
        for ex in EXCITATIONS:
            path = cubes.get(ex)
            if path is None:
                continue
            name = "white light" if ex == "white" else f"{ex} nm"
            versions.append({
                "id": f"{tree['key']}__ex__{ex}",
                "tree": tree["key"], "kind": "ex", "key": ex,
                "label": f"{tree['short']} · {name}",
                "group": f"{tree['label']} — single excitation",
                "path": path,
                "mask": tree_dir / "Distinct_cubes_labels" / leg / f"{path.stem}.png",
            })
    return versions


def _channel_table(version):
    """Per-channel (excitation, emission nm) for a cube, from its sidecar or itself.

    A concatenated cube has a `_channels.csv` next to it that says which
    excitation block every one of its 210/217 channels came from; a single
    cube's channels are all one excitation and only the emission varies.
    """
    csv_path = version["path"].with_name(version["path"].stem + "_channels.csv")
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open()))
        return ([r["excitation"] for r in rows],
                [float(r["emission_nm"]) for r in rows])
    info = read_cube_info(version["path"])
    return ([info["excitation"]] * info["C"], list(info["emission_nm"]))


def _fingerprint(version):
    """A cheap content hash of a cube's pixels, to spot versions that are equal.

    Trimming the overexposed leading bands only touches the 385 and 400 nm
    cubes, so most single-excitation cubes appear unchanged in both trees —
    same file, twice in the matrix. Saying so is better than letting two
    identical rows read as two independent results. 64 slices spread over the
    pixel block is plenty to separate cubes that genuinely differ.
    """
    import hashlib
    h = hashlib.sha1()
    h.update(f"{version['Y']}x{version['X']}x{version['C']}|".encode())
    size = version["path"].stat().st_size
    start, span = version["data_offset"], size - version["data_offset"]
    with open(version["path"], "rb") as fh:
        for i in range(64):
            fh.seek(start + (span - 65536) * i // 63)
            h.update(fh.read(65536))
    return h.hexdigest()


def _mark_duplicates(versions, cache_path):
    """Set `duplicate_of` on every version whose cube is byte-equal to an earlier one."""
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    seen, dirty = {}, False
    for v in versions:
        key = f"{v['path']}|{v['path'].stat().st_mtime_ns}|{v['path'].stat().st_size}"
        fp = cache.get(key)
        if fp is None:
            fp, dirty = _fingerprint(v), True
            cache[key] = fp
        v["duplicate_of"] = seen.get(fp)
        seen.setdefault(fp, v["id"])
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return versions


def discover(root=PROCESSED_ROOT, out_root=DERIVED_ROOT, trees=None):
    """{leg: {...}} for every leg that exists in the aligned trees.

    A leg is kept only if all of its versions share one (Y, X) — the grid the
    regions and the mask live on. Any version that disagrees is dropped and
    said so in `warnings`, rather than silently unmixing against regions that
    point at different tissue.
    """
    root = Path(root)
    wanted = [t for t in TREES if not trees or t["key"] in trees or t["dir"] in trees]
    if not wanted:
        raise ValueError(f"no such tree: {trees} — known keys are "
                         + ", ".join(t["key"] for t in TREES))
    legs = {}
    for tree in wanted:
        distinct = root / tree["dir"] / "Distinct_cubes"
        if distinct.is_dir():
            for d in sorted(p for p in distinct.iterdir() if p.is_dir()):
                legs.setdefault(d.name, None)

    out = {}
    for leg in sorted(legs):
        versions, shapes, warnings = [], {}, []
        for v in _versions_of_leg(root, leg, wanted):
            try:
                info = read_cube_info(v["path"])
            except Exception as exc:                       # a truncated or odd file
                warnings.append(f"{v['id']}: unreadable ({exc})")
                continue
            v |= {"Y": info["Y"], "X": info["X"], "C": info["C"],
                  "data_offset": info["data_offset"],
                  "has_mask": v["mask"].exists()}
            shapes.setdefault((info["Y"], info["X"]), []).append(v["id"])
            versions.append(v)
        if not versions:
            continue

        grid = max(shapes, key=lambda s: len(shapes[s]))   # the majority grid
        for shape, ids in shapes.items():
            if shape != grid:
                warnings.append(f"dropped {', '.join(ids)}: {shape[1]}x{shape[0]} "
                                f"is not the leg's grid {grid[1]}x{grid[0]}")
        versions = [v for v in versions if (v["Y"], v["X"]) == grid]
        missing = [v["id"] for v in versions if not v["has_mask"]]
        if missing:
            warnings.append(f"no label mask for {', '.join(missing)} — "
                            "unmixed, but not scored")

        _mark_duplicates(versions, Path(out_root) / "_fingerprints.json")
        dups = [v["id"] for v in versions if v["duplicate_of"]]
        if dups:
            warnings.append(f"{len(dups)} version(s) are byte-identical to another "
                            "and are marked as duplicates, not separate results")

        out[leg] = {"leg": leg, "Y": grid[0], "X": grid[1],
                    "versions": versions, "warnings": warnings}
    return out


def classes_json(root=PROCESSED_ROOT):
    """The shared id -> name/colour table the annotator wrote."""
    for tree in TREES:
        p = root / tree["dir"] / "Concated_cubes_labels" / "classes.json"
        if p.exists():
            return json.loads(p.read_text())
    return {"classes": [{"id": 2, "name": "vessel", "color": "#ff0000"},
                        {"id": 3, "name": "nerve", "color": "#0037ff"},
                        {"id": 4, "name": "muscle", "color": "#00ff40"}]}


# =============================================================================
# Cubes
# =============================================================================
def open_cube(version):
    """A read-only (Y, X, C) uint16 memory map of one version's cube.

    Pixels are band-interleaved-by-pixel, so this shape is the file's own
    layout and one pixel's whole spectrum is contiguous — which is exactly the
    access pattern unmixing wants. Cubes here are 18-100 MB, but they are still
    mapped rather than read, so several versions can be open at once.
    """
    return np.memmap(version["path"], dtype="<u2", mode="r",
                     offset=version["data_offset"],
                     shape=(version["Y"], version["X"], version["C"]))


# =============================================================================
# Selections: the structures, their regions, and the settings they were run with
# =============================================================================
DEFAULT_SOLVER = {
    "method": "nnls",              # nnls | ls
    "stat": "mean",                # how a region becomes one spectrum
    "extra": "none",               # none | constant | dark
    "normalize_endmembers": False,
}


def leg_dir(leg, out_root=DERIVED_ROOT):
    return Path(out_root) / leg


def selection_path(leg, out_root=DERIVED_ROOT):
    return leg_dir(leg, out_root) / "selection.json"


def rois_path(leg, out_root=DERIVED_ROOT):
    return leg_dir(leg, out_root) / "rois.png"


def default_structures(root=PROCESSED_ROOT):
    """One structure per painted mask class — what you get before you touch it."""
    return [{"name": c["name"], "color": c["color"], "class_id": int(c["id"])}
            for c in classes_json(root)["classes"] if int(c["id"]) != 1]


def load_selection(leg, out_root=DERIVED_ROOT):
    p = selection_path(leg, out_root)
    if not p.exists():
        return None
    sel = json.loads(p.read_text())
    sel["solver"] = DEFAULT_SOLVER | sel.get("solver", {})
    return sel


def save_selection(leg, sel, out_root=DERIVED_ROOT):
    p = selection_path(leg, out_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    sel = dict(sel, leg=leg, updated=_utc(), written_by="Code/Linear_unmixing_tool")
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(sel, indent=1))
    tmp.replace(p)
    return sel


def load_rois(leg, shape, out_root=DERIVED_ROOT):
    """(Y, X) uint8: 0 = unassigned, i+1 = the i-th structure's region."""
    p = rois_path(leg, out_root)
    if not p.exists():
        return np.zeros(shape, np.uint8)
    a = np.array(Image.open(p))
    if a.shape != shape:                     # a stale file from a different crop
        return np.zeros(shape, np.uint8)
    return a.astype(np.uint8)


def save_rois(leg, rois, out_root=DERIVED_ROOT):
    p = rois_path(leg, out_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".png.part")
    Image.fromarray(rois.astype(np.uint8), mode="L").save(tmp, format="PNG", optimize=True)
    tmp.replace(p)


def seed_rois_from_mask(mask, structures, fraction=0.02, min_px=200, seed=0):
    """Regions sampled out of the label mask, as a starting point for editing.

    Deliberately a *sample* and not the whole class: the endmember is meant to
    come from a region you chose, and taking every labelled pixel makes the
    accuracy score meaningless (the model would be fitted on every pixel it is
    then graded on). Pixels are taken in a few compact blobs rather than
    scattered, so what you get looks like something you would have painted.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(mask.shape, np.uint8)
    for i, s in enumerate(structures):
        cid = s.get("class_id")
        if cid is None:
            continue
        ys, xs = np.nonzero(mask == cid)
        if ys.size == 0:
            continue
        want = max(min_px, int(ys.size * fraction))
        want = min(want, ys.size)
        # Grow square blobs around a handful of random seed pixels until the
        # budget is spent; radius follows the budget so a thin nerve still gets
        # several small patches rather than one square that leaves the class.
        n_blobs = 6
        r = max(2, int(np.sqrt(want / n_blobs) / 2))
        picked = np.zeros(mask.shape, bool)
        for _ in range(200):
            if picked.sum() >= want:
                break
            k = rng.integers(ys.size)
            y, x = int(ys[k]), int(xs[k])
            y0, y1 = max(0, y - r), min(mask.shape[0], y + r + 1)
            x0, x1 = max(0, x - r), min(mask.shape[1], x + r + 1)
            block = mask[y0:y1, x0:x1] == cid
            picked[y0:y1, x0:x1] |= block
        out[picked & (out == 0)] = i + 1
    return out


# =============================================================================
# Endmembers
# =============================================================================
def endmembers(cube, rois, n_structures, stat="mean", extra="none",
               normalize=False, rng_seed=0):
    """(C, K) endmember matrix for one cube from the shared region map.

    The regions are pixel coordinates, identical for every version of the leg;
    what differs between versions is the spectrum those pixels hold, which is
    the whole comparison. Returns the matrix, the per-structure pixel counts,
    and the names of any extra components appended after the structures.
    """
    C = cube.shape[2]
    cols, counts = [], []
    for i in range(n_structures):
        ys, xs = np.nonzero(rois == i + 1)
        counts.append(int(ys.size))
        if ys.size == 0:
            cols.append(np.zeros(C, np.float64))
            continue
        # One gather of (n, C) — the spectra of a region are scattered rows of
        # the map, and a region is thousands of pixels at most.
        block = np.asarray(cube[ys, xs, :], dtype=np.float64)
        cols.append(np.median(block, axis=0) if stat == "median" else block.mean(axis=0))

    extras = []
    if extra == "constant":
        # A flat component. Soaks up an offset that is the same in every
        # channel, e.g. a camera pedestal.
        cols.append(np.ones(C, np.float64))
        extras.append("constant")
    elif extra == "dark":
        # A per-channel floor, measured as the 1st percentile of a pixel
        # sample. The z-normalised cubes carry a large per-channel additive
        # offset by construction (concat_lib maps z into uint16 with a shared
        # window), and this is the component that can absorb it.
        rng = np.random.default_rng(rng_seed)
        n = cube.shape[0] * cube.shape[1]
        take = min(20000, n)
        idx = rng.choice(n, take, replace=False)
        flat = np.asarray(cube).reshape(n, C)
        cols.append(np.percentile(flat[idx].astype(np.float64), 1.0, axis=0))
        extras.append("dark")

    E = np.stack(cols, axis=1)
    if normalize:
        norms = np.linalg.norm(E, axis=0)
        norms[norms == 0] = 1.0
        E = E / norms
    return E, counts, extras


# =============================================================================
# The solvers
# =============================================================================
def _subsets(K):
    """Every non-empty subset of K endmembers, as index arrays."""
    return [np.array([k for k in range(K) if m >> k & 1], dtype=np.intp)
            for m in range(1, 1 << K)]


def unmix_block(B, E, G, subsets, method="nnls"):
    """Abundances and squared residual for a block of pixels.

    B: (n, C) float64 spectra. E: (C, K). G: (K, K) = E'E.

    `nnls` is exact, not an approximation: the non-negative least squares
    solution is the unconstrained fit on one subset of the endmembers, and it
    is the feasible subset whose residual is smallest. With K under about six
    there are only a few dozen subsets, so enumerating them and letting numpy
    do every pixel of every subset at once is both exact and far faster than
    iterating an active-set solver per pixel.

    Writing ||b - E_S a_S||^2 = ||b||^2 - c_S'a_S for the least-squares a_S on
    subset S, picking the largest c_S'a_S among the subsets with a_S >= 0 both
    finds the solution and hands back its residual for free.
    """
    n, K = B.shape[0], E.shape[1]
    c = E.T @ B.T                                  # (K, n)
    bb = np.einsum("ij,ij->i", B, B)               # ||b||^2 per pixel

    if method == "ls":
        A = np.linalg.pinv(E) @ B.T                # (K, n), may be negative
        # A least-squares fit is orthogonal to its own residual, so the same
        # ||b||^2 - c'a identity used below gives the residual without ever
        # forming E @ A.
        return A.T, np.maximum(bb - np.einsum("kn,kn->n", c, A), 0.0)

    A = np.zeros((K, n))
    best = np.zeros(n)                             # the all-zero fit scores 0
    for idx in subsets:
        Gs = G[np.ix_(idx, idx)]
        try:
            a = np.linalg.solve(Gs, c[idx])
        except np.linalg.LinAlgError:              # two endmembers are identical
            a = np.linalg.pinv(Gs) @ c[idx]
        score = np.einsum("ij,ij->j", c[idx], a)
        better = np.all(a >= 0, axis=0) & (score > best)
        if not better.any():
            continue
        sel = np.nonzero(better)[0]
        best[sel] = score[sel]
        A[:, sel] = 0.0
        A[np.ix_(idx, sel)] = a[:, sel]
    return A.T, np.maximum(bb - best, 0.0)


def unmix_cube(cube, E, method="nnls", rows=64, progress=None):
    """Abundances, the per-pixel residual, and how big that residual really is.

    Returns (K, Y, X) abundances, a (Y, X) residual RMSE in the cube's own
    units, and the *relative* residual ``sqrt(Σ‖b−Ea‖² / Σ‖b‖²)`` — the fraction
    of the signal the model failed to explain.

    The relative number is the one worth comparing between versions. The
    absolute RMSE is in whatever units the cube holds, and the z-normalised
    cubes are mapped into the full uint16 range while the raw ones sit under
    4040 counts, so their residuals differ by a factor of sixteen for reasons
    that have nothing to do with how well anything fitted.
    """
    Y, X, C = cube.shape
    K = E.shape[1]
    G = E.T @ E
    subsets = _subsets(K) if method == "nnls" else None
    A = np.empty((K, Y, X), np.float32)
    R = np.empty((Y, X), np.float32)
    sse = sst = 0.0
    for y in range(0, Y, rows):
        block = np.asarray(cube[y:y + rows], dtype=np.float64)
        h = block.shape[0]
        flat = block.reshape(h * X, C)
        a, r = unmix_block(flat, E, G, subsets, method)
        A[:, y:y + h] = a.T.reshape(K, h, X).astype(np.float32)
        R[y:y + h] = np.sqrt(r / C).reshape(h, X).astype(np.float32)
        sse += float(r.sum())
        sst += float(np.einsum("ij,ij->", flat, flat))
        if progress:
            progress(min(y + rows, Y) / Y)
    return A, R, float(np.sqrt(sse / sst)) if sst > 0 else float("nan")


# =============================================================================
# Scoring
# =============================================================================
def confusion(pred, truth, ids):
    """(len(ids), len(ids)) counts, rows = truth, cols = prediction."""
    index = {c: i for i, c in enumerate(ids)}
    lut = np.full(256, -1, np.int64)
    for c, i in index.items():
        lut[c] = i
    t, p = lut[truth], lut[pred]
    ok = t >= 0
    # Predictions outside `ids` (an unmapped extra component winning a pixel)
    # are folded into a final "other" column so they are counted as errors
    # rather than quietly dropped.
    pcol = np.where(p[ok] >= 0, p[ok], len(ids))
    cm = np.zeros((len(ids), len(ids) + 1), np.int64)
    np.add.at(cm, (t[ok], pcol), 1)
    return cm


def metrics_from_confusion(cm, ids, names):
    """Overall accuracy, mean IoU, and per-class precision/recall/IoU/dice."""
    total = int(cm.sum())
    correct = int(np.trace(cm[:, :len(ids)]))
    per = {}
    ious = []
    for i, cid in enumerate(ids):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        per[names[i]] = {
            "class_id": cid, "support": int(cm[i].sum()),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "iou": iou,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        }
        if not np.isnan(iou):
            ious.append(iou)
    return {
        "pixels": total,
        "accuracy": correct / total if total else float("nan"),
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        "per_class": per,
        "confusion": cm.tolist(),
    }


def score(classmap, mask, roi, ids, names):
    """Scored over every labelled pixel, and again with the regions taken out.

    The second number is the one to trust. The endmembers are the mean spectra
    of the region pixels, so those pixels are close to fitted by construction
    and inflate the first number by however much of the frame they cover.
    """
    valid = mask != IGNORE
    both = {}
    both["all"] = metrics_from_confusion(
        confusion(classmap[valid], mask[valid], ids), ids, names)
    out = valid & (roi == 0)
    both["outside_rois"] = metrics_from_confusion(
        confusion(classmap[out], mask[out], ids), ids, names)
    return both


# =============================================================================
# Pictures
# =============================================================================
def _stretch(img, lo=1.0, hi=99.0, gamma=1.0):
    """Percentile-clipped 0-255. The sensor clips at 4040, so never min-max."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, np.uint8)
    a, b = np.percentile(finite, [lo, hi])
    if b <= a:
        b = a + 1.0
    v = np.clip((img - a) / (b - a), 0, 1)
    if gamma != 1.0:
        v = v ** (1.0 / gamma)
    return (v * 255 + 0.5).astype(np.uint8)


def png_bytes(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def render_cube(cube, mode="mean", band=0, rgb=(0, 0, 0), lo=1.0, hi=99.0, gamma=1.0):
    """A greyscale or false-colour view of one cube, as PNG bytes."""
    if mode == "rgb":
        planes = [_stretch(np.asarray(cube[:, :, int(c)], np.float32), lo, hi, gamma)
                  for c in rgb]
        return png_bytes(np.stack(planes, axis=-1), "RGB")
    if mode == "band":
        img = np.asarray(cube[:, :, int(band)], np.float32)
    else:
        idx = np.unique(np.linspace(0, cube.shape[2] - 1, min(cube.shape[2], 24))
                        .round().astype(int))
        img = np.zeros(cube.shape[:2], np.float64)
        for c in idx:
            img += cube[:, :, int(c)]
        img /= len(idx)
    return png_bytes(_stretch(img, lo, hi, gamma), "L")


def colour_map(ids_image, palette):
    """Paint an id image with {id: '#rrggbb'}; anything unlisted stays black."""
    out = np.zeros(ids_image.shape + (3,), np.uint8)
    for cid, colour in palette.items():
        out[ids_image == cid] = hex_rgb(colour)
    return out


def abundance_png(plane, colour=None, hi=99.5):
    """One abundance image, clipped at a high percentile and tinted."""
    v = _stretch(plane, 0.0, hi)
    if colour is None:
        return png_bytes(v, "L")
    r, g, b = hex_rgb(colour)
    rgb = np.stack([(v * (r / 255)).astype(np.uint8),
                    (v * (g / 255)).astype(np.uint8),
                    (v * (b / 255)).astype(np.uint8)], axis=-1)
    return png_bytes(rgb, "RGB")


# =============================================================================
# Running one leg
# =============================================================================
def run_leg(leg, legs=None, out_root=DERIVED_ROOT, save_npy=False,
            progress=None, cancelled=None, root=PROCESSED_ROOT, trees=None):
    """Unmix every version of one leg with its saved regions, and score them.

    Writes, under `<out_root>/<leg>/`:

        selection.json            what was run (copied forward with results)
        rois.png                  the regions, on the leg's grid
        metrics.csv               one row per version, the table to read
        metrics.json              the same with the confusion matrices
        <version>/classmap.png    argmax of the scored abundances, mask colours
        <version>/abundance_*.png one per structure
        <version>/residual.png    how badly the model fitted, per pixel
        <version>/endmembers.csv  the spectra this version measured
    """
    legs = legs or discover(root, out_root, trees)
    if leg not in legs:
        raise KeyError(f"unknown leg {leg!r}")
    entry = legs[leg]
    sel = load_selection(leg, out_root)
    if sel is None:
        raise ValueError(f"{leg}: nothing selected yet — draw regions first")

    structures = sel["structures"]
    solver = DEFAULT_SOLVER | sel.get("solver", {})
    rois = load_rois(leg, (entry["Y"], entry["X"]), out_root)
    if not rois.any():
        raise ValueError(f"{leg}: the region map is empty")

    scored = [i for i, s in enumerate(structures) if s.get("class_id") is not None]
    ids = [int(structures[i]["class_id"]) for i in scored]
    names = [structures[i]["name"] for i in scored]
    palette = {int(s["class_id"]): s["color"] for s in structures
               if s.get("class_id") is not None}

    # The two pictures every version is read against: the hand-painted truth in
    # the same colours the class maps use, and the regions the endmembers came
    # from. Written once per leg so the report can put them beside the results.
    truth_src = next((v for v in entry["versions"] if v["has_mask"]), None)
    if truth_src is not None:
        mask0 = np.array(Image.open(truth_src["mask"]))
        (leg_dir(leg, out_root) / "truth.png").write_bytes(
            png_bytes(colour_map(mask0, palette), "RGB"))

    # The regions are drawn *on the tissue*, not on a black field. A few hundred
    # coloured pixels floating on black cannot be checked against anything; over
    # the reference cube you can see at a glance whether the vessel region is
    # actually on the vessel.
    ref = next((v for v in entry["versions"]
                if v["id"] == sel.get("reference")), entry["versions"][0])
    ref_cube = open_cube(ref)
    idx = np.unique(np.linspace(0, ref["C"] - 1, min(ref["C"], 16)).round().astype(int))
    grey = np.zeros((entry["Y"], entry["X"]), np.float64)
    for c in idx:
        grey += ref_cube[:, :, int(c)]
    del ref_cube
    shot = np.repeat(_stretch(grey / len(idx), 1.0, 99.0)[:, :, None], 3, axis=2)
    shot = (shot * 0.55).astype(np.uint8)          # dim it so the regions carry
    for i, st in enumerate(structures):
        shot[rois == i + 1] = hex_rgb(st["color"])
    (leg_dir(leg, out_root) / "rois_preview.png").write_bytes(png_bytes(shot, "RGB"))

    rows, detail = [], []
    for n, v in enumerate(entry["versions"]):
        if cancelled and cancelled():
            break
        if progress:
            progress(n, len(entry["versions"]), v["id"], "opening")
        t0 = time.time()
        vdir = leg_dir(leg, out_root) / v["id"]
        vdir.mkdir(parents=True, exist_ok=True)

        cube = open_cube(v)
        E, counts, extras = endmembers(cube, rois, len(structures),
                                       stat=solver["stat"], extra=solver["extra"],
                                       normalize=solver["normalize_endmembers"])
        if progress:
            progress(n, len(entry["versions"]), v["id"], "unmixing")
        A, R, rel = unmix_cube(cube, E, method=solver["method"])
        del cube

        # The predicted class is the best-fitting *scored* structure. An extra
        # component, or a structure with no class of its own, still competes
        # for signal in the fit but cannot win a pixel here — it is not one of
        # the things the mask distinguishes.
        stack = A[scored]
        classmap = np.take(np.array(ids, np.uint8), stack.argmax(axis=0))

        row = {"leg": leg, "version": v["id"], "label": v["label"],
               "tree": v["tree"], "kind": v["kind"], "key": v["key"],
               "channels": int(v["C"]), "method": solver["method"],
               "extra": solver["extra"], "duplicate_of": v.get("duplicate_of") or "",
               "residual_rmse": float(R.mean()),
               # The comparable one: the absolute RMSE above is in the cube's
               # own units, which differ by ~16x between raw and z-normalised.
               "residual_rel": rel,
               "seconds": round(time.time() - t0, 2)}

        entry_detail = {**row, "endmember_pixels": dict(zip(
            [s["name"] for s in structures], counts)), "extras": extras}

        if v["has_mask"]:
            mask = np.array(Image.open(v["mask"]))
            if mask.shape != classmap.shape:
                row["note"] = "mask shape mismatch — not scored"
            else:
                m = score(classmap, mask, rois, ids, names)
                entry_detail["metrics"] = m
                row["accuracy"] = m["all"]["accuracy"]
                row["mean_iou"] = m["all"]["mean_iou"]
                row["accuracy_outside_rois"] = m["outside_rois"]["accuracy"]
                row["mean_iou_outside_rois"] = m["outside_rois"]["mean_iou"]
                for cname in names:
                    row[f"iou_{cname}"] = m["outside_rois"]["per_class"][cname]["iou"]
        else:
            row["note"] = "no label mask"

        (vdir / "classmap.png").write_bytes(png_bytes(colour_map(classmap, palette), "RGB"))
        (vdir / "residual.png").write_bytes(png_bytes(_stretch(R, 0.0, 99.0), "L"))
        for i, s in enumerate(structures):
            (vdir / f"abundance_{s['name']}.png").write_bytes(
                abundance_png(A[i], s["color"]))
        for j, name in enumerate(extras):
            (vdir / f"abundance_{name}.png").write_bytes(abundance_png(A[len(structures) + j]))
        _write_endmembers(vdir / "endmembers.csv", v, E,
                          [s["name"] for s in structures] + extras)
        if save_npy:
            np.save(vdir / "abundance.npy", A)
        (vdir / "metrics.json").write_text(json.dumps(entry_detail, indent=1))

        rows.append(row)
        detail.append(entry_detail)

    _write_metrics(leg_dir(leg, out_root), rows, detail, sel, entry)
    if progress:
        progress(len(rows), len(entry["versions"]), "", "done")
    return rows


def _write_endmembers(path, version, E, names):
    exc, emi = _channel_table(version)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["channel", "excitation", "emission_nm"] + names)
        for c in range(E.shape[0]):
            w.writerow([c, exc[c] if c < len(exc) else "",
                        emi[c] if c < len(emi) else ""] + [f"{x:.6g}" for x in E[c]])


CSV_HEAD = ["leg", "version", "label", "tree", "kind", "key", "channels",
            "method", "extra", "duplicate_of", "accuracy", "mean_iou",
            "accuracy_outside_rois", "mean_iou_outside_rois",
            "residual_rel", "residual_rmse", "seconds", "note"]


def _write_metrics(directory, rows, detail, sel, entry):
    directory.mkdir(parents=True, exist_ok=True)
    extra_cols = sorted({k for r in rows for k in r if k not in CSV_HEAD})
    with (directory / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, CSV_HEAD + extra_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (directory / "metrics.json").write_text(json.dumps({
        "leg": entry["leg"], "Y": entry["Y"], "X": entry["X"],
        "run": _utc(), "selection": sel, "versions": detail}, indent=1))


def run_all(legs=None, out_root=DERIVED_ROOT, save_npy=False, progress=None,
            cancelled=None, only=None, root=PROCESSED_ROOT, trees=None):
    """Every leg that has a saved selection. Returns all rows, flat."""
    legs = legs or discover(root, out_root, trees)
    rows = []
    for leg in sorted(legs):
        if only and leg not in only:
            continue
        if load_selection(leg, out_root) is None:
            continue
        if cancelled and cancelled():
            break
        rows += run_leg(leg, legs, out_root, save_npy, progress, cancelled, root)
    collect_all(out_root)
    return rows


def collect_all(out_root=DERIVED_ROOT):
    """Gather every leg's metrics.csv into one flat table and rewrite it.

    Read back off disk rather than kept in memory, so running a single leg
    still leaves `metrics_all.csv` describing all of them.
    """
    out_root = Path(out_root)
    rows = []
    for p in sorted(out_root.glob("*/metrics.csv")):
        with p.open() as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        return rows
    head = CSV_HEAD + sorted({k for r in rows for k in r if k not in CSV_HEAD})
    out = out_root / "metrics_all.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, head, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


# =============================================================================
# The report — every leg, every version, one page
# =============================================================================
REPORT_CSS = """
:root { color-scheme: light dark;
        --bg:#ffffff; --fg:#16181d; --dim:#6b7280; --line:#e3e6ec; --panel:#f7f8fa;
        --good:#0a7d34; --bad:#b3261e; }
/* Dark under both scopes, matching the chart tokens below: the media query
   covers the OS setting, the data-theme scope covers an explicit choice, and
   the :not() guard lets a light stamp beat OS-dark. */
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
        --bg:#14161a; --fg:#e8eaee; --dim:#9aa1ad; --line:#2a2e36; --panel:#1b1e24;
        --good:#4ade80; --bad:#f87171; } }
:root[data-theme="dark"] {
        --bg:#14161a; --fg:#e8eaee; --dim:#9aa1ad; --line:#2a2e36; --panel:#1b1e24;
        --good:#4ade80; --bad:#f87171; }
* { box-sizing:border-box }
body { margin:0; padding:28px 32px 64px; background:var(--bg); color:var(--fg);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif }
h1 { font-size:22px; margin:0 0 4px } h2 { font-size:17px; margin:34px 0 10px }
h3 { font-size:14px; margin:20px 0 8px; color:var(--dim); font-weight:600 }
.sub { color:var(--dim); margin:0 0 20px }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px }
.card { border:1px solid var(--line); border-radius:8px; overflow:hidden; background:var(--panel) }
.card img { width:100%; display:block; image-rendering:pixelated; background:#000 }
.card .cap { padding:6px 8px; font-size:11px; line-height:1.35 }
.card .cap b { font-weight:600 } .card .cap span { color:var(--dim) }
.card.truth { outline:2px solid var(--fg) }
table { border-collapse:collapse; width:100%; font-size:12.5px; margin-top:8px }
th,td { text-align:right; padding:5px 9px; border-bottom:1px solid var(--line);
        font-variant-numeric:tabular-nums }
th:first-child, td:first-child { text-align:left }
th { color:var(--dim); font-weight:600; position:sticky; top:0; background:var(--bg) }
tr.dup td { color:var(--dim) }
table.sortable th { cursor:pointer; user-select:none; white-space:nowrap }
table.sortable th:hover { color:var(--fg) }
table.sortable th::after { content:"\\2003"; opacity:0 }
table.sortable th[data-dir]::after { content:" \\25BE"; opacity:.75 }
table.sortable th[data-dir="asc"]::after { content:" \\25B4"; opacity:.75 }
.bar { display:inline-block; height:8px; border-radius:2px; background:currentColor;
       vertical-align:middle; opacity:.55 }
.note { color:var(--dim); font-size:12px; margin:6px 0 0 }
.warn { color:var(--bad); font-size:12px; margin:6px 0 0 }

/* The ranking chart. Thin marks, a solid hairline grid, and a value column in
   text ink rather than the series colour, so identity is never colour alone. */
.chart { margin:0 0 22px }
.vtitle { font-size:13px; font-weight:600; margin-bottom:4px; color:var(--fg) }
.viz text { font:11px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif }
.vlegend { fill:var(--c-second) }
.vgrid { stroke:var(--c-grid); stroke-width:1 }
.vtick { fill:var(--c-muted); text-anchor:middle; font-variant-numeric:tabular-nums }
.vhead { fill:var(--c-muted); text-anchor:end }
.vlabel { fill:var(--c-second); text-anchor:end }
.vband { fill:var(--c-grid); opacity:0 }
.vrow:hover .vband { opacity:.45 }
.vrow:hover .vlabel { fill:var(--c-ink) }
.vspread { stroke-width:2; opacity:.3; stroke-linecap:round }
.vdot { fill-opacity:.5; stroke:var(--c-surface); stroke-width:2 }
.vmean { stroke:var(--c-surface); stroke-width:2 }
.vvalue { fill:var(--c-ink); font-variant-numeric:tabular-nums; font-weight:600; text-anchor:end }
.vsd { fill:var(--c-muted); font-variant-numeric:tabular-nums; text-anchor:end }

/* ---- the per-leg line chart ---- */
.linechart .vtitle { display:flex; align-items:center; gap:14px; flex-wrap:wrap }
.vpick { font-weight:400; font-size:12px; color:var(--dim) }
.vpick select { font:inherit; color:var(--fg); background:var(--panel);
                border:1px solid var(--line); border-radius:6px; padding:3px 7px }
.vlegendrow { display:flex; gap:14px; flex-wrap:wrap; margin:2px 0 6px }
.lgd { display:inline-flex; align-items:center; gap:5px; font-size:12px;
       color:var(--fg); cursor:default }
.lgd i { width:9px; height:9px; border-radius:50%; display:inline-block }
.lgd small { color:var(--dim); font-size:11px }
#legchart { display:block; overflow:visible }
.vcut { stroke:var(--c-muted); stroke-width:1; opacity:.5 }
.vcap { fill:var(--c-muted); font-size:10px }
.vline { color:var(--s) }
.vpath { fill:none; stroke:var(--s); stroke-width:2; stroke-linejoin:round }
.vpt { fill:var(--s); stroke:var(--c-surface); stroke-width:2 }
.vend { fill:var(--c-second); font-size:10.5px }
/* Hovering a legend entry pushes the other five back rather than hiding them,
   so a crossing line can still be followed through the crowd. */
#legchart[data-focus] .vline { opacity:.13 }
"""


# --- the ranking chart ------------------------------------------------------
# Dots, not bars. The accuracies sit between about 0.65 and 0.90, so a bar from
# zero would show eleven near-identical bars, and a bar that does not start at
# zero lies about its own length. A dot per leg plus a mean marker keeps an
# honest non-zero axis and shows the spread that a single mean hides.
#
# Two hues, by what the cube is rather than by its rank: a concatenation of the
# seven excitations, or one excitation on its own — which is the comparison the
# whole exercise exists to make. Both pairs validate in light and dark.
CHART = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "second": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9",
              "concat": "#2a78d6", "single": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "second": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a",
             "concat": "#3987e5", "single": "#d95926"},
}


def _chart_tokens():
    """The palette as CSS custom properties, light plus both dark scopes."""
    def block(mode):
        return ("".join(f"  --c-{k}: {v};\n" for k, v in CHART[mode].items())
                + "".join(f"  --c-leg{i}: {v};\n"
                          for i, v in enumerate(LEG_COLOURS[mode])))
    return (f".viz {{\n{block('light')}}}\n"
            f"@media (prefers-color-scheme: dark) {{\n"
            f'  :root:where(:not([data-theme="light"])) .viz {{\n{block("dark")}  }}\n}}\n'
            f':root[data-theme="dark"] .viz {{\n{block("dark")}}}\n')


def _ranking_chart(rows, title, metric="accuracy_outside_rois"):
    """One inline SVG: every version ranked by mean accuracy, one dot per leg.

    Written straight into the page — no library, no request, and it stays
    readable in both themes because every colour is a token defined above.
    """
    by_version = {}
    for r in rows:
        if r.get("duplicate_of"):
            continue                                  # the same cube, reached twice
        try:
            v = float(r.get(metric) or "nan")
        except ValueError:
            continue
        if np.isnan(v):
            continue
        e = by_version.setdefault(r["version"], {"label": r["label"], "kind": r["kind"],
                                                 "ch": r["channels"], "pts": []})
        e["pts"].append((r["leg"], v))
    if not by_version:
        return ""

    items = sorted(by_version.values(), key=lambda e: -np.mean([p[1] for p in e["pts"]]))
    values = [p[1] for e in items for p in e["pts"]]
    lo = np.floor(min(values) * 20 - 0.5) / 20         # to the 0.05 below
    hi = np.ceil(max(values) * 20 + 0.5) / 20
    ticks = np.arange(lo, hi + 1e-9, 0.05)

    L, R = 214, 596                                    # the plot band, in user units
    W, ROW, TOP = 720, 27, 62
    H = TOP + ROW * len(items) + 34
    x = lambda v: L + (v - lo) / (hi - lo) * (R - L)

    out = [f'<figure class="viz chart"><figcaption class="vtitle">{title}</figcaption>',
           f'<svg viewBox="0 0 {W} {H}" role="img" width="100%" '
           f'aria-label="{title}: every cube version ranked by mean accuracy">']

    # legend — identity is never colour alone, so the labels sit beside swatches
    for i, (key, name) in enumerate([("concat", "7 excitations concatenated"),
                                     ("single", "one excitation alone")]):
        cx = 214 + i * 210
        out.append(f'<circle cx="{cx}" cy="20" r="6" fill="var(--c-{key})"/>'
                   f'<text x="{cx + 12}" y="24" class="vlegend">{name}</text>')

    for t in ticks:                                    # hairline grid, solid
        out.append(f'<line x1="{x(t):.1f}" y1="{TOP - 14}" x2="{x(t):.1f}" '
                   f'y2="{TOP + ROW * len(items) - 8}" class="vgrid"/>'
                   f'<text x="{x(t):.1f}" y="{TOP + ROW * len(items) + 12}" '
                   f'class="vtick">{t:.2f}</text>')

    for i, e in enumerate(items):
        y = TOP + i * ROW
        colour = "concat" if e["kind"] == "concat" else "single"
        pts = sorted(e["pts"], key=lambda p: p[1])
        mean, sd = float(np.mean([p[1] for p in pts])), float(np.std([p[1] for p in pts]))
        out.append(f'<g class="vrow">'
                   f'<rect x="0" y="{y - 13}" width="{W}" height="{ROW - 2}" class="vband"/>'
                   f'<text x="{L - 14}" y="{y + 4}" class="vlabel">{e["label"]}</text>')
        # the spread, then every leg, then the mean on top
        out.append(f'<line x1="{x(pts[0][1]):.1f}" y1="{y}" x2="{x(pts[-1][1]):.1f}" '
                   f'y2="{y}" class="vspread" stroke="var(--c-{colour})"/>')
        for leg, v in pts:
            out.append(f'<circle cx="{x(v):.1f}" cy="{y}" r="4.5" class="vdot" '
                       f'fill="var(--c-{colour})"><title>{leg}: {v:.4f}</title></circle>')
        out.append(f'<circle cx="{x(mean):.1f}" cy="{y}" r="6.5" class="vmean" '
                   f'fill="var(--c-{colour})">'
                   f'<title>mean of {len(pts)} legs: {mean:.4f} ± {sd:.4f}</title></circle>'
                   f'<text x="{R + 26}" y="{y + 4}" class="vvalue">{mean:.3f}</text>'
                   f'<text x="{R + 78}" y="{y + 4}" class="vsd">± {sd:.3f}</text>'
                   f'</g>')
    out.append(f'<text x="{R + 60}" y="{TOP - 22}" class="vhead">mean</text>'
               f'<text x="{R + 120}" y="{TOP - 22}" class="vhead">sd</text>')
    out.append("</svg></figure>")
    return "\n".join(out)


# --- the per-leg line chart -------------------------------------------------
# The question this one answers is the one the ranking chart averages away:
# does a given leg actually get better as you move from one excitation to a
# concatenation, or is the mean hiding six legs that disagree? So: the cube
# versions along x in the order they are built (310 nm through white, then the
# concatenations), one line per leg, and the metric is a control rather than a
# decision — accuracy ranks the muscle-dominated result, per-class IoU ranks
# the structure you care about.
#
# Six series is the soft cap for a categorical palette, so every line is also
# labelled at its end: identity never rests on colour alone, and three of the
# light-mode hues sit under 3:1 on the light surface, where the relief rule
# requires exactly that.
LEG_COLOURS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
}

# What the metric picker offers. `good` says which end of the scale is better,
# so the axis can say so without the reader having to remember.
CHART_METRICS = [
    ("accuracy_outside_rois", "accuracy (outside regions)", "high"),
    ("mean_iou_outside_rois", "mean IoU (outside regions)", "high"),
    ("accuracy", "accuracy (all pixels)", "high"),
    ("mean_iou", "mean IoU (all pixels)", "high"),
    ("residual_rel", "residual, as a fraction of signal", "low"),
    ("residual_rmse", "residual RMSE (cube units — not comparable)", "low"),
]

_EX_ORDER = {k: i for i, k in enumerate(EXCITATIONS)}
_NORM_ORDER = {n["key"]: i for i, n in enumerate(NORMS)}
_TREE_ORDER = {t["key"]: i for i, t in enumerate(TREES)}
_SHORT = {"raw": "raw", "per_chan": "z/chan", "per_exc": "z/exc", "whole": "z/cube"}


def _version_order(r):
    """Single excitations by wavelength first, then the cubes built from them."""
    return (_TREE_ORDER.get(r["tree"], 9),
            0 if r["kind"] == "ex" else 1,
            _EX_ORDER.get(r["key"], 9) if r["kind"] == "ex"
            else _NORM_ORDER.get(r["key"], 9))


def _chart_data(rows, class_names):
    """Everything the in-page chart needs, small enough to inline."""
    seen, versions = set(), []
    for r in sorted(rows, key=_version_order):
        if r.get("duplicate_of") or r["version"] in seen:
            continue
        seen.add(r["version"])
        short = (r["key"] if r["kind"] == "ex" else _SHORT.get(r["key"], r["key"]))
        if len(_TREE_ORDER) > 1 and len({x["tree"] for x in rows}) > 1:
            short = f"{r['tree']}·{short}"
        versions.append({"id": r["version"], "label": r["label"], "short": short,
                         "kind": r["kind"], "ch": r["channels"]})

    legs = sorted({r["leg"] for r in rows})
    values = {leg: {} for leg in legs}
    for r in rows:
        if r.get("duplicate_of"):
            continue
        cell = {}
        for k, _, _ in CHART_METRICS:
            try:
                v = float(r.get(k) or "nan")
            except ValueError:
                v = float("nan")
            if not np.isnan(v):
                cell[k] = v
        for n in class_names:
            try:
                v = float(r.get(f"iou_{n}") or "nan")
            except ValueError:
                v = float("nan")
            if not np.isnan(v):
                cell[f"iou_{n}"] = v
        values[r["leg"]][r["version"]] = cell

    metrics = ([{"k": k, "label": lab, "good": good} for k, lab, good in CHART_METRICS]
               + [{"k": f"iou_{n}", "label": f"IoU · {n} (outside regions)", "good": "high"}
                  for n in class_names])
    # Shorten the leg names for the line ends: the date and "leg" are the same
    # on every one of them and only the rat and the side tell them apart.
    def short_leg(leg):
        m = re.search(r"Rat[_ ]?(\d+).*?(Left|Right)", leg, re.I)
        return f"R{m.group(1)} {m.group(2)[0].upper()}" if m else leg

    return {"versions": versions, "legs": legs,
            "short": {leg: short_leg(leg) for leg in legs},
            "metrics": metrics, "values": values}


LINE_JS = """
(() => {
  const D = window.CHARTDATA;
  if (!D || !D.versions.length) return;
  const svg = document.getElementById('legchart');
  const pick = document.getElementById('legmetric');
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a, kids) => {
    const e = document.createElementNS(NS, n);
    for (const k in a) if (a[k] != null) e.setAttribute(k, a[k]);
    (kids || []).forEach(c => e.appendChild(c));
    return e;
  };
  const txt = (s, a) => { const t = el('text', a); t.textContent = s; return t; };

  pick.innerHTML = D.metrics.map(m => `<option value="${m.k}">${m.label}</option>`).join('');

  function draw() {
    const mk = pick.value || D.metrics[0].k;
    const meta = D.metrics.find(m => m.k === mk) || D.metrics[0];
    const W = 780, H = 400, L = 52, R = 96, T = 34, B = 74;
    const xs = D.versions.map((_, i) =>
      L + (D.versions.length === 1 ? 0 : i * (W - L - R) / (D.versions.length - 1)));

    const all = [];
    D.legs.forEach(leg => D.versions.forEach(v => {
      const y = D.values[leg]?.[v.id]?.[mk];
      if (y != null) all.push(y);
    }));
    if (!all.length) { svg.innerHTML = ''; return; }
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo || 1) * 0.05;
    lo -= pad; hi += pad;
    // Ticks land on round numbers, but the *domain* stays where the data is:
    // rounding the domain out to whole ticks as well throws away half the plot
    // when the values happen to sit just inside a tick.
    // Pick the round step that actually lands about five ticks inside the
    // domain. Taking the first candidate at or above range/5 rounds up every
    // time and can leave an axis with two labels on it.
    const raw = (hi - lo) / 5;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    let step = raw, bestScore = Infinity;
    for (const m of [mag / 10, mag, mag * 10]) {
      for (const f of [1, 2, 2.5, 5]) {
        const c = f * m;
        const n = Math.floor(hi / c) - Math.ceil(lo / c) + 1;
        const score = Math.abs(n - 5) + (n < 3 ? 100 : 0) + (n > 9 ? 100 : 0);
        if (score < bestScore) { bestScore = score; step = c; }
      }
    }
    const y = v => H - B - (v - lo) / (hi - lo) * (H - T - B);
    const dp = step < 0.02 ? 3 : step < 1 ? 2 : 0;
    const ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(t);

    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const g = [];

    ticks.forEach(t => {
      g.push(el('line', { x1: L, y1: y(t), x2: W - R, y2: y(t), class: 'vgrid' }));
      g.push(txt(t.toFixed(dp), { x: L - 8, y: y(t) + 3.5, class: 'vtick',
                                  'text-anchor': 'end' }));
    });

    // Where the single excitations stop and the concatenations begin — the
    // whole point of the x order, so it gets a rule and a caption.
    const cut = D.versions.findIndex(v => v.kind === 'concat');
    if (cut > 0) {
      const mid = (xs[cut - 1] + xs[cut]) / 2;
      g.push(el('line', { x1: mid, y1: T - 12, x2: mid, y2: H - B, class: 'vcut' }));
      g.push(txt('single excitation', { x: mid - 8, y: T - 16, class: 'vcap',
                                        'text-anchor': 'end' }));
      g.push(txt('concatenated', { x: mid + 8, y: T - 16, class: 'vcap' }));
    }

    D.versions.forEach((v, i) => {
      g.push(txt(v.short, { x: xs[i], y: H - B + 16, class: 'vtick',
                            transform: `rotate(-38 ${xs[i]} ${H - B + 16})`,
                            'text-anchor': 'end' }));
    });

    const ends = [];
    D.legs.forEach((leg, s) => {
      const pts = [];
      D.versions.forEach((v, i) => {
        const val = D.values[leg]?.[v.id]?.[mk];
        if (val != null) pts.push([xs[i], y(val), val, v]);
      });
      if (!pts.length) return;
      const line = el('g', { class: 'vline', 'data-leg': leg,
                             style: `--s: var(--c-leg${s})` });
      line.appendChild(el('polyline', {
        points: pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '),
        class: 'vpath' }));
      pts.forEach(p => {
        const dot = el('circle', { cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: 4,
                                   class: 'vpt' });
        dot.appendChild(txt(`${leg} · ${p[3].label} · ${p[2].toFixed(4)}`, {}));
        line.appendChild(dot);
      });
      const last = pts[pts.length - 1];
      const label = txt(D.short[leg], { x: last[0] + 10, y: last[1] + 3.5,
                                        class: 'vend' });
      ends.push({ label, y: last[1] + 3.5 });
      line.appendChild(label);
      g.push(line);
    });

    // Legs that finish within a few units of each other would print their names
    // on top of one another, so the labels are spread apart just enough to read
    // — the markers stay exactly where the data puts them.
    const GAP = 12;
    ends.sort((a, b) => a.y - b.y);
    ends.forEach((e, i) => {
      if (i) e.y = Math.max(e.y, ends[i - 1].y + GAP);
    });
    const overflow = ends.length ? ends[ends.length - 1].y - (H - B) : 0;
    if (overflow > 0) ends.forEach(e => e.y -= overflow);
    ends.forEach(e => e.label.setAttribute('y', e.y.toFixed(1)));

    svg.replaceChildren(...g);
  }

  const legend = document.getElementById('leglegend');
  legend.innerHTML = D.legs.map((leg, s) =>
    `<span class="lgd" data-leg="${leg}"><i style="background:var(--c-leg${s})"></i>` +
    `${D.short[leg]} <small>${leg}</small></span>`).join('');
  legend.querySelectorAll('.lgd').forEach(item => {
    item.addEventListener('mouseenter', () => {
      svg.setAttribute('data-focus', item.dataset.leg);
      svg.querySelectorAll('.vline').forEach(l =>
        l.style.opacity = l.dataset.leg === item.dataset.leg ? 1 : '');
    });
    item.addEventListener('mouseleave', () => {
      svg.removeAttribute('data-focus');
      svg.querySelectorAll('.vline').forEach(l => l.style.opacity = '');
    });
  });

  pick.addEventListener('change', draw);
  draw();
})();
"""


def _line_chart_markup():
    """The shell the script fills: a metric picker, a legend and an empty svg."""
    return ('<figure class="viz chart linechart">'
            '<figcaption class="vtitle">Every leg across every cube version'
            '<label class="vpick">metric '
            '<select id="legmetric"></select></label></figcaption>'
            '<div class="vlegendrow" id="leglegend"></div>'
            '<svg id="legchart" width="100%" role="img" '
            'aria-label="one line per leg across the cube versions"></svg>'
            '<p class="note">Hover a leg in the legend to pick its line out; '
            'hover a point for its exact value. Lines that rise from left to '
            'right are legs that gain from concatenation.</p></figure>')


def _fmt(x, nd=4):
    try:
        return "—" if x is None or x == "" or np.isnan(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def read_all_rows(out_root=DERIVED_ROOT):
    """Every leg's metrics as flat rows, read only — the cross-leg table.

    `collect_all` rebuilds this file after every run, so reading it is enough
    and nothing here writes.
    """
    p = Path(out_root) / "metrics_all.csv"
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


_rows_from_disk = read_all_rows            # the name write_report already uses


def _table(rows, cols, sort_key, grid=None):
    """One metrics table, best first, and re-sortable on any column in the page.

    Every cell carries its numeric value in `data-v`, so the sorter in the page
    never has to parse the formatted text — which is what makes a blank, a
    "—" and a 0.0 sort correctly rather than alphabetically. `grid` names the
    thumbnail strip this table belongs to, so re-sorting the numbers re-orders
    the pictures with them.
    """
    def val(r, k):
        try:
            return float(r.get(k) or "nan")
        except ValueError:
            return float("nan")

    rows = sorted(rows, key=lambda r: (-val(r, sort_key) if not np.isnan(val(r, sort_key))
                                       else 1e9, r.get("version", "")))
    head = "".join(
        f'<th data-k="{k}"{" data-text=1" if k == "label" else ""}'
        f'{" data-dir=desc" if k == sort_key else ""}>{h}</th>' for k, h in cols)
    open_tag = f'<table class="sortable"{f' data-grid="{grid}"' if grid else ""}>'
    out = [open_tag + f"<thead><tr>{head}</tr></thead><tbody>"]
    for r in rows:
        cls = ' class="dup"' if r.get("duplicate_of") else ""
        cells = []
        for k, _ in cols:
            v = val(r, k)
            d = "" if np.isnan(v) else f' data-v="{v}"'
            if k == "label":
                dup = (f' <span style="opacity:.6">= {r["duplicate_of"]}</span>'
                       if r.get("duplicate_of") else "")
                cells.append(f'<td data-v="{r.get("label", "")}">{r.get("label", "")}{dup}</td>')
            elif k == sort_key:
                w = 0 if np.isnan(v) else max(1, round(v * 46))
                cells.append(f'<td{d}>{_fmt(v)} <i class="bar" style="width:{w}px"></i></td>')
            elif k in ("channels", "seconds", "legs"):
                cells.append(f'<td{d}>{r.get(k, "")}</td>')
            else:
                cells.append(f"<td{d}>{_fmt(v)}</td>")
        out.append(f'<tr{cls} data-version="{r.get("version", "")}">'
                   + "".join(cells) + "</tr>")
    return "\n".join(out) + "</tbody></table>"


# Sorting lives in the page rather than in the build, because the question
# changes with what you are looking for: accuracy ranks the muscle-dominated
# result, per-class IoU ranks the structure you actually care about, and the
# residual ranks how well the model fitted at all. The pictures follow the
# numbers so the two never disagree.
SORT_JS = """
document.querySelectorAll('table.sortable').forEach(table => {
  const body = table.tBodies[0];
  const grid = table.dataset.grid
    ? document.querySelector(`.grid[data-grid="${table.dataset.grid}"]`) : null;

  table.tHead.querySelectorAll('th').forEach((th, col) => {
    th.addEventListener('click', () => {
      // First click on a column: numbers high-to-low, names A-to-Z. Clicking
      // the column that is already sorted reverses it.
      const text = th.hasAttribute('data-text');
      const dir = th.dataset.dir === 'desc' ? 'asc'
                : th.dataset.dir === 'asc' ? 'desc'
                : (text ? 'asc' : 'desc');
      table.tHead.querySelectorAll('th').forEach(o => delete o.dataset.dir);
      th.dataset.dir = dir;

      const key = tr => {
        const v = tr.cells[col].dataset.v;
        if (text) return v ?? '';
        const n = parseFloat(v);
        return Number.isNaN(n) ? -Infinity : n;      // blanks sink, either way
      };
      const rows = [...body.rows].sort((a, b) => {
        const x = key(a), y = key(b);
        const c = text ? String(x).localeCompare(String(y)) : x - y;
        return dir === 'asc' ? c : -c;
      });
      rows.forEach(r => body.appendChild(r));

      if (!grid) return;
      const cards = new Map([...grid.querySelectorAll('[data-version]')]
        .map(el => [el.dataset.version, el]));
      rows.forEach(r => {
        const card = cards.get(r.dataset.version);
        if (card) grid.appendChild(card);            // the two refs stay in front
      });
    });
  });
});
"""


def write_report(out_root=DERIVED_ROOT, rows=None, title="Linear unmixing — all legs"):
    """One self-navigating HTML page: the ranking per leg, and across legs.

    Images are referenced relatively, so the page only works where it is
    written — next to the per-leg folders it points at.
    """
    out_root = Path(out_root)
    rows = rows if rows is not None else _rows_from_disk(out_root)
    if not rows:
        raise ValueError("no metrics to report — run a leg first")

    legs = sorted({r["leg"] for r in rows})
    # In the mask's own order (vessel, nerve, muscle), not alphabetical, so the
    # per-structure columns read the same everywhere in the tool.
    found = {k[4:] for r in rows for k in r if k.startswith("iou_")}
    try:
        canon = [c["name"] for c in classes_json()["classes"]]
    except Exception:
        canon = []
    class_names = ([n for n in canon if n in found]
                   + sorted(n for n in found if n not in canon))
    cols = ([("label", "version"), ("channels", "ch"),
             ("accuracy_outside_rois", "accuracy"), ("mean_iou_outside_rois", "mean IoU")]
            + [(f"iou_{n}", f"IoU {n}") for n in class_names]
            + [("accuracy", "acc (all px)"), ("residual_rel", "resid frac"),
               ("seconds", "s")])

    # Across legs: a version is only as good as its average rank of accuracy.
    per_version = {}
    for r in rows:
        per_version.setdefault(r["version"], []).append(r)
    def column(rs, key):
        """The mean of one column over the legs a version was run on."""
        vals = []
        for r in rs:
            try:
                v = float(r.get(key) or "nan")
            except ValueError:
                continue
            if not np.isnan(v):
                vals.append(v)
        return vals

    summary = []
    for vid, rs in per_version.items():
        accs = column(rs, "accuracy_outside_rois")
        row = {"version": vid, "label": rs[0]["label"],
               "channels": rs[0]["channels"],
               "duplicate_of": rs[0].get("duplicate_of", ""),
               "legs": len(rs),
               "accuracy_outside_rois": float(np.mean(accs)) if accs else "",
               "accuracy_sd": float(np.std(accs)) if accs else ""}
        # The averaged half of the report now carries the same per-structure
        # detail the per-leg tables do: a version that wins on accuracy can be
        # losing on the two small classes, and the mean hid that.
        for key in ["mean_iou_outside_rois", "residual_rel"] + \
                   [f"iou_{n}" for n in class_names]:
            vals = column(rs, key)
            row[key] = float(np.mean(vals)) if vals else ""
        summary.append(row)

    # One more row: how each structure does on average over every version, which
    # is the "is nerve simply hard?" question rather than "which cube is best?".
    per_class_note = []
    for n in class_names:
        vals = column(rows, f"iou_{n}")
        best = max(((np.mean(column(rs, f"iou_{n}")) if column(rs, f"iou_{n}") else -1,
                     rs[0]["label"]) for rs in per_version.values()), default=(0, ""))
        if vals:
            per_class_note.append(
                f"<b>{n}</b> mean IoU {np.mean(vals):.3f} "
                f"(best {best[0]:.3f}, {best[1]})")

    html = [f"<!doctype html><meta charset=utf-8><title>{title}</title>",
            f"<style>{REPORT_CSS}\n{_chart_tokens()}</style>", f"<h1>{title}</h1>",
            f'<p class="sub">{len(legs)} legs · {len(per_version)} versions each · '
            f'written {_utc()} by Code/Linear_unmixing_tool</p>',
            "<h2>Across all legs</h2>",
            _line_chart_markup(),
            _ranking_chart(rows, "Accuracy by cube version, one dot per leg"),
            '<p class="note">Accuracy is the mean over legs, scored outside the '
            'endmember regions — the pixels whose spectra defined the model are '
            'left out, because they flatter it.'
            + (' Rows in grey are byte-identical to the version named beside '
               'them: the same cube reached through two trees.'
               if any(r.get("duplicate_of") for r in rows) else '')
            + '</p>',
            '<p class="note">Click a column heading to sort by it; click again to '
            'reverse. Every table on this page sorts, and a leg\'s thumbnails '
            're-order with its table.</p>',
            _table(summary,
                   [("label", "version"), ("channels", "ch"), ("legs", "legs"),
                    ("accuracy_outside_rois", "mean accuracy"),
                    ("accuracy_sd", "sd"),
                    ("mean_iou_outside_rois", "mean IoU")]
                   + [(f"iou_{n}", f"IoU {n}") for n in class_names]
                   + [("residual_rel", "resid frac")],
                   "accuracy_outside_rois"),
            '<p class="note">Per-structure columns are that structure\'s IoU '
            'averaged over the legs. Averaged over every version too: '
            + " · ".join(per_class_note) + '.</p>']

    for leg in legs:
        lrows = [r for r in rows if r["leg"] == leg]
        html.append(f"<h2>{leg}</h2>")
        cards = [f'<figure class="card truth" style="margin:0">'
                 f'<img src="{leg}/truth.png" alt="truth"><figcaption class="cap">'
                 f'<b>hand-painted truth</b><br><span>what every version is scored '
                 f'against</span></figcaption></figure>',
                 f'<figure class="card truth" style="margin:0">'
                 f'<img src="{leg}/rois_preview.png" alt="regions"><figcaption class="cap">'
                 f'<b>endmember regions</b><br><span>the pixels the spectra came from'
                 f'</span></figcaption></figure>']
        ordered = sorted(lrows, key=lambda r: -float(r.get("accuracy_outside_rois") or 0))
        for r in ordered:
            acc = _fmt(r.get("accuracy_outside_rois"), 3)
            dup = " · duplicate" if r.get("duplicate_of") else ""
            cards.append(
                f'<figure class="card" style="margin:0" data-version="{r["version"]}">'
                f'<img src="{leg}/{r["version"]}/classmap.png" alt="{r["version"]}">'
                f'<figcaption class="cap"><b>{acc}</b> {r["label"]}<br>'
                f'<span>{r["channels"]} ch{dup}</span></figcaption></figure>')
        html.append(f'<div class="grid" data-grid="{leg}">' + "".join(cards) + "</div>")
        html.append(_table(lrows, cols, "accuracy_outside_rois", grid=leg))

    html.append('<p class="note">Click any column heading to sort by it — the '
                'thumbnails above each table follow the numbers. Click it again '
                'to reverse.</p>')
    html.append("<script>window.CHARTDATA = "
                + json.dumps(_chart_data(rows, class_names)) + ";</script>")
    html.append(f"<script>{LINE_JS}</script>")
    html.append(f"<script>{SORT_JS}</script>")

    path = out_root / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(html))
    return path
