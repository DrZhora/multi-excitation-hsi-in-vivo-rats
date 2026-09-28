"""The .im3 concatenation machinery, lifted verbatim out of Code/concat.ipynb.

`concat.ipynb` built the cubes under `Data/Processed/ALL_channels_kept` and
`Overexposed_channels_removed` straight from `Data/Raw`, where a leg is a
`Distinct_Cubes` folder three levels down. `concat_aligned.ipynb` builds the
same four ways from the *aligned, cropped* legs, where a leg is a single folder
whose name already is the leg, and additionally carries the label masks across.
Rather than copy six hundred lines into a second notebook and let the two
drift, the shared half lives here and each notebook supplies only its own
leg-finding and its own drivers.

Nothing below is new work: the record model, the writer, the trim and the three
z-norm ways are exactly what concat.ipynb ran, with three edits:

  * the hardcoded RAW / OUT / ALL_KEPT paths are gone - every function takes
    the folders it works on,
  * `find_legs` (the RAW layout) is joined by `find_legs_flat` (the aligned
    layout), and both return the same leg dict,
  * `build_way1` takes the legs to build rather than discovering them, and
    `BUILDER` names whichever notebook is driving, for the provenance stamped
    into each cube's UserComments.
"""

# =============================================================================
# Concatenate the 7 excitation cubes of each leg into one 217-channel .im3.
#
# Each leg was imaged at 310 / 325 / 340 / 365 / 385 / 400 nm excitation plus a
# white-light cube, and every cube is 31 emission bands (420-720 nm in 10 nm
# steps). Stacking them in that order gives 7 x 31 = 217 channels: channel
# 31*k + i is band i of the k-th cube, and nothing is resampled, rescaled or
# normalised on the way — the pixel values are the raw sensor counts.
#
# Which layer came from where is recorded three ways, so it never has to be
# guessed: a .csv and a .json sidecar next to the cube, a compact block map
# embedded in the file's own UserComments record, and the per-band Spectra
# records, which are carried over from the source cubes untouched so each
# channel keeps its own emission wavelength and exposure.
#
# Step 1: the .im3 record model. The format is a tree of records:
#   int32 nameLen, name, int32 recLen (= payload + 8), int32 type, payload
# Containers (type 0) open with int32 1, int32 nChildren, then their children,
# and may end with a few bytes of padding that must be preserved. Bio-Formats
# reads .im3 but ships no writer, so both directions live here.
# =============================================================================
import csv
import json
import mmap
import re
import struct
from pathlib import Path

import numpy as np

PROCESSED_ROOT = Path("/Users/zmkoyan/User Files/multi-excitation-hsi-in-vivo-rats/Data/Processed")

# Stamped into every cube's UserComments as `concatenated_by`; each notebook
# sets it to its own name so a cube on disk says which driver made it.
BUILDER = "Code/concat_lib.py"

COOKIE = 1985
T_CONTAINER, T_INT, T_FLOAT, T_DATA, T_STRING = 0, 6, 7, 3, 10


def _read_record(buf, p):
    """(name, type, payload_start, payload_len) of the record at p, and the next p."""
    n, = struct.unpack_from("<i", buf, p)
    name = bytes(buf[p + 4:p + 4 + n]).decode("utf-8", "replace")
    p += 4 + n
    reclen, rtype = struct.unpack_from("<ii", buf, p)
    start, size = p + 8, reclen - 8
    return (name, rtype, start, size), start + size


def children(buf, rec):
    """Child records of a container, in order."""
    p, end, out = rec[2] + 8, rec[2] + rec[3], []
    while p < end - 8:                      # a tail shorter than a header is padding
        child, p = _read_record(buf, p)
        out.append(child)
    return out


def child_pad(buf, rec):
    """The bytes left over after a container's last child."""
    p, end = rec[2] + 8, rec[2] + rec[3]
    while p < end - 8:
        _, p = _read_record(buf, p)
    return bytes(buf[p:end])


def find(buf, rec, name):
    for child in children(buf, rec):
        if child[0] == name:
            return child
    raise KeyError(f"no {name!r} record inside {rec[0]!r}")


def numbers(buf, rec):
    """Values of an int (type 6) or float (type 7) record.

    Both use the same shape: int32 flag, then either (count, values) when the
    flag is 1 or a single value when it is 0, then a trailing int32 0.
    """
    fmt = "i" if rec[1] == T_INT else "f"
    flag, = struct.unpack_from("<i", buf, rec[2])
    if flag == 1:
        count, = struct.unpack_from("<i", buf, rec[2] + 4)
        return list(struct.unpack_from(f"<{count}{fmt}", buf, rec[2] + 8))
    return list(struct.unpack_from(f"<1{fmt}", buf, rec[2] + 4))


def raw(buf, rec):
    """A record's payload bytes, exactly as stored."""
    return bytes(buf[rec[2]:rec[2] + rec[3]])


# --- writing side: payload builders and one record serialiser ----------------
def enc_record(name, rtype, payload):
    nm = name.encode("utf-8")
    return (struct.pack("<i", len(nm)) + nm
            + struct.pack("<ii", len(payload) + 8, rtype) + payload)


def enc_container(child_blobs, pad=b""):
    return struct.pack("<ii", 1, len(child_blobs)) + b"".join(child_blobs) + pad


def enc_ints(values):
    return struct.pack("<ii", 1, len(values)) + struct.pack(f"<{len(values)}i", *values) + b"\0" * 4


def enc_string(text):
    b = text.encode("utf-8")
    return struct.pack("<ii", 0, len(b)) + b + b"\0" * 4


# =============================================================================
# Step 2: find the legs, and read what each cube says about itself.
#
# A leg is any Distinct_Cubes folder; its cubes are ordered by excitation with
# the white-light cube last, which is also the channel order of the result.
# Filenames carry the excitation and the exposure ("385_1500exp" = 385 nm,
# 1500 ms), and the file itself carries the 31 emission wavelengths and their
# exposures inside each band's Spectra record. Both are kept: the filename is
# what the operator wrote down, the records are what the instrument recorded.
# =============================================================================
def band_order_key(path):
    """Excitation ascending, white light last (one leg spells it 'whte')."""
    m = re.match(r"\s*(\d{3})", path.stem)
    return (0, int(m.group(1)), path.stem.lower()) if m else (1, 0, path.stem.lower())


def find_legs(root):
    """Every Distinct_Cubes folder under root, with its cubes in channel order."""
    legs = []
    for folder in sorted(root.glob("*/*/*/Distinct_Cubes")):
        cubes = sorted(folder.glob("*.im3"), key=band_order_key)
        if cubes:
            legs.append({"dir": folder,
                         "date": folder.parts[-4],
                         "rat": folder.parts[-3],
                         "leg": folder.parts[-2],
                         "cubes": cubes})
    return legs


def leg_name(leg):
    if "name" in leg:                      # the aligned export already named it
        return leg["name"]
    return f"{leg['date']}_{leg['rat']}_{leg['leg']}".replace(" ", "_")


def find_legs_flat(root):
    """Every leg folder directly under root, with its cubes in channel order.

    The aligned, cropped export writes one folder per leg whose name is already
    the leg ("12.09_Rat_1_Left_leg"), so there is nothing to reassemble - but
    the rest of the module reads leg["date"]/["rat"]/["leg"], so split the name
    back apart and keep `name` as the authority for the output filename.
    """
    legs = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        cubes = sorted(folder.glob("*.im3"), key=band_order_key)
        if not cubes:
            continue
        m = re.match(r"([\d.]+)_(Rat_\d+)_(.+)", folder.name)
        legs.append({"dir": folder, "name": folder.name,
                     "date": m.group(1) if m else "", "rat": m.group(2) if m else "",
                     "leg": m.group(3) if m else folder.name, "cubes": cubes})
    return legs


def read_cube_info(path):
    """Shape, pixel offset and per-band wavelength/exposure of one .im3."""
    path = Path(path)
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if struct.unpack_from("<i", mm, 0)[0] != COOKIE:
            raise ValueError(f"{path} is not an .im3 (bad cookie)")
        p, top = 4, []
        while p < len(mm) - 16:
            rec, p = _read_record(mm, p)
            top.append(rec)
        inner = children(mm, find(mm, top[-1], "DataSet"))[0]
        data = find(mm, inner, "Data")
        _, X, Y, C = struct.unpack_from("<iiii", mm, data[2])

        bands = children(mm, find(mm, inner, "Spectra"))
        emission = [numbers(mm, find(mm, children(mm, find(mm, b, "Spectrum"))[0],
                                     "Wavelengths"))[0] for b in bands]
        # Every band repeats the whole acquisition plan; band 0's copy is enough.
        plan = children(mm, find(mm, children(mm, find(mm, bands[0], "AcquisitionSettings"))[0],
                                 "WavelengthsAndExposures"))
        exposures_ms = [numbers(mm, find(mm, e, "ExposureTime"))[0] / 1000.0 for e in plan]

    m = re.match(r"\s*(\d{3})", path.stem)
    n = re.search(r"(\d+)exp", path.stem)
    return {"path": path, "name": path.name,
            "excitation": m.group(1) + " nm" if m else "white",
            "band": m.group(1) if m else "white",
            "exposure_ms_filename": int(n.group(1)) if n else None,
            "X": X, "Y": Y, "C": C, "data_offset": data[2] + 16,
            # The half-open range of the cube's channels to keep. Everything
            # downstream reads start/stop, never 0..C, so dropping channels is
            # a question of setting these and nothing else.
            "start": 0, "stop": C,
            "emission_nm": emission, "exposure_ms": exposures_ms}


def keep(info):
    """How many channels of a cube survive its kept range."""
    return info["stop"] - info["start"]


def apply_trim(infos, trim):
    """trim = {"385": 2, "400": 5} drops that many LEADING channels of a cube."""
    for info in infos:
        info["start"] = int(trim.get(info["band"], 0))
        info["stop"] = info["C"]
    return infos


def build_manifest(leg, trim=None):
    """One row per output channel, plus the checks that the cubes are stackable."""
    infos = [read_cube_info(p) for p in leg["cubes"]]
    if trim:
        apply_trim(infos, trim)
    shapes = {(i["X"], i["Y"]) for i in infos}
    if len(shapes) != 1:
        raise ValueError(f"{leg_name(leg)}: cubes disagree on frame size: {shapes}")
    X, Y = shapes.pop()

    rows, band = [], 0
    for block, info in enumerate(infos):
        for c in range(info["start"], info["stop"]):
            rows.append({
                "channel": band,
                "block": block,
                "excitation": info["excitation"],
                "emission_nm": round(info["emission_nm"][c], 1),
                "exposure_ms": round(info["exposure_ms"][c], 3),
                "exposure_ms_filename": info["exposure_ms_filename"],
                "source_file": info["name"],
                "source_channel": c,
            })
            band += 1
    return {"leg": leg_name(leg), "date": leg["date"], "rat": leg["rat"], "side": leg["leg"],
            "X": X, "Y": Y, "C": band, "infos": infos, "channels": rows}


# =============================================================================
# Step 3: write the concatenated cube.
#
# The output is the first cube of the leg used as a template — VersionInfo,
# DisplayPipeline, Thumbnail, Protocol and the rest are copied byte for byte —
# with four records rebuilt:
#   Shape           -> (X, Y, 217)
#   Spectra         -> the 31 band records of all 7 cubes, concatenated, so
#                      every channel keeps its own emission wavelength
#   UserComments    -> a compact JSON block map (which channels came from which
#                      file), so the file alone answers the question
#   SaturatedPixels -> emptied: the template's mask describes only its own 31
#                      channels, and a stale mask is worse than none
# Protocol still describes the template's excitation and exposure — the per-band
# records and the manifest are the authority for the other six blocks.
#
# The pixels are never all in memory at once: the file is laid out first with a
# hole where the Data payload goes, then each source cube is copied into its
# 31-channel slice through a memory map. Pixels are stored (Y, X, C), so one
# pixel's whole 217-value spectrum ends up contiguous on disk.
# =============================================================================
HOLE = "HOLE"                                 # placeholder for the pixel region
# Every record in this format ends with four zero bytes, the instrument's own
# Data record included. Leaving it off makes a file our reader and Bio-Formats
# both accept and Nuance rejects with "File position = 4294967295" (-1).
DATA_TAIL = b"\0" * 4


def _header(name, rtype, payload_len):
    nm = name.encode("utf-8")
    return struct.pack("<i", len(nm)) + nm + struct.pack("<ii", payload_len + 8, rtype)


def _part(name, rtype, body, body_len):
    """A record as (chunks, total length); body may contain the pixel HOLE."""
    head = _header(name, rtype, body_len)
    return [head] + body, len(head) + body_len


def _container(name, parts, pad=b""):
    body_len = 8 + sum(n for _, n in parts) + len(pad)
    body = [struct.pack("<ii", 1, len(parts))]
    for chunks, _ in parts:
        body.extend(chunks)
    if pad:
        body.append(pad)
    return _part(name, T_CONTAINER, body, body_len)


def _verbatim(buf, rec):
    """A record copied exactly as it stands in the source file."""
    blob = _header(rec[0], rec[1], rec[3]) + raw(buf, rec)
    return [blob], len(blob)


def _all_band_records(infos):
    """The Spectra band records of every cube of the leg, in channel order."""
    out = []
    for info in infos:
        with open(info["path"], "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            p, top = 4, []
            while p < len(mm) - 16:
                rec, p = _read_record(mm, p)
                top.append(rec)
            inner = children(mm, find(mm, top[-1], "DataSet"))[0]
            bands = children(mm, find(mm, inner, "Spectra"))
            # Slice the band records exactly like the pixels: a file carrying
            # 31 Spectra entries for 29 channels is quietly corrupt metadata.
            for band in bands[info["start"]:info["stop"]]:
                out.append(_verbatim(mm, band))
    return out


def block_map(man):
    """The compact 'which channels came from which cube' map stored in the file."""
    blocks, first = [], 0
    for block, info in enumerate(man["infos"]):
        blocks.append({"block": block, "channels": [first, first + keep(info) - 1],
                       "source_channels": [info["start"], info["stop"] - 1],
                       "dropped_leading": info["start"],
                       "file": info["name"], "excitation": info["excitation"],
                       "exposure_ms": info["exposure_ms_filename"]})
        first += keep(info)
    return {"concatenated_by": BUILDER, "leg": man["leg"],
            "order": "excitation ascending, white light last",
            "channel": "the n-th kept channel of block b, see blocks[]",
            "blocks": blocks,
            "saturated_pixels": "dropped (template's mask covered only its own cube)"}


def write_concat(man, out_path):
    """Build the 217-channel .im3 and return its path."""
    X, Y, C = man["X"], man["Y"], man["C"]
    infos = man["infos"]
    spectra_parts = _all_band_records(infos)
    comments = json.dumps(block_map(man), separators=(",", ":"))

    pixel_bytes = X * Y * C * 2
    data_head = (_header("Data", T_DATA, 16 + pixel_bytes + len(DATA_TAIL))
                 + struct.pack("<iiii", 3, X, Y, C))
    data_part = ([data_head, (HOLE, pixel_bytes), DATA_TAIL],
                 len(data_head) + pixel_bytes + len(DATA_TAIL))

    with open(infos[0]["path"], "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        p, top = 4, []
        while p < len(mm) - 16:
            rec, p = _read_record(mm, p)
            top.append(rec)
        root = top[-1]
        dataset = find(mm, root, "DataSet")
        inner = children(mm, dataset)[0]

        inner_parts = []
        for rec in children(mm, inner):
            if rec[0] == "Shape":
                inner_parts.append(_part("Shape", T_INT, [enc_ints([X, Y, C])],
                                         len(enc_ints([X, Y, C]))))
            elif rec[0] == "Spectra":
                inner_parts.append(_container("Spectra", spectra_parts,
                                              child_pad(mm, rec)))
            elif rec[0] == "UserComments":
                body = enc_string(comments)
                inner_parts.append(_part("UserComments", T_STRING, [body], len(body)))
            elif rec[0] == "SaturatedPixels":
                inner_parts.append(_container("SaturatedPixels", []))
            elif rec[0] == "Data":
                inner_parts.append(data_part)
            else:
                inner_parts.append(_verbatim(mm, rec))

        inner_part = _container("", inner_parts, child_pad(mm, inner))
        dataset_part = _container("DataSet", [inner_part], child_pad(mm, dataset))

        root_parts = [dataset_part if rec[0] == "DataSet" else _verbatim(mm, rec)
                      for rec in children(mm, root)]
        root_part = _container(root[0], root_parts, child_pad(mm, root))

        chunks = [struct.pack("<i", COOKIE)]
        for rec in top[:-1]:                       # FileVersion and friends
            chunks.extend(_verbatim(mm, rec)[0])
        chunks.extend(root_part[0])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".im3.part")
        data_offset = None
        with open(tmp, "wb") as out:
            for chunk in chunks:
                if isinstance(chunk, tuple) and chunk[0] is HOLE:
                    data_offset = out.tell()       # leave the pixel region sparse
                    out.seek(chunk[1] - 1, 1)
                    out.write(b"\0")
                else:
                    out.write(chunk)
            total = out.tell()

    # Pixels: copy each source cube straight into its own slice of the output.
    dst = np.memmap(tmp, dtype="<u2", mode="r+", offset=data_offset, shape=(Y, X, C))
    first = 0
    for info in infos:
        src = np.memmap(info["path"], dtype="<u2", mode="r", offset=info["data_offset"],
                        shape=(info["Y"], info["X"], info["C"]))
        dst[:, :, first:first + keep(info)] = src[:, :, info["start"]:info["stop"]]
        first += keep(info)
        del src
    dst.flush()
    del dst

    tmp.replace(out_path)
    return out_path, total


def write_sidecars(man, out_path):
    """The same channel table next to the cube, as .csv and .json."""
    fields = ["channel", "block", "excitation", "emission_nm", "exposure_ms",
              "exposure_ms_filename", "source_file", "source_channel"]
    csv_path = out_path.with_name(out_path.stem + "_channels.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(man["channels"])

    json_path = out_path.with_name(out_path.stem + "_channels.json")
    meta = {k: man[k] for k in ("leg", "date", "rat", "side", "X", "Y", "C")}
    meta["sources"] = [{"block": i, "file": info["name"],
                        "path": str(info["path"]),
                        "excitation": info["excitation"],
                        "exposure_ms": info["exposure_ms_filename"],
                        "channels": keep(info),
                        "source_channels": [info["start"], info["stop"] - 1],
                        "dropped_leading": info["start"]}
                       for i, info in enumerate(man["infos"])]
    meta["channels"] = man["channels"]
    json_path.write_text(json.dumps(meta, indent=1))
    return csv_path, json_path


# =============================================================================
# Step 4: build the concatenated cubes.
#
# `build_way1` is the whole of way 1: one cube per leg, written to the given
# folder with its _channels.csv / _channels.json beside it. `trim` names how
# many LEADING channels to drop from a cube, keyed by its excitation — leave it
# None here, where every channel is kept; Part 2 passes one in. Existing outputs
# are left alone unless overwrite is set.
# =============================================================================
def build_way1(out_dir, legs, trim=None, overwrite=False):
    print(f"{len(legs)} legs  ->  {out_dir}\n")

    for leg in legs:
        out_path = out_dir / f"{leg_name(leg)}_concat.im3"
        if out_path.exists() and not overwrite:
            print(f"{out_path.name:34s} exists, skipped")
            continue

        man = build_manifest(leg, trim)
        print(f"{man['leg']:28s} {len(man['infos'])} cubes -> "
              f"{man['C']} channels  {man['X']}x{man['Y']}", flush=True)
        first = 0
        for i, info in enumerate(man["infos"]):
            dropped = f"  (dropped {info['start']})" if info["start"] else ""
            print(f"    block {i}  ch {first:3d}-{first + keep(info) - 1:3d}  "
                  f"{info['excitation']:>7s}  {info['exposure_ms_filename']:>6} ms  "
                  f"{info['name']}{dropped}")
            first += keep(info)

        path, total = write_concat(man, out_path)
        csv_path, json_path = write_sidecars(man, out_path)
        print(f"    wrote {path.name}  ({total / 1e6:.0f} MB) + "
              f"{csv_path.name}, {json_path.name}\n", flush=True)



# =============================================================================
# Step 5: check the concatenated cubes.
#
# For each cube: re-read its records (shape, band count, the embedded block map),
# then compare a scattered sample of pixels against the source cubes and the
# exact source channels they came from. The comparison is exact — nothing was
# rescaled — so any difference at all is a bug.
# =============================================================================
SAMPLE = 2000                                  # pixels compared per source cube


def verify_way1(out_dir, seed=0):
    rng = np.random.default_rng(seed)
    for out_path in sorted(out_dir.glob("*_concat.im3")):
        info = read_cube_info(out_path)
        with open(out_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            p, top = 4, []
            while p < len(mm) - 16:
                rec, p = _read_record(mm, p)
                top.append(rec)
            inner = children(mm, find(mm, top[-1], "DataSet"))[0]
            shape = numbers(mm, find(mm, inner, "Shape"))
            comments = find(mm, inner, "UserComments")
            length, = struct.unpack_from("<i", mm, comments[2] + 4)
            embedded = json.loads(bytes(mm[comments[2] + 8:comments[2] + 8 + length]))

        print(f"{out_path.name}  Shape={tuple(shape)}  Data={info['C']} channels  "
              f"bands in Spectra={len(info['emission_nm'])}  "
              f"emission {min(info['emission_nm']):.0f}-{max(info['emission_nm']):.0f} nm")

        # Source paths come from the sidecar: several legs share a filename, so
        # the name in the embedded map is not enough to find the right cube.
        sidecar = json.loads(out_path.with_name(out_path.stem + "_channels.json").read_text())
        sources = {s["block"]: Path(s["path"]) for s in sidecar["sources"]}

        dst = np.memmap(out_path, dtype="<u2", mode="r", offset=info["data_offset"],
                        shape=(info["Y"], info["X"], info["C"]))
        ys = rng.integers(0, info["Y"], SAMPLE)
        xs = rng.integers(0, info["X"], SAMPLE)
        for block in embedded["blocks"]:
            src_path = sources[block["block"]]
            src_info = read_cube_info(src_path)
            src = np.memmap(src_path, dtype="<u2", mode="r", offset=src_info["data_offset"],
                            shape=(src_info["Y"], src_info["X"], src_info["C"]))
            lo, hi = block["channels"]
            slo, shi = block.get("source_channels", [0, src_info["C"] - 1])
            same = np.array_equal(dst[ys, xs, lo:hi + 1], src[ys, xs, slo:shi + 1])
            note = f" (src {slo}-{shi})" if slo else ""
            print(f"    block {block['block']}  ch {lo:3d}-{hi:3d}  {block['file']:20s} "
                  f"{'OK' if same else 'MISMATCH'}{note}")
            del src
        del dst



# =============================================================================
# Step 6: three normalised ways of concatenating the same 7 cubes.
#
# Way 1 (above) keeps the raw counts. Ways 2-4 stack the identical pixels but
# z-score them, and differ only in HOW WIDE the statistics are pooled:
#
#   way 2  per channel     217 scopes — every channel gets its own mean/std
#   way 3  per excitation  7 scopes — one mean/std per 31-channel cube, so the
#                          shape of each excitation's emission spectrum survives
#   way 4  whole cube      1 scope — the entire 217-channel stack, so all
#                          excitations stay on one common scale
#
# "Z-norm of 1-99% pixel values" here means: the mean and std come only from the
# pixels between the scope's 1st and 99th percentile, then z = (x - mean) / std
# is applied to EVERY pixel, outliers included. Trimming the statistics, not the
# data, keeps the 4040-count saturated pixels and any dead pixels from dragging
# the scale around while still mapping them through it. Note that ~2/3 of every
# frame is exactly 0 (the dark surround), so p1 = 0 and the background is inside
# the window and dominates the mean — that is the honest statistic of the cube
# as acquired, not a masking of the specimen.
#
# Storage: .im3 pixels are uint16, so z cannot be written directly. Each file
# gets ONE affine map shared by all 217 channels — u = (z - z_min) * 65535 /
# (z_max - z_min), where z_min/z_max are the true extremes of that file's z —
# so nothing is clipped, channels stay comparable, and the step is ~0.001 sigma.
# Both sidecars and the file's own UserComments carry z_min, z_max and the
# per-channel mean/std, so raw counts invert back exactly:
#     z = z_min + u * (z_max - z_min) / 65535
#     x = z * std[c] + mean[c]
# Inputs are the way-1 cubes in Raw_Concatenated, which already carry the band
# records and the block map; only the pixels and the comment are rebuilt.
# =============================================================================
ROWS = 128                                     # rows of the cube handled at once
U16_MAX = 65535

# Each way is a subfolder name and a label; the tree it lands in is passed in,
# so the same three functions serve both the 217- and the 210-channel builds.
WAYS = {
    "per_channel":    {"folder": "Per_Chanell_Norm_1_99_Concat",
                       "label": "way 2 - z-norm per channel"},
    "per_excitation": {"folder": "Per_Cube_Norm_1_99_Concat",
                       "label": "way 3 - z-norm per excitation cube"},
    "whole_cube":     {"folder": "Overall_4D_Norm_1_99_Concat",
                       "label": "way 4 - z-norm over the whole 4D cube"},
}


def open_concat(path):
    """A read-only (Y, X, C) memory map of a concatenated cube, plus its info."""
    info = read_cube_info(path)
    cube = np.memmap(path, dtype="<u2", mode="r", offset=info["data_offset"],
                     shape=(info["Y"], info["X"], info["C"]))
    return cube, info


def channel_histograms(cube):
    """Exact value histogram per channel — every statistic below comes from it.

    The values are small integers, so counting them is both cheaper and more
    accurate than sorting 1.4 M pixels per channel for a percentile.
    """
    C = cube.shape[2]
    top = 0
    for y in range(0, cube.shape[0], ROWS):
        top = max(top, int(cube[y:y + ROWS].max()))
    hist = np.zeros((C, top + 1), dtype=np.int64)
    for y in range(0, cube.shape[0], ROWS):
        chunk = np.asarray(cube[y:y + ROWS])
        for c in range(C):
            hist[c] += np.bincount(chunk[:, :, c].ravel(), minlength=top + 1)
    return hist


def _scope_stats(counts):
    """(mean, std, p1, p99, vmin, vmax, n) of one pooled histogram.

    The mean and std use only the bins between the 1st and 99th percentile.
    """
    values = np.arange(counts.size, dtype=np.float64)
    total = counts.sum()
    cum = np.cumsum(counts)
    p1 = int(np.searchsorted(cum, 0.01 * total, side="left"))
    p99 = int(np.searchsorted(cum, 0.99 * total, side="left"))
    nz = np.nonzero(counts)[0]
    window = counts[p1:p99 + 1]
    n = window.sum()
    v = values[p1:p99 + 1]
    mean = float((v * window).sum() / n)
    var = float((v * v * window).sum() / n) - mean ** 2
    return {"mean": mean, "std": float(np.sqrt(max(var, 0.0))),
            "p1": p1, "p99": p99, "min": int(nz[0]), "max": int(nz[-1]),
            "n_in_window": int(n), "n_total": int(total)}


def scope_ids(way, block_of):
    """The pooling scope each channel belongs to, for the given way."""
    if way == "per_channel":
        return list(range(len(block_of)))
    if way == "per_excitation":
        return list(block_of)
    return [0] * len(block_of)


def channel_stats(hist, way, block_of):
    """Per-channel mean/std for one way, pooled over that way's scopes."""
    C = hist.shape[0]
    scopes = scope_ids(way, block_of)
    groups = {}
    for c in range(C):
        groups.setdefault(scopes[c], []).append(c)

    per_scope = {key: _scope_stats(hist[chans].sum(axis=0)) for key, chans in groups.items()}
    stats = []
    for c in range(C):
        s = dict(per_scope[scopes[c]])
        s["scope"] = scopes[c]
        s["degenerate"] = s["std"] == 0.0
        s["std_used"] = s["std"]
        nz = np.nonzero(hist[c])[0]
        s["chan_min"], s["chan_max"] = int(nz[0]), int(nz[-1])
        stats.append(s)
    return _fix_degenerate(stats)


def _fix_degenerate(stats):
    """Give a channel with no spread in its window a scale that still inverts.

    In every leg the white cube's first band is >99% zeros, so its 1-99 window
    is a single value and its std is 0. Dividing by 1 instead would throw that
    channel's few hundred counts hundreds of sigma out and squeeze every other
    channel into a sliver of the uint16 range. Scaling it to just fill the
    window the real channels define keeps its own signal, keeps the others
    precise, and the std that did it is recorded like any other.
    """
    good = [s for s in stats if not s["degenerate"]]
    if not good:
        for s in stats:
            s["std_used"] = 1.0
        return stats
    zmax = max((s["chan_max"] - s["mean"]) / s["std"] for s in good)
    for s in stats:
        if s["degenerate"]:
            spread = s["chan_max"] - s["mean"]
            s["std_used"] = spread / zmax if spread > 0 and zmax > 0 else 1.0
    return stats


def build_luts(stats, top):
    """value -> uint16 lookup per channel, plus the shared z window."""
    zmin = min((s["chan_min"] - s["mean"]) / s["std_used"] for s in stats)
    zmax = max((s["chan_max"] - s["mean"]) / s["std_used"] for s in stats)
    # Both ends come from real channel extremes, so nothing is ever clipped.
    span = zmax - zmin or 1.0
    values = np.arange(top + 1, dtype=np.float64)
    luts = np.empty((len(stats), top + 1), dtype="<u2")
    for c, s in enumerate(stats):
        z = (values - s["mean"]) / s["std_used"]
        luts[c] = np.clip(np.rint((z - zmin) * U16_MAX / span), 0, U16_MAX)
    return luts, zmin, zmax


def write_normalised(src_path, out_path, luts, comments):
    """Copy a concatenated cube's records, replacing UserComments and the pixels."""
    src, info = open_concat(src_path)
    X, Y, C = info["X"], info["Y"], info["C"]
    pixel_bytes = X * Y * C * 2
    data_head = (_header("Data", T_DATA, 16 + pixel_bytes + len(DATA_TAIL))
                 + struct.pack("<iiii", 3, X, Y, C))
    data_part = ([data_head, (HOLE, pixel_bytes), DATA_TAIL],
                 len(data_head) + pixel_bytes + len(DATA_TAIL))

    with open(src_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        p, top_recs = 4, []
        while p < len(mm) - 16:
            rec, p = _read_record(mm, p)
            top_recs.append(rec)
        root = top_recs[-1]
        dataset = find(mm, root, "DataSet")
        inner = children(mm, dataset)[0]

        inner_parts = []
        for rec in children(mm, inner):
            if rec[0] == "UserComments":
                body = enc_string(comments)
                inner_parts.append(_part("UserComments", T_STRING, [body], len(body)))
            elif rec[0] == "Data":
                inner_parts.append(data_part)
            else:
                inner_parts.append(_verbatim(mm, rec))

        inner_part = _container("", inner_parts, child_pad(mm, inner))
        dataset_part = _container("DataSet", [inner_part], child_pad(mm, dataset))
        root_parts = [dataset_part if rec[0] == "DataSet" else _verbatim(mm, rec)
                      for rec in children(mm, root)]
        root_part = _container(root[0], root_parts, child_pad(mm, root))

        chunks = [struct.pack("<i", COOKIE)]
        for rec in top_recs[:-1]:
            chunks.extend(_verbatim(mm, rec)[0])
        chunks.extend(root_part[0])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".im3.part")
        data_offset = None
        with open(tmp, "wb") as out:
            for chunk in chunks:
                if isinstance(chunk, tuple) and chunk[0] is HOLE:
                    data_offset = out.tell()
                    out.seek(chunk[1] - 1, 1)
                    out.write(b"\0")
                else:
                    out.write(chunk)
            total = out.tell()

    dst = np.memmap(tmp, dtype="<u2", mode="r+", offset=data_offset, shape=(Y, X, C))
    for y in range(0, Y, ROWS):
        chunk = np.asarray(src[y:y + ROWS])
        out_chunk = np.empty_like(chunk)
        for c in range(C):
            out_chunk[:, :, c] = luts[c][chunk[:, :, c]]
        dst[y:y + ROWS] = out_chunk
    dst.flush()
    del dst, src

    tmp.replace(out_path)
    return out_path, total


def normalise_leg(src_path, way, out_root, overwrite=False):
    """Build one normalised cube (and its sidecars) from a way-1 cube."""
    spec = WAYS[way]
    out_path = (out_root / spec["folder"] /
                (src_path.stem.replace("_concat", "") + f"_concat_znorm_{way}.im3"))
    if out_path.exists() and not overwrite:
        return out_path, None

    raw_side = json.loads(src_path.with_name(src_path.stem + "_channels.json").read_text())
    blocks = [{"block": s["block"], "size": s["channels"], "file": s["file"],
               "excitation": s["excitation"]} for s in raw_side["sources"]]

    block_of = [b["block"] for b in blocks for _ in range(b["size"])]

    cube, info = open_concat(src_path)
    hist = channel_histograms(cube)
    stats = channel_stats(hist, way, block_of)
    luts, zmin, zmax = build_luts(stats, hist.shape[1] - 1)
    del cube

    comments = {"normalisation": f"z-score, statistics from the 1-99 percentile window ({way})",
                "way": spec["label"], "source": src_path.name, "leg": raw_side["leg"],
                "z_min": round(zmin, 6), "z_max": round(zmax, 6), "u16_max": U16_MAX,
                "invert": "z = z_min + u*(z_max - z_min)/65535 ; x = z*std[c] + mean[c]",
                "mean": [round(s["mean"], 4) for s in stats],
                "std": [round(s["std_used"], 4) for s in stats],
                "blocks": blocks}
    path, total = write_normalised(src_path, out_path,
                                   luts, json.dumps(comments, separators=(",", ":")))

    rows = []
    for row, s in zip(raw_side["channels"], stats):
        rows.append({**row, "scope": s["scope"], "mean": round(s["mean"], 4),
                     "std": round(s["std_used"], 4), "p1": s["p1"], "p99": s["p99"],
                     "raw_min": s["chan_min"], "raw_max": s["chan_max"],
                     "degenerate": s["degenerate"]})
    fields = list(rows[0].keys())
    csv_path = out_path.with_name(out_path.stem + "_channels.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    meta = {k: raw_side[k] for k in ("leg", "date", "rat", "side", "X", "Y", "C")}
    meta.update({"way": spec["label"], "scope": way, "source": str(src_path),
                 "z_min": zmin, "z_max": zmax, "u16_max": U16_MAX,
                 "invert": comments["invert"], "sources": raw_side["sources"],
                 "channels": rows})
    json_path = out_path.with_name(out_path.stem + "_channels.json")
    json_path.write_text(json.dumps(meta, indent=1))
    return path, total


def run_way(way, src_dir, out_root, overwrite=False):
    """Every concatenated cube in src_dir through one normalisation way."""
    spec = WAYS[way]
    print(f"{spec['label']}  ->  {out_root / spec['folder']}\n")
    for src_path in sorted(src_dir.glob("*_concat.im3")):
        path, total = normalise_leg(src_path, way, out_root, overwrite)
        if total is None:
            print(f"  {path.name:52s} exists, skipped")
        else:
            side = json.loads(path.with_name(path.stem + "_channels.json").read_text())
            print(f"  {path.name:52s} {total / 1e6:.0f} MB  "
                  f"z {side['z_min']:+.2f} .. {side['z_max']:+.2f}", flush=True)


# =============================================================================
# Step 7: check the normalised cubes.
#
# For every file of all three ways: invert a sample of pixels back through the
# recorded transform and compare with the raw cube it came from. The round trip
# is not bit-exact — z is quantised into 65535 steps on the way in — so what is
# reported is the worst error in counts and, more meaningfully, in sigma. Also
# printed: how much of the uint16 range each file actually uses, and what the
# scope statistics did to the mean spread across channels.
# =============================================================================
NORM_SAMPLE = 500


def source_cube(side, out_root):
    """The way-1 cube a normalised one was built from.

    The sidecar records an absolute path, which goes stale the moment a tree is
    moved, so fall back to the same filename in this tree's Raw_Concatenated.
    """
    path = Path(side["source"])
    if path.exists():
        return path
    moved = out_root / "Raw_Concatenated" / path.name
    if moved.exists():
        return moved
    raise FileNotFoundError(f"{path} is gone and {moved} does not exist either")


def verify_norm(out_root, seed=0):
    rng = np.random.default_rng(seed)
    for way, spec in WAYS.items():
        print(f"{spec['label']}")
        for out_path in sorted((out_root / spec["folder"]).glob("*.im3")):
            side = json.loads(out_path.with_name(out_path.stem + "_channels.json").read_text())
            info = read_cube_info(out_path)
            src_path = source_cube(side, out_root)
            src_info = read_cube_info(src_path)

            dst = np.memmap(out_path, dtype="<u2", mode="r", offset=info["data_offset"],
                            shape=(info["Y"], info["X"], info["C"]))
            src = np.memmap(src_path, dtype="<u2", mode="r", offset=src_info["data_offset"],
                            shape=(src_info["Y"], src_info["X"], src_info["C"]))
            ys = rng.integers(0, info["Y"], NORM_SAMPLE)
            xs = rng.integers(0, info["X"], NORM_SAMPLE)

            mean = np.array([c["mean"] for c in side["channels"]])
            std = np.array([c["std"] for c in side["channels"]])
            u = dst[ys, xs, :].astype(np.float64)
            z = side["z_min"] + u * (side["z_max"] - side["z_min"]) / U16_MAX
            err = np.abs(z * std + mean - src[ys, xs, :].astype(np.float64))
            step = (side["z_max"] - side["z_min"]) / U16_MAX

            print(f"  {out_path.name:52s} C={info['C']} "
                  f"scopes={len(set(c['scope'] for c in side['channels']))}  "
                  f"z {side['z_min']:+.2f}..{side['z_max']:+.2f}  step {step:.4f} sigma  "
                  f"round trip <= {err.max():.2f} counts  u {int(u.min())}-{int(u.max())}")
            del dst, src
        print()
