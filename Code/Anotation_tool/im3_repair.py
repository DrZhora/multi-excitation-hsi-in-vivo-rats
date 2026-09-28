"""Rebuild a written .im3 closer to what the instrument itself writes.

`write_cropped` produces files that are self-consistent — Shape, the Data
header and the pixel block all agree, our own reader and Bio-Formats both open
them — but they differ from an instrument-written cube in two places, and
Nuance is stricter than Bio-Formats:

  * **the Data record has no 4-byte tail.** Every genuine record in this format
    ends with four zero bytes; `enc_ints` and `enc_string` in im3_write honour
    that, and the Data record does not. Measured on Data/Raw, an original Data
    payload is `16 + X*Y*C*2 + 4` bytes and ours is `16 + X*Y*C*2`.
  * **SaturatedPixels is an empty container.** The template's mask described the
    full frame, so the crop emptied it rather than carry a stale one, and a
    reader that expects the BoundingRect/Color/Pixels inside may read back "not
    found" — which is what a file position of 4294967295 (a 32-bit -1) looks
    like.

**Tested against Nuance on 2026-09-21: the Data tail is the cause.** A cube with
the tail added opens whether SaturatedPixels is restored or not; without it,
Nuance refuses. Both writers now emit the tail, and

    python im3_repair.py <folder-or-cube> --patch [--dry-run]

fixes files already on disk. That is an in-place edit, not a rewrite: `Data` is
the last record and everything after it is zero padding, so the tail is four
appended bytes plus four length fields bumped by four — a few hundred bytes of
I/O whatever the cube's size, and the pixel block never moves. It verifies each
file afterwards and refuses anything whose `Data` is not last.

To build the diagnostic variants instead of patching:

    python im3_repair.py <cube.im3> --out-dir DIR [--template ORIGINAL.im3]

Pixels are copied through byte for byte; only the record framing changes.
"""

from __future__ import annotations

import argparse
import mmap
import struct
from pathlib import Path

from im3 import COOKIE, T_DATA, _children, _find, _read_record
from im3_write import _child_pad, _container, _header, _Part, _part, _raw, _verbatim

TAIL = b"\0" * 4


def rebuild(src_path, out_path, data_tail=True, saturated_from=None):
    """Copy `src_path` record for record, optionally restoring the two fields.

    `data_tail` appends the four zero bytes an instrument-written Data record
    carries. `saturated_from` copies the SaturatedPixels record out of another
    file verbatim — meant as an experiment, since that mask describes the frame
    it came from and not necessarily this one.
    """
    src_path, out_path = Path(src_path), Path(out_path)

    donor = None
    if saturated_from is not None:
        with open(saturated_from, "rb") as fh:
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as dm:
                p, top = 4, []
                while p < len(dm) - 16:
                    rec, p = _read_record(dm, p)
                    top.append(rec)
                inner = _children(dm, _find(dm, top[-1], "DataSet"))[0]
                rec = _find(dm, inner, "SaturatedPixels")
                donor = _header(rec[0], rec[1], rec[3]) + _raw(dm, rec)

    with open(src_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if struct.unpack_from("<i", mm, 0)[0] != COOKIE:
            raise ValueError(f"{src_path.name} is not an .im3 (bad cookie)")
        p, top = 4, []
        while p < len(mm) - 16:
            rec, p = _read_record(mm, p)
            top.append(rec)

        root = top[-1]
        dataset = _find(mm, root, "DataSet")
        inner = _children(mm, dataset)[0]
        data = _find(mm, inner, "Data")
        _, X, Y, C = struct.unpack_from("<iiii", mm, data[2])
        pixel_bytes = X * Y * C * 2
        pixels_off = data[2] + 16
        if pixels_off + pixel_bytes > len(mm):
            raise ValueError(f"{src_path.name}: truncated pixel block")

        tail = TAIL if data_tail else b""
        head = (_header("Data", T_DATA, 16 + pixel_bytes + len(tail))
                + struct.pack("<iiii", 3, X, Y, C))
        body = [head, ("PIXELS", pixels_off, pixel_bytes)]
        if tail:
            body.append(tail)
        data_part = _Part(body, len(head) + pixel_bytes + len(tail))

        parts = []
        for rec in _children(mm, inner):
            if rec[0] == "Data":
                parts.append(data_part)
            elif rec[0] == "SaturatedPixels" and donor is not None:
                parts.append(_Part([donor], len(donor)))
            else:
                parts.append(_verbatim(mm, rec))

        inner_part = _container("", parts, _child_pad(mm, inner))
        dataset_part = _container("DataSet", [inner_part], _child_pad(mm, dataset))
        root_parts = [dataset_part if rec[0] == "DataSet" else _verbatim(mm, rec)
                      for rec in _children(mm, root)]
        root_part = _container(root[0], root_parts, _child_pad(mm, root))

        chunks = [struct.pack("<i", COOKIE)]
        for rec in top[:-1]:
            chunks.extend(_verbatim(mm, rec).chunks)
        chunks.extend(root_part.chunks)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".im3.part")
        try:
            with open(tmp, "wb") as out:
                for chunk in chunks:
                    if isinstance(chunk, tuple) and chunk[0] == "PIXELS":
                        _, off, n = chunk
                        step = 8 << 20
                        for s in range(off, off + n, step):
                            out.write(mm[s:min(s + step, off + n)])
                    else:
                        out.write(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    tmp.replace(out_path)
    return out_path


# ---------------------------------------------------------------------------
# The in-place patch
#
# Rewriting 39 GB of cubes to insert four bytes each would be absurd, and it is
# not necessary: `Data` is the last record in the file, and everything after it
# is container padding that is already zero. So adding the tail means appending
# four zero bytes and bumping the four length fields that enclose it — a few
# hundred bytes of I/O per file, whatever its size, and the pixel block never
# moves at all.
# ---------------------------------------------------------------------------
def _tree(mm):
    """(root, DataSet, inner, Data) as records, plus each one's length offset."""
    def rd(pos):
        n, = struct.unpack_from("<i", mm, pos)
        name = bytes(mm[pos + 4:pos + 4 + n]).decode("utf-8", "replace")
        q = pos + 4 + n
        reclen, rtype = struct.unpack_from("<ii", mm, q)
        return dict(name=name, type=rtype, lenfield=q, reclen=reclen,
                    start=q + 8, size=reclen - 8)

    def kids(rec):
        out, pos, end = [], rec["start"] + 8, rec["start"] + rec["size"]
        while pos < end - 8:
            c = rd(pos)
            out.append(c)
            pos = c["start"] + c["size"]
        return out

    pos, top = 4, []
    while pos < len(mm) - 16:
        r = rd(pos)
        top.append(r)
        pos = r["start"] + r["size"]
    root = top[-1]
    ds = next(c for c in kids(root) if c["name"] == "DataSet")
    inner = kids(ds)[0]
    inner_kids = kids(inner)
    data = inner_kids[-1]
    return root, ds, inner, data, data["name"] == "Data"


def patch_data_tail(path, dry_run=False):
    """Give one cube the 4-byte Data tail in place. Returns a status string."""
    path = Path(path)
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if struct.unpack_from("<i", mm, 0)[0] != COOKIE:
            return "not an .im3"
        try:
            root, ds, inner, data, data_is_last = _tree(mm)
        except Exception as exc:
            return f"unreadable ({exc})"
        if not data_is_last:
            return "Data is not the last record — needs a full rebuild"

        _, X, Y, C = struct.unpack_from("<iiii", mm, data["start"])
        pixels = X * Y * C * 2
        have = data["size"] - 16
        if have == pixels + len(TAIL):
            return "already has the tail"
        if have != pixels:
            return f"unexpected Data payload ({have} vs {pixels})"

        end = data["start"] + data["size"]
        if root["start"] + root["size"] != len(mm):
            return "root does not end at EOF"
        if any(mm[end:len(mm)]):
            return "trailing bytes are not padding"
        lens = [(r["lenfield"], r["reclen"]) for r in (data, inner, ds, root)]

    if dry_run:
        return "would patch"

    # Append first, then grow the lengths. The other order would leave a file
    # claiming four bytes it does not have if this were interrupted.
    with open(path, "r+b") as fh:
        fh.seek(0, 2)
        fh.write(TAIL)
        fh.flush()
        for off, reclen in lens:
            fh.seek(off)
            fh.write(struct.pack("<i", reclen + 4))
        fh.flush()

    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        root, ds, inner, data, ok = _tree(mm)
        _, nX, nY, nC = struct.unpack_from("<iiii", mm, data["start"])
        if not ok or (nX, nY, nC) != (X, Y, C):
            return "VERIFY FAILED: shape changed"
        if data["size"] - 16 != pixels + len(TAIL):
            return "VERIFY FAILED: payload wrong"
        if root["start"] + root["size"] != len(mm):
            return "VERIFY FAILED: root no longer ends at EOF"
    return "patched"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cube", help="a written .im3, or a folder to walk with --patch")
    ap.add_argument("--out-dir")
    ap.add_argument("--patch", action="store_true",
                    help="add the 4-byte Data tail in place, recursively")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--template", help="an instrument-written .im3 to take "
                                       "SaturatedPixels from, for variant C")
    args = ap.parse_args(argv)

    src = Path(args.cube)

    if args.patch:
        cubes = sorted(src.rglob("*.im3")) if src.is_dir() else [src]
        tally = {}
        for c in cubes:
            status = patch_data_tail(c, dry_run=args.dry_run)
            tally[status] = tally.get(status, 0) + 1
            if status not in ("patched", "would patch", "already has the tail"):
                print(f"  !! {status:46s} {c}")
            print(f"\r  {sum(tally.values())}/{len(cubes)}", end="", flush=True)
        print("\r" + " " * 30 + "\r", end="")
        for status, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {status}")
        return 0 if all(k in ("patched", "would patch", "already has the tail")
                        for k in tally) else 1

    if not args.out_dir:
        ap.error("--out-dir is required unless --patch is given")
    out = Path(args.out_dir)
    made = [
        ("A_unchanged", rebuild(src, out / f"A_unchanged__{src.name}", data_tail=False)),
        ("B_data_tail", rebuild(src, out / f"B_data_tail__{src.name}", data_tail=True)),
    ]
    if args.template:
        made.append(("C_tail_and_saturated",
                     rebuild(src, out / f"C_tail_and_saturated__{src.name}",
                             data_tail=True, saturated_from=args.template)))
    for name, path in made:
        print(f"  {name:22s} {path.stat().st_size:>12,} bytes  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
