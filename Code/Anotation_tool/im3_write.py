"""Write an .im3 cube by editing an existing one.

Bio-Formats reads .im3 but ships no writer, so this is hand-rolled from the
same record model as the reader — see im3.py and Code/concat.ipynb, which
carries the sibling writer for concatenation.

The approach throughout is **template editing**: an output cube starts as one
of the source files and only the records that must change are rebuilt. Every
acquisition record the instrument wrote — VersionInfo, DisplayPipeline,
Protocol, and the per-band Spectra — is copied through byte for byte, so the
result keeps its wavelengths, exposures and calibration.

For a crop that means three records change:

  Shape            -> the new (X, Y, C)
  Data             -> the cropped pixels, plus the format's 4-byte record tail
  SaturatedPixels  -> emptied; the template's mask described the full frame and
                      a stale mask is worse than none
  UserComments     -> a provenance record saying where the crop came from

The `Thumbnail` record still shows the uncropped preview. It is decoration, not
data, and rebuilding it would mean re-deriving the instrument's own rendering.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from im3 import COOKIE, T_CONTAINER, T_DATA, T_INT, T_STRING, _children, _find, _read_record

CHUNK_ROWS = 64          # stream pixels in row blocks; never hold a whole cube
DATA_TAIL = b"\0" * 4    # the four zero bytes every record in this format ends with


# --- record encoding -------------------------------------------------------
def _header(name, rtype, payload_len):
    nm = name.encode("utf-8")
    return struct.pack("<i", len(nm)) + nm + struct.pack("<ii", payload_len + 8, rtype)


def _raw(buf, rec):
    return bytes(buf[rec[2]:rec[2] + rec[3]])


def _child_pad(buf, rec):
    """The bytes left over after a container's last child, which must survive."""
    p, end = rec[2] + 8, rec[2] + rec[3]
    while p < end - 8:
        _, p = _read_record(buf, p)
    return bytes(buf[p:end])


def enc_ints(values):
    return (struct.pack("<ii", 1, len(values))
            + struct.pack(f"<{len(values)}i", *values) + b"\0" * 4)


def enc_string(text):
    b = text.encode("utf-8")
    return struct.pack("<ii", 0, len(b)) + b + b"\0" * 4


class _Part:
    """A record as (chunks, total length). PIXELS marks the streamed region."""

    __slots__ = ("chunks", "size")

    def __init__(self, chunks, size):
        self.chunks, self.size = chunks, size


PIXELS = object()


def _part(name, rtype, body, body_len):
    head = _header(name, rtype, body_len)
    return _Part([head] + body, len(head) + body_len)


def _container(name, parts, pad=b""):
    body_len = 8 + sum(p.size for p in parts) + len(pad)
    body = [struct.pack("<ii", 1, len(parts))]
    for p in parts:
        body.extend(p.chunks)
    if pad:
        body.append(pad)
    return _part(name, T_CONTAINER, body, body_len)


def _verbatim(buf, rec):
    blob = _header(rec[0], rec[1], rec[3]) + _raw(buf, rec)
    return _Part([blob], len(blob))


# --- the writer ------------------------------------------------------------
def write_cropped(src_path, out_path, box, comments=None):
    """Write `src_path` cropped to box = (x0, y0, x1, y1), returning (X, Y, C).

    Pixels are band-interleaved-by-pixel — the payload is (Y, X, C) — so a crop
    is a plain slice of rows and columns with the spectral axis untouched. They
    are streamed in row blocks rather than materialised, which keeps a 217-band
    cube writable in a few MB of memory.
    """
    import mmap

    x0, y0, x1, y1 = (int(v) for v in box)
    src_path, out_path = Path(src_path), Path(out_path)

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
        pixels_off = data[2] + 16

        if not (0 <= x0 < x1 <= X and 0 <= y0 < y1 <= Y):
            raise ValueError(f"crop {box} is outside {src_path.name} ({X}x{Y})")
        nx, ny = x1 - x0, y1 - y0
        pixel_bytes = nx * ny * C * 2

        # Every record in this format ends with four zero bytes — enc_ints and
        # enc_string above do it, and so does the instrument's own Data record.
        # Omitting it produced files our reader and Bio-Formats both accepted
        # but Nuance rejected as "Data defined in im3 is corrupt, File position
        # = 4294967295" (a 32-bit -1: it went looking and found nothing).
        data_head = _header("Data", T_DATA, 16 + pixel_bytes + len(DATA_TAIL)) \
            + struct.pack("<iiii", 3, nx, ny, C)
        data_part = _Part([data_head, PIXELS, DATA_TAIL],
                          len(data_head) + pixel_bytes + len(DATA_TAIL))
        shape_body = enc_ints([nx, ny, C])

        inner_parts = []
        for rec in _children(mm, inner):
            if rec[0] == "Shape":
                inner_parts.append(_part("Shape", T_INT, [shape_body], len(shape_body)))
            elif rec[0] == "Data":
                inner_parts.append(data_part)
            elif rec[0] == "SaturatedPixels":
                inner_parts.append(_container("SaturatedPixels", []))
            elif rec[0] == "UserComments" and comments is not None:
                body = enc_string(comments)
                inner_parts.append(_part("UserComments", T_STRING, [body], len(body)))
            else:
                inner_parts.append(_verbatim(mm, rec))

        inner_part = _container("", inner_parts, _child_pad(mm, inner))
        dataset_part = _container("DataSet", [inner_part], _child_pad(mm, dataset))
        root_parts = [dataset_part if rec[0] == "DataSet" else _verbatim(mm, rec)
                      for rec in _children(mm, root)]
        root_part = _container(root[0], root_parts, _child_pad(mm, root))

        chunks = [struct.pack("<i", COOKIE)]
        for rec in top[:-1]:                        # FileVersion and friends
            chunks.extend(_verbatim(mm, rec).chunks)
        chunks.extend(root_part.chunks)

        src = np.frombuffer(mm, dtype="<u2", count=X * Y * C,
                            offset=pixels_off).reshape(Y, X, C)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".im3.part")
        try:
            with open(tmp, "wb") as out:
                for chunk in chunks:
                    if chunk is PIXELS:
                        # The slice stays an unnamed temporary on purpose: any
                        # view still bound when the mmap closes raises
                        # "cannot close exported pointers exist".
                        for y in range(y0, y1, CHUNK_ROWS):
                            out.write(np.ascontiguousarray(
                                src[y:min(y + CHUNK_ROWS, y1), x0:x1, :]).tobytes())
                    else:
                        out.write(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            del src

    tmp.replace(out_path)
    return nx, ny, C
