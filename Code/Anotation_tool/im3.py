"""Minimal, JVM-free reader for Nuance/Maestro .im3 cubes.

The record model is the same one Code/concat.ipynb carries (derived from
Bio-Formats' IM3Reader): a file is `int32 cookie = 1985` followed by records to
EOF, and a record is

    int32 nameLen, name, int32 recLen (= payload + 8), int32 type, payload

Only the reading half lives here, and only the parts the annotator needs: the
shape, the per-band emission wavelengths and single-channel access.

Pixels are band-interleaved-by-pixel — the payload is (Y, X, C) — so one
channel is a strided slice of the file and one pixel's whole spectrum is
contiguous. Cubes run from 90 MB (31 bands) to 630 MB (217 bands), so nothing
here ever materialises a whole cube: the file stays memory-mapped and each
channel is copied out on demand.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path

import numpy as np

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


def _children(buf, rec):
    """Child records of a container, in order."""
    p, end, out = rec[2] + 8, rec[2] + rec[3], []
    while p < end - 8:                      # a tail shorter than a header is padding
        child, p = _read_record(buf, p)
        out.append(child)
    return out


def _find(buf, rec, name):
    for child in _children(buf, rec):
        if child[0] == name:
            return child
    raise KeyError(f"no {name!r} record inside {rec[0]!r}")


def _numbers(buf, rec):
    """Values of an int (type 6) or float (type 7) record."""
    fmt = "i" if rec[1] == T_INT else "f"
    flag, = struct.unpack_from("<i", buf, rec[2])
    if flag == 1:
        count, = struct.unpack_from("<i", buf, rec[2] + 4)
        return list(struct.unpack_from(f"<{count}{fmt}", buf, rec[2] + 8))
    return list(struct.unpack_from(f"<1{fmt}", buf, rec[2] + 4))


class Im3Cube:
    """A memory-mapped .im3. Open it, ask for channels, close it.

    Nothing is cached in here — the caller owns caching — but the mapping is
    kept open, so reading channel after channel costs a page-cache hit rather
    than a re-open. Use it as a context manager or call close().
    """

    def __init__(self, path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    def _parse(self):
        mm = self._mm
        if struct.unpack_from("<i", mm, 0)[0] != COOKIE:
            raise ValueError(f"{self.path.name} is not an .im3 (bad cookie)")

        p, top = 4, []
        while p < len(mm) - 16:
            rec, p = _read_record(mm, p)
            top.append(rec)

        inner = _children(mm, _find(mm, top[-1], "DataSet"))[0]
        data = _find(mm, inner, "Data")
        _, X, Y, C = struct.unpack_from("<iiii", mm, data[2])
        self.width, self.height, self.bands = int(X), int(Y), int(C)
        self._pixels_off = data[2] + 16

        need = self.width * self.height * self.bands * 2
        have = len(mm) - self._pixels_off
        if have < need:
            raise ValueError(
                f"{self.path.name}: truncated, need {need} pixel bytes, file has {have}")

        # Per-band emission wavelength, when the Spectra records are intact.
        # A cube missing them is still perfectly annotatable, so this never
        # raises — the UI just falls back to channel indices.
        self.wavelengths = []
        try:
            for band in _children(mm, _find(mm, inner, "Spectra")):
                spectrum = _children(mm, _find(mm, band, "Spectrum"))[0]
                self.wavelengths.append(float(_numbers(mm, _find(mm, spectrum, "Wavelengths"))[0]))
        except Exception:
            self.wavelengths = []
        if len(self.wavelengths) != self.bands:
            self.wavelengths = []

    @property
    def shape(self):
        """(C, Y, X), matching how the rest of the project talks about cubes."""
        return (self.bands, self.height, self.width)

    def channel(self, c):
        """Channel c as a (Y, X) uint16 array, copied out of the mapping."""
        if not 0 <= c < self.bands:
            raise IndexError(f"channel {c} out of range 0..{self.bands - 1}")
        n = self.width * self.height * self.bands
        flat = np.frombuffer(self._mm, dtype="<u2", count=n, offset=self._pixels_off)
        out = np.ascontiguousarray(flat[c::self.bands].reshape(self.height, self.width))
        del flat                    # views must die before the mmap can close
        return out

    def mean_image(self, max_channels=32):
        """Mean over the spectral axis, as a (Y, X) float32.

        Channels are subsampled to at most max_channels because the point is a
        quick anatomical overview, not a radiometric quantity, and a 217-band
        cube is 630 MB to walk in full.
        """
        idx = np.linspace(0, self.bands - 1, min(self.bands, max_channels))
        idx = np.unique(idx.round().astype(int))
        acc = np.zeros((self.height, self.width), np.float64)
        for c in idx:
            acc += self.channel(int(c))
        return (acc / len(idx)).astype(np.float32)

    def close(self):
        mm = getattr(self, "_mm", None)
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                pass                # a caller still holds a view; the fh close below is enough
            self._mm = None
        fh = getattr(self, "_fh", None)
        if fh is not None:
            fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def peek(path):
    """(bands, height, width, wavelengths) of a cube, without keeping it open."""
    with Im3Cube(path) as cube:
        return cube.bands, cube.height, cube.width, list(cube.wavelengths)
