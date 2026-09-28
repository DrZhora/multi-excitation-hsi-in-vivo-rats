"""Translation-only registration for the cubes of one leg.

The seven cubes of a leg are the same scene under different excitations, taken
minutes apart, so the animal can drift between them. They are annotated once,
in a shared frame, which only works if we know how far each cube has moved.

Everything here is whole-pixel translation. That keeps the mapping lossless —
a mask warped into a cube's frame and back is unchanged — and it is what the
data needs: the stage does not rotate, the animal shifts. Rotation and scale
are not estimated, and would show up as a stubbornly low score that no nudge
can fix.

Two things make this work on real cubes:

- **Match on structure, not brightness.** Excitations differ wildly in
  intensity and in which tissue fluoresces, but the vessels sit in the same
  place, so registration runs on gradient magnitude, standardised and
  Hann-windowed.
- **Measure every pair, then solve jointly.** A direct match between two
  distant excitations can fail outright (on the 12.09 Rat_1 left leg, 310 nm
  against 325 nm scores 0.14 and lands 300 px out). Measuring all pairs and
  fitting one offset per cube by weighted least squares routes such a cube
  through the neighbours it does match — that same 325 nm cube lands within
  a few pixels of its true position.
"""

from __future__ import annotations

import itertools

import numpy as np


def prep(img):
    """Gradient magnitude, standardised and windowed — the registration image.

    The Hann window matters: the FFT treats the image as periodic, and without
    it the frame edges correlate with each other and can outvote the anatomy.
    """
    x = np.asarray(img, dtype=np.float32)
    gy, gx = np.gradient(x)
    g = np.hypot(gx, gy)
    g -= g.mean()
    s = g.std()
    if s > 0:
        g /= s
    h, w = g.shape
    return g * np.hanning(h).astype(np.float32)[:, None] * np.hanning(w).astype(np.float32)[None, :]


def phase_shift(a, b):
    """Whole-pixel (dx, dy) such that a feature at p in `b` sits at p + t in `a`.

    Returns (dx, dy, peak, psr). `psr` is the peak-to-sidelobe ratio: how much
    the winning offset stands out from the rest of the correlation surface.
    """
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)          # phase only: amplitude carries no shift
    c = np.fft.irfft2(R, s=a.shape)

    h, w = a.shape
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    peak = float(c[iy, ix])
    dy = int(iy - h) if iy > h // 2 else int(iy)
    dx = int(ix - w) if ix > w // 2 else int(ix)

    skirt = c.copy()
    skirt[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4] = 0
    psr = float((peak - skirt.mean()) / (skirt.std() + 1e-12))
    return dx, dy, peak, psr


def ncc(a, b, dx, dy):
    """Normalised cross-correlation over the overlap, after shifting b by t.

    This is the number worth showing a human: 1.0 is identical, and on these
    cubes a correct match between two different excitations lands around
    0.4-0.9 while a failed one sits near zero.
    """
    h, w = a.shape
    ay0, ay1 = max(0, dy), min(h, h + dy)
    ax0, ax1 = max(0, dx), min(w, w + dx)
    if ay1 - ay0 < 8 or ax1 - ax0 < 8:
        return 0.0
    u = a[ay0:ay1, ax0:ax1].ravel()
    v = b[ay0 - dy:ay1 - dy, ax0 - dx:ax1 - dx].ravel()
    u = u - u.mean()
    v = v - v.mean()
    d = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(u @ v / d) if d > 0 else 0.0


def shift_image(img, dx, dy, fill=0):
    """out[y, x] = img[y - dy, x - dx], with anything shifted in set to `fill`."""
    img = np.asarray(img)
    h, w = img.shape
    out = np.full((h, w), fill, img.dtype)
    ty0, ty1 = max(0, dy), min(h, h + dy)
    tx0, tx1 = max(0, dx), min(w, w + dx)
    if ty1 > ty0 and tx1 > tx0:
        out[ty0:ty1, tx0:tx1] = img[ty0 - dy:ty1 - dy, tx0 - dx:tx1 - dx]
    return out


def ref_to_cube(mask_ref, dx, dy, fill=255):
    """A reference-frame mask expressed in the frame of a cube at offset (dx, dy).

    The cube's pixel (x, y) is the reference's (x + dx, y + dy), so this is
    `shift_image` run backwards. Pixels the cube cannot see become `fill`.
    """
    return shift_image(mask_ref, -dx, -dy, fill)


def cube_to_ref(mask_cube, dx, dy, fill=255):
    """The inverse of ref_to_cube: a cube-frame mask lifted into the reference."""
    return shift_image(mask_cube, dx, dy, fill)


GOOD_ENOUGH = 0.45          # below this, do not trust one correlation peak


def _search(a, b, cx, cy, span, step=1):
    """Exhaustive NCC over a window. Slow but incapable of picking a side lobe."""
    best = (-2.0, cx, cy)
    for dy in range(cy - span, cy + span + 1, step):
        for dx in range(cx - span, cx + span + 1, step):
            s = ncc(a, b, dx, dy)
            if s > best[0]:
                best = (s, dx, dy)
    return best


def _polish(pyr_a, pyr_b, dx, dy):
    """Walk a candidate down the pyramid: +-3 at 1/2 scale, then +-1 at full."""
    s, dx, dy = _search(pyr_a[1], pyr_b[1], dx // 2, dy // 2, 3)
    s, dx, dy = _search(pyr_a[0], pyr_b[0], dx * 2, dy * 2, 1)
    return s, dx, dy


def measure(pyr_a, pyr_b, max_shift=256):
    """The shift between two cubes, by phase correlation with a safety net.

    Phase correlation is fast and usually right, but it commits to a single
    peak — and on a pair that genuinely does not match well it commits to the
    wrong one. On 10.09/RAT_2 Right_leg it put 385 nm 155 px from where an
    exhaustive search puts it, and the difference is not detectable from the
    peak alone.

    So: take the peak, polish it, and if the result is still unconvincing, pay
    for a coarse exhaustive search at 1/8 scale and polish that too. Whichever
    candidate correlates better wins. Good pairs cost one extra 1/2-scale
    window; only doubtful ones pay for the sweep.
    """
    dx0, dy0, _, psr = phase_shift(pyr_a[0], pyr_b[0])
    best = _polish(pyr_a, pyr_b, dx0, dy0)

    if best[0] < GOOD_ENOUGH:
        span = max(1, max_shift // 8)
        _, cx, cy = _search(pyr_a[3], pyr_b[3], 0, 0, span)
        alt = _polish(pyr_a, pyr_b, cx * 8, cy * 8)
        if alt[0] > best[0]:
            best = alt
    return {"dx": int(best[1]), "dy": int(best[2]), "ncc": max(0.0, float(best[0])),
            "psr": float(psr)}


def pyramid(img, levels=4):
    """[full, 1/2, 1/4, 1/8] prepped images — gradients taken at each scale."""
    return [prep(img[::2 ** k, ::2 ** k]) for k in range(levels)]


def pairwise(pyrs):
    """Every pair measured. pyrs: list of pyramids. -> list of records."""
    out = []
    for i, j in itertools.combinations(range(len(pyrs)), 2):
        rec = measure(pyrs[i], pyrs[j])
        out.append({"i": i, "j": j, **rec})
    return out


def solve_offsets(n, obs, ref, rounds=6):
    """One (dx, dy) per cube from all pairwise measurements, reference pinned at 0.

    Each observation says `o[j] - o[i] = (dx, dy)`. That is an over-determined
    linear system; it is solved by weighted least squares, weighted by score²,
    and re-solved a few times with outliers pulled down (IRLS with a Huber-like
    cut at 2 robust sigmas) so one confidently-wrong pair cannot drag the rest.
    """
    if not obs:
        return np.zeros((n, 2), int), np.zeros(0)

    m = np.array([[o["dx"], o["dy"]] for o in obs], float)
    base = np.array([o["ncc"] for o in obs], float) ** 2
    w = base.copy()
    o = np.zeros((n, 2))
    res = np.zeros(len(obs))

    for _ in range(rounds):
        A = np.zeros((len(obs) + 1, n))
        B = np.zeros((len(obs) + 1, 2))
        for k, rec in enumerate(obs):
            A[k, rec["j"]] = w[k]
            A[k, rec["i"]] = -w[k]
            B[k] = m[k] * w[k]
        A[-1, ref] = 1e3                         # pin: the reference does not move
        o = np.linalg.lstsq(A, B, rcond=None)[0]
        res = np.hypot(*((o[[r["j"] for r in obs]] - o[[r["i"] for r in obs]] - m).T))
        sigma = max(float(np.median(res)), 1.0) * 1.4826
        w = base * np.minimum(1.0, (2 * sigma) / np.maximum(res, 1e-6))

    o = o - o[ref]
    return np.rint(o).astype(int), res


def best_reference(n, obs):
    """The cube that agrees best with all the others, by mean pair score.

    Whichever cube is most central makes the smallest shifts for everyone else
    and gives the sharpest picture to annotate against. On the 12.09 Rat_1 left
    leg that is the 400 nm cube, not the first one in band order.
    """
    if n == 1 or not obs:
        return 0
    total = np.zeros(n)
    for rec in obs:
        total[rec["i"]] += rec["ncc"]
        total[rec["j"]] += rec["ncc"]
    return int(np.argmax(total))


def overlap_box(offsets, width, height):
    """The reference-frame rectangle every cube can see: [x0, y0, x1, y1).

    Painting outside it still works, but those labels fall off the edge of at
    least one cube and are saved as unlabeled there.
    """
    x0 = y0 = 0
    x1, y1 = width, height
    for dx, dy in offsets:
        x0 = max(x0, dx)
        y0 = max(y0, dy)
        x1 = min(x1, width + dx)
        y1 = min(y1, height + dy)
    return [int(x0), int(y0), int(max(x0, x1)), int(max(y0, y1))]


# A cube is judged unalignable by comparison, not by an absolute score.
#
# Absolute correlation is the wrong gate: a white-light cube against a
# fluorescence one scores ~0.28 however perfectly it is placed, because the two
# image different things. What gives a cube away is matching something ELSE far
# better than it matches the reference frame — that is a second pose, not a
# weak match. Measured over all 48 cubes of this dataset the ratio is 0.62-1.00
# for every cube that is genuinely alignable, and 0.35 / 0.28 for the two that
# are not (10.09 RAT_2 Right_leg, 385 and 400 nm).
ALIGN_RATIO = 0.50          # score against the reference, over its best pair score
ALIGN_SPREAD = 12.0         # px: how far partners disagree about where it goes


def _to(obs_by, j, k):
    """The measured shift putting cube k into cube j's frame."""
    return obs_by[(j, k)]


def register(images, ref=None):
    """Full registration of one group. images: list of 2-D arrays (any dtype).

    Returns (offsets, reference_index, records). Each record carries the direct
    score against the chosen reference and an `alignable` flag — a cube that
    cannot be brought onto the reference by ANY translation is a fact about the
    data, not a number to be quietly rounded off.
    """
    pyrs = [pyramid(im) for im in images]
    obs = pairwise(pyrs)
    obs_by = {(o["i"], o["j"]): (o["dx"], o["dy"]) for o in obs}
    obs_by.update({(o["j"], o["i"]): (-o["dx"], -o["dy"]) for o in obs})
    if ref is None:
        ref = best_reference(len(pyrs), obs)
    offsets, _ = solve_offsets(len(pyrs), obs, ref)

    # The joint solve is a compromise across every pair; for the cube itself
    # what matters is how well it lands on the reference, so each offset gets a
    # last local polish against it.
    records = []
    for k in range(len(pyrs)):
        if k == ref:
            offsets[k] = [0, 0]
            records.append({"dx": 0, "dy": 0, "score": 1.0, "score_unshifted": 1.0,
                            "alignable": True})
            continue
        dx, dy = int(offsets[k][0]), int(offsets[k][1])
        score, dx, dy = _polish(pyrs[ref], pyrs[k], dx, dy)
        offsets[k] = [dx, dy]

        best_pair = max((o["ncc"] for o in obs if k in (o["i"], o["j"])), default=0.0)
        ratio = float(score) / max(best_pair, 1e-6)
        implied = np.array([[offsets[j][0] + _to(obs_by, j, k)[0],
                             offsets[j][1] + _to(obs_by, j, k)[1]]
                            for j in range(len(pyrs)) if j != k], float)
        spread = float(np.max(np.hypot(*(implied - np.median(implied, axis=0)).T)))

        records.append({
            "dx": int(dx), "dy": int(dy),
            "score": round(float(score), 4),
            "score_unshifted": round(ncc(pyrs[ref][0], pyrs[k][0], 0, 0), 4),
            "best_pair": round(best_pair, 4),
            "ratio": round(ratio, 3),
            "spread_px": round(spread, 1),
            "alignable": bool(ratio >= ALIGN_RATIO and spread <= ALIGN_SPREAD),
        })
    return offsets, ref, records
