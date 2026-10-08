"""Ceiling of a second, 2-pixel tile level under the 4-pixel beam tiles (offline, no Mogwai).

Today a 4 x 4 tile the test fails marches all 16 of its pixels as units (listBeamFailedTileUnits). With a second level the failed
tile would add the 2-pixel lattice points it lacks (shared along edges with its neighbours) and test its four 2 x 2 sub-tiles,
marching only the 3 non-lattice pixels of a sub-tile that fails. This scores both tilings with a PERFECT test on one exact frame
(sunset_exact_capture.py: the per-pixel march at the shipping step): a tile or sub-tile fails when any of its pixels, bilinear
from its corners in linear radiance, is off the exact pixel by more than BAR in the compare metric (mean over rgb of
|log(1 + a) - log(1 + b)|, beamOracleBar). The ratio of the two costs is what the level could at best save the units; the real
test (corner curvature) lists more at both levels.

    python scripts/HSTR/bench/tile_oracle.py <frame>_color.raw [bar 0.02]

MEASURED (walkstart: sunset start view, 3840x2160, shipping step, residency settled; cost in units with a new lattice ray at 1.3):
- Perfect tests: bar 0.02 - 56,687 of 516,901 tiles fail (11.0%), 907k units; two levels 314k units + 192k lattice rays, -38%
  (-44% at 1.0). Bar 0.05: -46%. Only 46% (35%) of a failed tile's sub-tiles fail.
- The real test emulated (corner curvature, no edge test), error = share of pixels over 0.02 against the exact frame:
  one level 0.4 419k / 2.255%, 0.35 469k / 1.972%, 0.3 528k / 1.667%, 0.25 600k / 1.346%, 0.2 703k / 0.982%, 0.1 1096k / 0.263%.
  Two levels, curvature below too: 0.4 + 0.1 394k / 2.266%, 0.3 + 0.05 511k / 1.668%, 0.25 + 0.05 578k / 1.347% - 3-8% at equal
  error: the curvature test cannot tell which sub-tiles reconstruct.
  Two levels with a centre test below (the sub-tile's centre pixel marched first, its bilinear prediction against it): 0.3 + 0.02
  403k / 1.863%, 0.25 + 0.01 501k / 1.444%, 0.2 + 0.01 574k / 1.090% - 13-18% fewer than one level at equal error (~490k / 578k
  / 672k interpolated). Static frame only; the dirty build's units are a subset, and motion is untested.
"""
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        w, h, bpp = struct.unpack("<3I", f.read(12))
        dtype = {8: np.float16, 16: np.float32}[bpp]
        img = np.frombuffer(f.read(), dtype=dtype).reshape(h, w, 4)[..., :3].astype(np.float32)
    return np.maximum(img, 0.0)


def fails(img, s, bar):
    """Per s-pixel tile (s divides the frame), whether bilinear from its corners misses any pixel by more than bar."""
    h, w, _ = img.shape
    th, tw = (h - 1) // s, (w - 1) // s  # tiles whose far corners exist
    lat = img[: th * s + 1 : s, : tw * s + 1 : s]  # (th+1, tw+1, 3)
    logimg = np.log1p(img)
    worst = np.zeros((th, tw), np.float32)
    for dy in range(s):
        fy = dy / s
        for dx in range(s):
            fx = dx / s
            pred = ((1 - fx) * (1 - fy) * lat[:-1, :-1] + fx * (1 - fy) * lat[:-1, 1:] + (1 - fx) * fy * lat[1:, :-1] + fx * fy * lat[1:, 1:])
            exact = logimg[dy : th * s : s, dx : tw * s : s][:th, :tw]
            err = np.abs(np.log1p(pred) - exact).mean(axis=-1)
            worst = np.maximum(worst, err)
    return worst > bar


def curvature_fails(img, s, tol):
    """The centreless test (beamCornerCurvature) on the s-pixel lattice: per tile, the largest second difference of log(1 + radiance)
    along either axis at its four corners, against tol. The edge (transmittance contrast) test is not emulated."""
    h, w, _ = img.shape
    th, tw = (h - 1) // s, (w - 1) // s
    lat = np.log1p(img[: th * s + 1 : s, : tw * s + 1 : s])
    p = np.pad(lat, ((1, 1), (1, 1), (0, 0)), mode="edge")
    c = p[1:-1, 1:-1]
    dxx = np.abs(p[1:-1, :-2] - 2 * c + p[1:-1, 2:]).max(axis=-1)
    dyy = np.abs(p[:-2, 1:-1] - 2 * c + p[2:, 1:-1]).max(axis=-1)
    k = np.maximum(dxx, dyy)
    corner = np.maximum(np.maximum(k[:-1, :-1], k[:-1, 1:]), np.maximum(k[1:, :-1], k[1:, 1:]))
    return corner > tol


def bilinear_image(img, s):
    """Every pixel of the frame's whole tiles bilinear from its s-pixel tile's corners."""
    h, w, _ = img.shape
    th, tw = (h - 1) // s, (w - 1) // s
    lat = img[: th * s + 1 : s, : tw * s + 1 : s]
    out = np.empty((th * s, tw * s, 3), np.float32)
    for dy in range(s):
        fy = dy / s
        for dx in range(s):
            fx = dx / s
            out[dy::s, dx::s] = ((1 - fx) * (1 - fy) * lat[:-1, :-1] + fx * (1 - fy) * lat[:-1, 1:] + (1 - fx) * fy * lat[1:, :-1] +
                                 fx * fy * lat[1:, 1:])
    return out


def score(recon, exact, bar):
    err = np.abs(np.log1p(recon) - np.log1p(exact)).mean(axis=-1)
    return float((err > bar).mean() * 100.0), float(err.mean()), float(np.percentile(err, 99.9))


def emulate(img, tol, bar):
    """One level (today) against two levels, both with the curvature test at tol; units, lattice rays, and the image's error."""
    f4 = curvature_fails(img, 4, tol)
    th, tw = f4.shape
    f2 = curvature_fails(img, 2, tol)[: 2 * th, : 2 * tw]
    exact = img[: 4 * th, : 4 * tw]
    b4 = bilinear_image(img, 4)[: 4 * th, : 4 * tw]
    b2 = bilinear_image(img, 2)[: 4 * th, : 4 * tw]
    m4 = np.repeat(np.repeat(f4, 4, 0), 4, 1)
    sub = f2 & np.repeat(np.repeat(f4, 2, 0), 2, 1)
    m2 = np.repeat(np.repeat(sub, 2, 0), 2, 1)
    one = np.where(m4[..., None], exact, b4)
    two = np.where(m4[..., None], np.where(m2[..., None], exact, b2), b4)
    pts = np.zeros((2 * th + 1, 2 * tw + 1), bool)
    for oy in range(3):
        for ox in range(3):
            pts[oy : oy + 2 * th : 2, ox : ox + 2 * tw : 2] |= f4
    pts[::2, ::2] = False
    u1 = 16 * int(f4.sum())
    u2 = 3 * int(sub.sum())
    n2 = int(pts.sum())
    s1, s2 = score(one, exact, bar), score(two, exact, bar)
    print(f"curvature test {tol}: one level {u1} units, >{bar} {s1[0]:.3f}% mean {s1[1]:.5f} p99.9 {s1[2]:.4f} | two levels "
          f"{u2} units + {n2} lattice rays ({100.0 * (1 - (u2 + 1.3 * n2) / max(u1, 1)):+.1f}% at 1.3), >{bar} {s2[0]:.3f}% mean "
          f"{s2[1]:.5f} p99.9 {s2[2]:.4f}")


def main():
    path = sys.argv[1]
    bar = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    img = load(path)
    h, w, _ = img.shape
    f4 = fails(img, 4, bar)
    f2 = fails(img, 2, bar)
    th, tw = f4.shape
    f2 = f2[: 2 * th, : 2 * tw]
    tiles = th * tw
    failed = int(f4.sum())
    one_level = 16 * failed
    # Two levels: a failed 4-tile's four sub-tiles; a sub-tile is tested only inside a failed tile.
    sub = f2.reshape(th, 2, tw, 2).transpose(0, 2, 1, 3)  # (th, tw, 2, 2)
    sub_failed = int((sub & f4[:, :, None, None]).sum())
    # New 2-pixel lattice points: every 2-lattice point (even, even) that is not a 4-lattice point and lies on a failed tile, once.
    pts = np.zeros((2 * th + 1, 2 * tw + 1), bool)
    for oy in range(3):
        for ox in range(3):
            pts[oy : oy + 2 * th : 2, ox : ox + 2 * tw : 2] |= f4
    pts[::2, ::2] = False  # already 4-lattice points
    new_points = int(pts.sum())
    two_level_units = 3 * sub_failed
    print(f"frame {w}x{h}, bar {bar}: {tiles} 4-px tiles, {failed} fail ({100.0 * failed / tiles:.2f}%) -> {one_level} units")
    print(f"  two levels: {sub_failed} of {4 * failed} sub-tiles fail ({100.0 * sub_failed / max(4 * failed, 1):.1f}%) -> "
          f"{two_level_units} units + {new_points} new lattice rays = {two_level_units + new_points}")
    for query_cost in (1.0, 1.3):
        cost = two_level_units + query_cost * new_points
        print(f"  lattice ray at {query_cost:.1f} units: {cost:.0f} against {one_level} ({100.0 * (1 - cost / max(one_level, 1)):+.1f}% "
              f"saved)")
    for tol in (0.4, 0.2, 0.1):
        emulate(img, tol, bar)


if __name__ == "__main__":
    main()
