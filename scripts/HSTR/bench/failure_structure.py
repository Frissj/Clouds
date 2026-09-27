"""Failure structure of the snapped temporal cache inside one cell x tile interaction (offline, no Mogwai).

share8 measured, at 0.02: walk 90.8% of snapped crossings right (99.1% for share7's exact midpoint ray) but only 58.3% of the
interactions' chord steps in interactions whose EVERY crossing is right. This asks whether the gap is exploitable:
  1. Structure: do the failing crossings of an interaction cluster (a boundary, a patch) or scatter? Measured as the steps a
     bounded quadtree mask over the interaction keeps (re-march only the blocks holding a failure), against the same number of
     failures placed at random among the same rays.
  2. Detection: can the interaction find its failures WITHOUT marching them, from the old frame's own records? A hazard is the
     disagreement between the snapped texel's record and its neighbours' records towards the new ray's offset - rays the old
     frame marched anyway. (certify2's guards were local derivatives at a quarter voxel: they cannot see the voxel-edge kinks
     inside the half-texel the snap moves. A neighbour one texel away spans that whole interval.)

Setup follows ray_jet_fit.py: the bench cloud at mip 1 (0.445 world a voxel), one beam texel = 1 fine voxel at the cell, the
camera D world units away moving forward (walk 2, sprint 20 a frame). A cell is a world-aligned box of 5 world units (the sea's
domain voxel, 11.2 fine voxels here); the interaction is every new-frame texel ray that marches something in it (a cell's whole
footprint: the probe's interactions are clipped to 16-pixel tiles and hold only the frame's dirty rays, so the all-pass veto
is also reported on random 65-ray subsets, the probe's mean). Each crossing is scored as pushShareLocal does: light and
transmittance through what lies in front of it. The source is ray_jet_fit's smooth stand-in, so E errors are optimistic.

    python scripts/HSTR/bench/failure_structure.py [interactions per case, default 160]
"""
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import map_coordinates

VOLUME = os.path.join(os.environ.get("TEMP", "."), "hstr_jet_level1.npy")  # built by ray_jet_fit.py
CACHE = os.environ.get("HSTR_CLOUD_CACHE", "C:/Users/Friss/Downloads/clouds_hr/cache/cloud_cumulus_1_size_1.hstrcloud")
STEP = 0.25  # fine voxels
CELL_WORLD = 5.0
TOLS = [0.005, 0.01, 0.02]
MOTIONS = [("walk", 2.0), ("sprint", 20.0)]
DISTANCES = [300.0]
PROBE_RAYS = 65  # share1: rays an interaction
WORKERS = 16
_vol = None


def volume():
    global _vol
    if _vol is None:
        _vol = np.load(VOLUME, mmap_mode="r")
    return _vol


def voxel_world():
    h = open(CACHE, "rb").read(264)
    vw = struct.unpack_from("<f", h, 128)[0]
    extent = np.array(volume().shape) * 2 * vw
    return vw * 0.98 * min(320 / max(extent[0], extent[2]), 160 / extent[1]) * 2


def box_chord(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(d != 0, (lo - o) / d, -np.inf)
        t1 = np.where(d != 0, (hi - o) / d, np.inf)
    return max(np.max(np.minimum(t0, t1)), 0.0), np.min(np.maximum(t0, t1))


def march(o, d, a, b, vw):
    """(T, E, occupied steps) of o + t d over [a, b], midpoint rule at STEP from a."""
    if b <= a:
        return 1.0, 0.0, 0
    t = np.arange(a + 0.5 * STEP, b, STEP)
    if len(t) == 0:
        return 1.0, 0.0, 0
    x = o[None] + t[:, None] * d[None]
    sigma = np.maximum(map_coordinates(volume(), (x - 0.5).T, order=1, mode="constant"), 0) * vw
    dt = np.diff(np.append(t - 0.5 * STEP, b))
    T = np.exp(-np.cumsum(sigma * dt))
    Tb = np.concatenate([[1.0], T[:-1]])
    source = 0.3 + 0.7 * np.clip(x[:, 1] / volume().shape[1], 0, 1)
    return float(T[-1]), float(np.sum(source * (Tb - T))), int(np.count_nonzero(sigma > 0))


def local(a, b, before):
    """pushShareLocal on (T, E) pairs: light and transmittance seen through `before`."""
    return max(before * abs(a[0] - b[0]), abs(np.log1p(before * max(a[1], 0)) - np.log1p(before * max(b[1], 0))))


def frame(d):
    a = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def one_interaction(args):
    seed, distance, step_world, vw = args
    r = np.random.default_rng(seed)
    vol = volume()
    n = np.array(vol.shape, float)
    cell = CELL_WORLD / vw
    for _ in range(400):
        lo = np.floor(r.uniform(0, n - cell) / cell) * cell
        hi = lo + cell
        block = vol[int(lo[0]):int(hi[0]) + 1, int(lo[1]):int(hi[1]) + 1, int(lo[2]):int(hi[2]) + 1]
        if block.max() > 0 and (block > 0).mean() < 0.98:  # partly occupied: where structure (and failure) lives
            break
    else:
        return None
    centre = 0.5 * (lo + hi)
    view = r.normal(size=3)
    view[1] = -abs(view[1]) * 0.3
    view /= np.linalg.norm(view)
    # The cell sits somewhere in the frustum, not on its axis: rotate the view off the cell by up to the half field of view.
    k1, k2 = frame(view)
    off = np.tan(np.radians(r.uniform(-30, 30))) * k1 + np.tan(np.radians(r.uniform(-18, 18))) * k2
    new_o = centre - (view + off) / np.linalg.norm(view + off) * distance / vw
    old_o = new_o - view * step_world / vw
    k1, k2 = frame(view)

    def grid(o):
        """Texel lattice of a camera at o: one fine voxel at the cell's distance, on its tangent plane."""
        texel = 1.0 / np.linalg.norm(centre - o)
        def coords(dirn):
            return np.array([np.dot(dirn, k1), np.dot(dirn, k2)]) / np.dot(dirn, view) / texel
        def direction(c):
            v = view + (c[0] * k1 + c[1] * k2) * texel
            return v / np.linalg.norm(v)
        return coords, direction

    new_coords, new_dir = grid(new_o)
    old_coords, old_dir = grid(old_o)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    cc = np.array([new_coords((c - new_o) / np.linalg.norm(c - new_o)) for c in corners])
    old_cache = {}

    def old_record(key):
        if key not in old_cache:
            d = old_dir(np.array(key, float))
            a, b = box_chord(old_o, d, lo, hi)
            old_cache[key] = march(old_o, d, a, b, vw)[:2]
        return old_cache[key]

    rows = []
    for i in range(int(np.floor(cc[:, 0].min())), int(np.ceil(cc[:, 0].max())) + 1):
        for j in range(int(np.floor(cc[:, 1].min())), int(np.ceil(cc[:, 1].max())) + 1):
            d = new_dir(np.array([i + 0.5, j + 0.5]))
            a, b = box_chord(new_o, d, lo, hi)
            if b <= a:
                continue
            T, E, steps = march(new_o, d, a, b, vw)
            if steps == 0:
                continue
            va, vb = box_chord(new_o, d, np.zeros(3), n)
            before = march(new_o, d, va, a, vw)[0] if a > va else 1.0
            now = (T, E)
            mid = new_o + d * 0.5 * (a + b)
            key = (mid - old_o) / np.linalg.norm(mid - old_o)
            oa, ob = box_chord(old_o, key, lo, hi)
            oracle = local(march(old_o, key, oa, ob, vw)[:2], now, before)
            c = old_coords(key) - 0.5  # texel centres at integer + 0.5
            s = np.round(c)
            f = c - s
            sx, sy = (1 if f[0] >= 0 else -1), (1 if f[1] >= 0 else -1)
            base = tuple(s.astype(int))
            nx, ny, nxy = (base[0] + sx, base[1]), (base[0], base[1] + sy), (base[0] + sx, base[1] + sy)
            rec, rx, ry, rxy = old_record(base), old_record(nx), old_record(ny), old_record(nxy)
            nearest = local(rec, now, before)
            ex, ey, exy = local(rec, rx, before), local(rec, ry, before), local(rec, rxy, before)
            hz_lin = abs(f[0]) * ex + abs(f[1]) * ey
            hz_max = max(ex, ey, exy)
            fx, fy = abs(f[0]), abs(f[1])
            w = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
            vals = [rec, rx, ry, rxy]
            bil = (sum(wi * v[0] for wi, v in zip(w, vals)), sum(wi * v[1] for wi, v in zip(w, vals)))
            bilinear = local(bil, now, before)
            spread = max(local(p, q, before) for p in vals for q in vals)
            # The whole 3 x 3 around the snapped texel, both sides: what the new ray's tilt can reach past the offset's own side.
            ring = max(local(rec, old_record((base[0] + u, base[1] + v)), before) for u in (-1, 0, 1) for v in (-1, 0, 1))
            rows.append((i, j, steps, before, nearest, oracle, hz_lin, hz_max, bilinear, spread, ring))
    return np.array(rows, float) if rows else None


def quadtree_kept(ij, steps, fail, block):
    """Steps a mask of block x block texel blocks keeps: every block holding a failure re-marches whole."""
    keys = [tuple(k) for k in np.floor(ij / block).astype(int)]
    bad = {k for k, f in zip(keys, fail) if f}
    return sum(s for k, s in zip(keys, steps) if k not in bad)


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    vw = voxel_world()
    print(f"fine voxel {vw:.3f} world, cell {CELL_WORLD / vw:.1f} fine voxels, texel = 1 fine voxel")
    rng_null = np.random.default_rng(7)
    for distance in DISTANCES:
        for motion, step in MOTIONS:
            rng = np.random.default_rng(int(distance) * 100 + int(step))
            jobs = [(int(rng.integers(1 << 30)), distance, step, vw) for _ in range(count)]
            with ProcessPoolExecutor(WORKERS) as ex:
                its = [x for x in ex.map(one_interaction, jobs) if x is not None]
            allrows = np.concatenate(its)
            steps = allrows[:, 2]
            total = steps.sum()
            print(f"\nD {distance:.0f}, {motion}: {len(its)} interactions, {len(allrows)} crossings "
                  f"({len(allrows) / len(its):.0f} a cell), {steps.mean():.1f} occupied samples a crossing")
            for tol in TOLS:
                fail_n = allrows[:, 4] > tol
                line = [f"crossings right: exact midpoint {100 * steps[allrows[:, 5] <= tol].sum() / total:.1f}%, "
                        f"snapped {100 * steps[~fail_n].sum() / total:.1f}%"]
                # The veto: whole interactions, and random probe-sized subsets of them.
                whole = sum(it[:, 2].sum() for it in its if not (it[:, 4] > tol).any())
                sub_keep = sub_tot = 0.0
                for it in its:
                    for _ in range(4):
                        pick = it[rng_null.permutation(len(it))[:PROBE_RAYS]]
                        sub_tot += pick[:, 2].sum()
                        sub_keep += 0 if (pick[:, 4] > tol).any() else pick[:, 2].sum()
                line.append(f"all-pass whole cell {100 * whole / total:.1f}%, {PROBE_RAYS}-ray subsets {100 * sub_keep / sub_tot:.1f}%")
                print(f"  tol {tol}: " + "; ".join(line))
                # Structure: bounded masks against the same failure count scattered at random.
                cells = []
                for block in (8, 4, 2, 1):
                    kept = null = 0.0
                    for it in its:
                        f = it[:, 4] > tol
                        kept += quadtree_kept(it[:, :2], it[:, 2], f, block)
                        null += np.mean([quadtree_kept(it[:, :2], it[:, 2], rng_null.permutation(f), block) for _ in range(4)])
                    cells.append(f"{block}x{block} {100 * kept / total:.1f}% (random {100 * null / total:.1f}%)")
                print("    mask kept, real (scattered):  " + ", ".join(cells))
                adj = adj_null = nf = 0
                for it in its:
                    f = it[:, 4] > tol
                    pos = {(int(a), int(b)): k for k, (a, b) in enumerate(it[:, :2])}
                    for fl, tag in ((f, 0), (rng_null.permutation(f), 1)):
                        for k in np.nonzero(fl)[0]:
                            a, b = int(it[k, 0]), int(it[k, 1])
                            hit = any(fl[pos[q]] for q in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)) if q in pos)
                            if tag == 0:
                                adj += hit
                                nf += 1
                            else:
                                adj_null += hit
                print(f"    failing crossings with a failing 4-neighbour: {100 * adj / max(nf, 1):.0f}% (random {100 * adj_null / max(nf, 1):.0f}%)")
                # Detection from the old records alone.
                for name, pred, hz in (("nearest, hazard f.dx", 4, 6), ("nearest, hazard max", 4, 7), ("bilinear, hazard spread", 8, 9),
                                       ("nearest, hazard 3x3", 4, 10)):
                    for scale in (1.0, 0.5, 0.25):
                        acc = allrows[:, hz] <= scale * tol
                        wrong = acc & (allrows[:, pred] > tol)
                        worst = allrows[wrong, pred]
                        tail = f", wrong accepts p50 {np.median(worst) / tol:.1f}x tol, max {worst.max() / tol:.1f}x" if wrong.any() else ""
                        print(f"    {name:24s} x{scale:<4g} accepts {100 * steps[acc].sum() / total:.1f}% of steps, wrong "
                              f"{100 * steps[wrong].sum() / total:.3f}%{tail}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
