"""Compile-time brick error bounds, offline (no Mogwai): how much fine work would per-brick error metrics baked by the compiler
save on the real bench cloud?

Two candidate baked quantities, both scored as chord optical-depth error against the level-0 field at 1/64-chord steps (random
isotropic chords through each level-0 8^3 brick: length-weighted, like ray crossings):
  - LOD error: replace the brick by its ancestor at level 1-4 (box-pooled, trilinear). A level-l brick replaces all its children,
    empty ones included (the coarse field bleeds into them), so its error is the worst over every child brick. The cut takes the
    coarsest level whose error, and every finer ancestor's, is within tolerance.
  - Step bound: midpoint rule at step h = 1, 2, 4, 8 level-0 voxels with one global phase (random per chord), the largest step
    within tolerance.
Shares are of non-empty level-0 bricks, by count and weighted by visibility from outside (the best of the six axis transmittances
from the volume boundary, level-1 volume of ray_jet_fit.py): hidden interior bricks are smooth and would flatter the count.
A ray sums the errors of the bricks it crosses (~2-4 fine visits a dirty ray), so a brick tolerance is not a pixel tolerance.

    python scripts/HSTR/bench/brick_lod_fit.py
Needs the canonical cache (spectral_fit.CACHE) and %TEMP%/hstr_jet_level1.npy (ray_jet_fit.py builds it). ~8 workers, ~2 GB.

MEASURED (brick_lod_fit.log, bench cloud, 374k non-empty level-0 bricks, visibility-weighted 39% of them): the surface is dense
and high-frequency - a brick chord's tau is 2.3 median, and a 1-voxel midpoint step alone misses 0.005-0.02 on 84-88% of bricks
(visible median 0.16 worst chord). At a fixed 0.005 / 0.01 / 0.02 tolerance 83-90% of VISIBLE bricks must stay level 0 (fine
visits x0.91-0.94); the coarse share is hidden interior. The step bound lets 22-28% of visible bricks take 8 voxels - the near-empty
ones - and every other brick keeps step 1 (samples x0.71-0.78). Against each brick's own march error instead (errors.npz): no worse
than its step-1 error, 94% of visible bricks stay level 0 (visits x0.97); than step 2, x0.85; than step 4, x0.68. The level-1
brick costs about what a one-voxel step already costs (median 0.12 vs 0.16), so baked error metrics buy <=1.5x on visits even at
the loosest baseline: REJECTED as a route to the frame target. Fewer samples inside a brick were free on the direct path anyway
(direct_path_rate.py).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import map_coordinates

import spectral_fit

VOLUME = os.path.join(os.environ.get("TEMP", "."), "hstr_jet_level1.npy")
OUT = "C:/Users/Friss/Documents/HSTR_results/brick_lod_fit"
LEVELS = [1, 2, 4, 8, 16]  # pooling factor: level 0-4
STEPS = [1.0, 2.0, 4.0, 8.0]
TOLS = [0.005, 0.01, 0.02]
CHORDS = 16
SAMPLES = 64
WORKERS = 8


def chunk_errors(args):
    key, vol, voxel_world, seed = args
    r = np.random.default_rng(seed)
    nb = 16 * 16 * 16
    origin = np.stack(np.meshgrid(*[np.arange(16)] * 3, indexing="ij"), -1).reshape(nb, 1, 3) * 8.0
    d = r.normal(size=(nb, CHORDS, 3))
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    p = r.uniform(0, 8, size=(nb, CHORDS, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(d != 0, -p / d, -np.inf)
        t1 = np.where(d != 0, (8 - p) / d, np.inf)
    a = np.max(np.minimum(t0, t1), 2)
    b = np.min(np.maximum(t0, t1), 2)
    length = b - a
    x0 = origin + p + a[..., None] * d  # chord entry, chunk voxel coordinates

    def tau(field, k, t, w):
        """Sum of w * sigma at the chord parameters t (nb, CHORDS, m) on field pooled by k."""
        x = x0[:, :, None] + t[..., None] * d[:, :, None]
        s = map_coordinates(field, (x.reshape(-1, 3) / k - 0.5).T, order=1, mode="nearest").reshape(t.shape)
        return (np.maximum(s, 0) * w).sum(2) * voxel_world

    t = (np.arange(SAMPLES) + 0.5)[None, None] / SAMPLES * length[..., None]
    w = np.broadcast_to(length[..., None] / SAMPLES, t.shape)
    ref = tau(vol, 1, t, w)
    lod = np.stack([np.abs(tau(spectral_fit.pool(vol, k), k, t, w) - ref).max(1) for k in LEVELS[1:]], 1)
    step = []
    for h in STEPS:
        m = int(np.ceil(8 * np.sqrt(3) / h)) + 1
        phase = r.uniform(0, h, size=(nb, CHORDS, 1))
        ts = phase + h * np.arange(m)[None, None]
        inside = ts < length[..., None]
        step.append(np.abs(tau(vol, 1, np.where(inside, ts, 0), np.where(inside, h, 0.0)) - ref).max(1))
    occupied = vol.reshape(16, 8, 16, 8, 16, 8).max((1, 3, 5)).reshape(nb) > 0
    return key, occupied, lod, np.stack(step, 1), ref.max(1)


def visibility(voxel_world):
    """Per level-0 brick: best axis transmittance from the volume boundary (level-1 volume, 4^3 voxels a brick)."""
    vol = np.load(VOLUME).astype(np.float32) * (2 * voxel_world)
    best = np.zeros(vol.shape, np.float32)
    for ax in range(3):
        c = np.cumsum(vol, ax)
        best = np.maximum(best, np.exp(-(c - vol)))
        best = np.maximum(best, np.exp(-(np.take(c, [-1], ax) - c)))
        del c
    s = best.shape
    return best.reshape(s[0] // 4, 4, s[1] // 4, 4, s[2] // 4, 4).max((1, 3, 5))


def main():
    dims, vw, chunks = spectral_fit.load()
    lo = np.array(list(chunks)).min(0)
    hi = np.array(list(chunks)).max(0) + 1
    extent = (hi - lo) * 128 * vw
    voxel_world = vw * 0.98 * min(320 / max(extent[0], extent[2]), 160 / extent[1])  # ray_jet_fit.build_volume's fit
    vis = visibility(voxel_world)
    print(f"{len(chunks)} chunks, level-0 voxel {voxel_world:.4f} world; brick visibility grid {vis.shape}", flush=True)
    keys = list(chunks)
    jobs = [(k, chunks[k], voxel_world, i) for i, k in enumerate(keys)]
    occ, lod, step, refmax, weight, gid = [], [], [], [], [], []
    with ProcessPoolExecutor(WORKERS) as ex:
        for n, (key, o, l, s, rm) in enumerate(ex.map(chunk_errors, jobs, chunksize=2)):
            base = (np.array(key) - lo) * 16
            idx = base[None] + np.stack(np.meshgrid(*[np.arange(16)] * 3, indexing="ij"), -1).reshape(-1, 3)
            occ.append(o), lod.append(l), step.append(s), refmax.append(rm)
            weight.append(vis[idx[:, 0], idx[:, 1], idx[:, 2]])
            gid.append(idx)
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} chunks", flush=True)
    occ, lod, step, refmax, weight, gid = map(np.concatenate, (occ, lod, step, refmax, weight, gid))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/errors.npz", occ=occ, lod=lod, step=step, refmax=refmax, weight=weight, gid=gid)

    # A level-l brick's error is the worst over its children (empty children included): group level-0 bricks by gid >> l.
    coarse = np.zeros_like(lod)
    for j, k in enumerate(LEVELS[1:]):
        g = gid // k
        _, inv = np.unique(g[:, 0] * 10**8 + g[:, 1] * 10**4 + g[:, 2], return_inverse=True)
        worst = np.zeros(inv.max() + 1)
        np.maximum.at(worst, inv, lod[:, j])
        coarse[:, j] = worst[inv]
    w = weight[occ]
    n = occ.sum()
    print(f"\n{n} non-empty level-0 bricks, visibility-weighted count {w.sum():.0f} ({w.sum() / n:.0%}); "
          f"brick chord tau: median {np.median(refmax[occ]):.3f}, p90 {np.percentile(refmax[occ], 90):.3f}")
    for tol in TOLS:
        ok = np.cumprod(coarse[occ] <= tol, 1)  # every finer ancestor within tolerance too
        level = ok.sum(1)
        bricks = (1.0 / 8.0 ** level).sum()  # cut bricks per level-0 brick they replace
        visits = (1.0 / 2.0 ** level)  # a ray crosses 2^l fewer bricks per unit length
        line = ", ".join(f"L{l} {np.mean(level == l):5.1%}/{np.average(level == l, weights=w):5.1%}" for l in range(5))
        print(f"tol {tol}: cut level share (count/visible) {line}; cut bricks {bricks / n:.1%} of level 0, "
              f"fine visits x{visits.mean():.2f} (visible x{np.average(visits, weights=w):.2f})")
        hmax = np.array(STEPS)[np.maximum((np.cumprod(step[occ] <= tol, 1)).sum(1) - 1, 0)]
        fail1 = step[occ][:, 0] > tol
        line = ", ".join(f"h{h:.0f} {np.mean(hmax == h):5.1%}/{np.average(hmax == h, weights=w):5.1%}" for h in STEPS)
        print(f"         step bound (count/visible) {line}; step 1 itself over tol {fail1.mean():.1%}; "
              f"samples vs step 1 x{(1 / hmax).mean():.2f} (visible x{np.average(1 / hmax, weights=w):.2f})")


if __name__ == "__main__":
    sys.exit(main())
