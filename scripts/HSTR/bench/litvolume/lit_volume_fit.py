"""Lit World Volume gate S0 (offline, no Mogwai): what does de-instancing cost in quality?

LIT_VOLUME.md section 6. A world-aligned lit volume cannot line up with every sea instance (CloudSea::makeTile scales each tile's
cloud by 0.65-1 and offsets it freely), so its bake point-samples each instance's trilinear field at world texel centres and the
march then interpolates those texels again. This measures what that second interpolation does to the two things the image
depends on, against the instanced field the shipping march reads:

  columns   the compiler's own metric (measureTransmittance): axis columns through the crop, |exp(-tau) - exp(-tau')|, p99
  rays      backlit single scatter along oblique rays: L = sum T (1 - exp(-sigma dt)) exp(-tau_sun), the sun depth integrated
            through the instance field at its texels (what bakeCloudSun stores) and, for the world grid, point-sampled from that at
            world texel centres (what the lit bake would store); |dL| against the crop's 90th-percentile ray (what a pixel
            shows), log error of the rays at least 5% of it, and transmittance error, over rays that touch density

Instances are scaled signed permutations, so the resampling is separable: each axis maps instance voxels to world voxels by a
scale and a phase. Arms vary the world voxel W = c * s_max (s_max: the largest instance's voxel at this level):
  point c         world texel = the instance field at its centre (the bake as specified)
  prefilter c     world texels = the least-squares fit of a trilinear field at W to the instance field (per-axis L2 projection)
  SANITY aligned  W = s, phase 0: the world grid IS the instance grid - must score ~0
  SANITY coarse   c = 2: must be clearly worse than c = 1
  anchor          the first arm again, last: must repeat to the digit (same seeds)
Stop rule (stated in LIT_VOLUME.md before any run): close the world-aligned form unless some c >= 0.65 keeps column p99 <=
0.0365 (the packages' own compression p99, IntelSea/manifest.json) AND puts <= 1% of rays over 0.02 in |dL| / L90.

    python lit_volume_fit.py --volume a0_L1.npy --level 1 [--voxel-world 0.25] [--crops 48] [--workers 8]
    python lit_volume_fit.py --synthetic          smoke test of the script itself (numbers mean nothing)

The volume is one asset at one pyramid level as written by hstrlib_dense (float16, shape (z, y, x)). numpy only.
"""
import argparse
import functools
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SUN = np.array([0.30, 0.07, 0.95])  # SunsetCloudSea.py sunDirection (towards the sun)
SUN = SUN / np.linalg.norm(SUN)
TILE_WORLD = 320.0  # CloudSeaDesc
LAYER_HEIGHT = 160.0
MIN_SCALE = 0.65  # kCloudSeaMinScale
DENSITY_SCALE = 1.0  # cloud_sea.pyscene densityScale x HSTRCloud densityScale: extinction per world unit per density unit
CROP = 40  # level voxels of the scored interior
MARGIN = 6  # level voxels around it, so the world grid's texels near the interior see real density
QUAD = 4  # quadrature samples per instance voxel along columns and rays
RAYS = 768
SUN_STEP = 0.5  # instance voxels per sun-depth step

ARMS = [
    ("point c=1.0", 1.0, "point"),
    ("point c=0.8", 0.8, "point"),
    ("point c=0.65", 0.65, "point"),
    ("point c=0.5", 0.5, "point"),
    ("prefilter c=1.0", 1.0, "prefilter"),
    ("prefilter c=0.8", 0.8, "prefilter"),
    ("SANITY aligned", None, "aligned"),
    ("SANITY coarse c=2", 2.0, "point"),
    ("anchor point c=1.0", 1.0, "point"),
]


# --- fields -----------------------------------------------------------------------------------------------------------------------
def trilinear(grid, p):
    """grid (nz, ny, nx); p (..., 3) as (x, y, z) in voxel-index units (centres at integers). Zero outside."""
    nz, ny, nx = grid.shape
    padded = np.zeros((nz + 2, ny + 2, nx + 2), np.float32)
    padded[1:-1, 1:-1, 1:-1] = grid
    q = p + 1.0
    q = np.clip(q, 0.0, np.array([nx + 1, ny + 1, nz + 1], np.float64) - 1e-6)
    i = np.floor(q).astype(np.int64)
    f = (q - i).astype(np.float32)
    i0 = np.minimum(i, np.array([nx, ny, nz]))
    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    c000 = padded[z0, y0, x0]
    c100 = padded[z0, y0, x0 + 1]
    c010 = padded[z0, y0 + 1, x0]
    c110 = padded[z0, y0 + 1, x0 + 1]
    c001 = padded[z0 + 1, y0, x0]
    c101 = padded[z0 + 1, y0, x0 + 1]
    c011 = padded[z0 + 1, y0 + 1, x0]
    c111 = padded[z0 + 1, y0 + 1, x0 + 1]
    c00 = c000 + (c100 - c000) * fx
    c10 = c010 + (c110 - c010) * fx
    c01 = c001 + (c101 - c001) * fx
    c11 = c011 + (c111 - c011) * fx
    c0 = c00 + (c10 - c00) * fy
    c1 = c01 + (c11 - c01) * fy
    return c0 + (c1 - c0) * fz


def hat_projection(n_inst, n_world, ratio, phase, method):
    """1D operator taking instance voxel values (linear spline, centres at 0..n_inst-1) to world texel values (centres at
    phase + k * ratio in instance voxel units, k < n_world). point: sample at the centres. prefilter: L2 projection onto the
    world hat basis."""
    centres = phase + np.arange(n_world) * ratio
    if method == "point":
        m = np.zeros((n_world, n_inst))
        i = np.floor(centres).astype(int)
        f = centres - i
        for k in range(n_world):
            if 0 <= i[k] < n_inst:
                m[k, i[k]] += 1.0 - f[k]
            if 0 <= i[k] + 1 < n_inst:
                m[k, i[k] + 1] += f[k]
        return m
    # Galerkin: G w = B v, G_kl = <h^W_k, h^W_l>, B_kj = <h^W_k, h^s_j>, by quadrature on a fine grid.
    lo, hi = centres[0] - ratio, centres[-1] + ratio
    x = np.linspace(lo, hi, int((hi - lo) * 16) + 1)
    dx = x[1] - x[0]
    hw = np.maximum(0.0, 1.0 - np.abs(x[None, :] - centres[:, None]) / ratio)
    hs = np.maximum(0.0, 1.0 - np.abs(x[None, :] - np.arange(n_inst)[:, None]))
    g = hw @ hw.T * dx
    b = hw @ hs.T * dx
    return np.linalg.solve(g, b)


def resample(field, ratio, phase, method, n_world):
    """Separable world resample of an instance-voxel grid (z, y, x); ratio and phase per axis (x, y, z)."""
    out = field.astype(np.float64)
    for axis_xyz in range(3):
        axis = 2 - axis_xyz  # array axis of x, y, z
        m = hat_projection(field.shape[axis], n_world[axis_xyz], ratio[axis_xyz], phase[axis_xyz], method)
        out = np.moveaxis(np.tensordot(m, np.moveaxis(out, axis, 0), axes=(1, 0)), 0, axis)
    return np.maximum(out, 0.0).astype(np.float32)


def sun_depth(field, sun_local, voxel_world):
    """Sun optical depth at every texel of field, integrated through field towards the sun to the crop's edge."""
    nz, ny, nx = field.shape
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    p = np.stack([xx, yy, zz], -1).astype(np.float64).reshape(-1, 3)
    tau = np.zeros(p.shape[0])
    reach = math.sqrt(nx * nx + ny * ny + nz * nz)
    t = 0.5 * SUN_STEP
    while t < reach:
        tau += trilinear(field, p + t * sun_local) * SUN_STEP
        t += SUN_STEP
    return (tau * voxel_world * DENSITY_SCALE).reshape(field.shape).astype(np.float32)


# --- crops ------------------------------------------------------------------------------------------------------------------------
def pick_crops(volume, count, rng):
    """Interior boxes of CROP^3 with content, stratified: rim (partly occupied, steep), interior (fully occupied), wispy (low mean,
    partly occupied), and any."""
    nz, ny, nx = volume.shape
    side = CROP + 2 * MARGIN
    candidates = []
    for _ in range(count * 60):
        o = [rng.integers(0, max(1, d - side)) for d in (nx, ny, nz)]
        box = volume[o[2]:o[2] + side, o[1]:o[1] + side, o[0]:o[0] + side].astype(np.float32)
        if box.shape != (side, side, side) or box.max() <= 0:
            continue
        occupied = float((box > 1e-4).mean())
        if occupied < 0.02:
            continue
        g = np.abs(np.diff(box, axis=0)).mean() + np.abs(np.diff(box, axis=1)).mean() + np.abs(np.diff(box, axis=2)).mean()
        candidates.append((tuple(o), occupied, float(box.mean()), float(g)))
    if not candidates:
        raise SystemExit("no crop with content")
    occ = np.array([c[1] for c in candidates])
    mean = np.array([c[2] for c in candidates])
    grad = np.array([c[3] for c in candidates])
    classes = {
        "rim": np.argsort(-(grad * (occ < 0.9))),
        "interior": np.argsort(-occ),
        "wispy": np.argsort(mean + 1e9 * (occ > 0.6)),
        "any": rng.permutation(len(candidates)),
    }
    per = max(1, count // len(classes))
    chosen, seen = [], set()
    for name, order in classes.items():
        taken = 0
        for i in order:
            if taken >= per:
                break
            if candidates[i][0] in seen:
                continue
            seen.add(candidates[i][0])
            chosen.append((name, candidates[i][0]))
            taken += 1
    return chosen


# --- scoring ----------------------------------------------------------------------------------------------------------------------
def score(job):
    (crop_name, origin, volume_path, level, s_max_world, trial_seed, arms, synthetic) = job
    volume = synthetic_volume() if synthetic else np.load(volume_path, mmap_mode="r")
    side = CROP + 2 * MARGIN
    o = origin
    inst = np.asarray(volume[o[2]:o[2] + side, o[1]:o[1] + side, o[0]:o[0] + side], np.float32)
    rng = np.random.default_rng(trial_seed)
    scale = MIN_SCALE + (1.0 - MIN_SCALE) * rng.random()
    s_world = s_max_world * scale  # world units per instance voxel at this level
    # The tile's orientation: a quarter turn about y and a mirror (makeTile); in the asset's frame the sun turns the other way.
    turns = int(rng.integers(0, 4))
    mirror = rng.random() < 0.5
    sun = SUN.copy()
    for _ in range(turns):
        sun = np.array([-sun[2], sun[1], sun[0]])
    if mirror:
        sun[0] = -sun[0]
    tau_inst = sun_depth(inst, sun, s_world)
    phase_seed = rng.random(3)
    # Rays: random interior point, direction within 60 degrees of the sun (the backlit sunset view), the chord through the interior.
    dirs = []
    while len(dirs) < RAYS:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        if d @ sun > 0.5:
            dirs.append(d)
    dirs = np.array(dirs)
    starts = MARGIN + rng.random((RAYS, 3)) * (CROP - 1)
    results = {}
    for name, c, method in arms:
        if method == "aligned":
            ratio = np.ones(3)
            phase = np.zeros(3)
            w_method = "point"
        else:
            # World voxel W = c * s_max; in instance voxels ratio = W / s = c / scale. Phase: a random sub-texel world origin.
            ratio = np.full(3, c / scale)
            phase = -phase_seed * ratio
            w_method = method
        n_world = [int(math.ceil((side - 1 - phase[a]) / ratio[a])) + 2 for a in range(3)]
        world = resample(inst, ratio, phase, w_method, n_world)
        tau_world = resample(tau_inst, ratio, phase, "point", n_world)  # the bake reads the instance sun depth at its texels

        def to_world(p):
            return (p - phase) / ratio

        # Columns along each axis through the interior, at instance voxel centres, QUAD samples a voxel.
        col_err = []
        ts = MARGIN - 0.5 + (np.arange(CROP * QUAD) + 0.5) / QUAD
        grid_u, grid_v = np.meshgrid(np.arange(MARGIN, MARGIN + CROP), np.arange(MARGIN, MARGIN + CROP), indexing="ij")
        for axis in range(3):
            pts = np.zeros((CROP, CROP, ts.size, 3))
            others = [a for a in range(3) if a != axis]
            pts[..., axis] = ts[None, None, :]
            pts[..., others[0]] = grid_u[..., None]
            pts[..., others[1]] = grid_v[..., None]
            p = pts.reshape(-1, 3)
            a = trilinear(inst, p).reshape(CROP * CROP, -1).sum(1) / QUAD * s_world * DENSITY_SCALE
            b = trilinear(world, to_world(p)).reshape(CROP * CROP, -1).sum(1) / QUAD * s_world * DENSITY_SCALE
            keep = np.maximum(a, b) > 1e-3  # a prefilter's clamped ringing leaves ~0 tails everywhere; they are not columns
            col_err.append(np.abs(np.exp(-a[keep]) - np.exp(-b[keep])))
        col_err = np.concatenate(col_err) if col_err else np.zeros(0)

        # Rays: midpoint quadrature at 1 / QUAD instance voxels, inside the interior box only.
        lo, hi = MARGIN - 0.5, MARGIN + CROP - 0.5
        n = int(CROP * 1.8 * QUAD)
        dt = 1.0 / QUAD
        tt = (np.arange(-n, n) + 0.5) * dt
        p = starts[:, None, :] + tt[None, :, None] * dirs[:, None, :]
        inside = np.all((p >= lo) & (p <= hi), axis=-1)
        pf = p.reshape(-1, 3)
        sig_a = trilinear(inst, pf).reshape(RAYS, -1) * inside * s_world * DENSITY_SCALE
        sig_b = trilinear(world, to_world(pf)).reshape(RAYS, -1) * inside * s_world * DENSITY_SCALE
        sun_a = trilinear(tau_inst, pf).reshape(RAYS, -1)
        sun_b = trilinear(tau_world, to_world(pf)).reshape(RAYS, -1)

        def light(sig, tau):
            # The camera looks along dir, towards the sun (backlit): front to back is increasing t.
            alpha = 1.0 - np.exp(-sig * dt)
            trans = np.cumprod(np.concatenate([np.ones((RAYS, 1)), 1.0 - alpha[:, :-1]], 1), 1)
            return (trans * alpha * np.exp(-tau)).sum(1), trans[:, -1] * (1.0 - alpha[:, -1])

        la, ta = light(sig_a, sun_a)
        lb, tb = light(sig_b, sun_b)
        # Radiance errors against the crop's bright end (its 90th percentile ray), so a black ray's relative error does not count:
        # abs = |dL| / L90 (what a pixel shows), log = |log(L' / L)| for rays at least 5% of L90.
        l90 = float(np.quantile(la, 0.9)) if np.any(la > 0) else 0.0
        if l90 > 0:
            abs_err = np.abs(lb - la) / l90
            bright = la > 0.05 * l90
            log_err = np.abs(np.log((lb[bright] + 1e-3 * l90) / (la[bright] + 1e-3 * l90)))
        else:
            abs_err = np.zeros(0)
            log_err = np.zeros(0)
        touched = (ta < 0.999) | (tb < 0.999)
        results[name] = dict(col=col_err, abs=abs_err[touched] if abs_err.size else abs_err, log=log_err,
                             terr=np.abs(ta - tb)[touched], memory=(1.0 / ratio[0]) ** 3 if method != "aligned" else 1.0)
    return crop_name, results


@functools.lru_cache(maxsize=1)
def synthetic_volume():
    """A procedural cumulus heap for smoke tests: overlapping balls with a noisy, steep fringe. Deterministic."""
    rng = np.random.default_rng(7)
    n = 96
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    field = np.zeros((n, n, n), np.float32)
    for _ in range(14):
        c = rng.uniform(20, 76, 3)
        r = rng.uniform(8, 18)
        d = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / r
        field = np.maximum(field, np.clip(1.6 - 1.6 * d, 0, 1).astype(np.float32))
    noise = np.zeros_like(field)
    p = np.stack([x, y, z], -1).reshape(-1, 3).astype(np.float64)
    for octave in range(4):
        cells = 6 * 2 ** octave
        coarse = rng.random((cells + 1,) * 3).astype(np.float32)
        noise += (trilinear(coarse, p * cells / n).reshape(n, n, n) - 0.5) / 2 ** octave
    return np.clip(field * 3.0 + noise * 1.5 - 1.2, 0, None).astype(np.float16)


def summary(name, rows):
    cat = lambda k: np.concatenate([r[k] for r in rows])
    col, log, ab, terr = cat("col"), cat("log"), cat("abs"), cat("terr")
    mem = np.mean([r["memory"] for r in rows])
    q = lambda a, p: float(np.quantile(a, p)) if a.size else 0.0
    m = lambda a: float(a.mean()) if a.size else 0.0
    over = lambda a, t: 100.0 * float((a > t).mean()) if a.size else 0.0
    mx = lambda a: float(a.max()) if a.size else 0.0
    return (f"{name:20s} mem x{mem:5.2f} | columns n {col.size:6d} mean {m(col):.5f} p99 {q(col, 0.99):.4f} >0.0365 {over(col, 0.0365):5.2f}% "
            f">0.009 {over(col, 0.009):5.2f}% max {mx(col):.3f} | rays n {ab.size:5d} |dL|/L90 mean {m(ab):.4f} >0.02 {over(ab, 0.02):5.2f}% "
            f"p99 {q(ab, 0.99):.4f} max {mx(ab):.3f} | log (bright) mean {m(log):.4f} >0.02 {over(log, 0.02):5.2f}% p99 {q(log, 0.99):.4f} "
            f"| |dT| >0.02 {over(terr, 0.02):5.2f}% p99 {q(terr, 0.99):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--voxel-world", type=float, default=0.25, help="the asset's source voxel in VDB units (manifest voxel_world)")
    ap.add_argument("--fit-scale", type=float, default=0.0, help="CloudSea fit scale; 0: from the volume's content extent")
    ap.add_argument("--crops", type=int, default=48)
    ap.add_argument("--trials", type=int, default=2, help="instance transforms per crop")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    global CROP, RAYS
    if args.synthetic:
        CROP, RAYS = 24, 192
        volume = synthetic_volume()
        args.crops = min(args.crops, 8)
    else:
        volume = np.load(args.volume, mmap_mode="r")
    nz, ny, nx = volume.shape
    if args.fit_scale <= 0:
        occupied = np.argwhere(np.asarray(volume[::4, ::4, ::4]) > 0)
        ext = (occupied.max(0) - occupied.min(0) + 1)[::-1] * 4 * (2 ** args.level) * args.voxel_world  # (x, y, z) VDB units
        fit = 0.98 * min(TILE_WORLD / max(ext[0], ext[2]), LAYER_HEIGHT / ext[1])
    else:
        fit = args.fit_scale
    s_max_world = args.voxel_world * fit * 2 ** args.level
    print(f"volume {nx} x {ny} x {nz} level {args.level}; fit scale {fit:.4f}; largest instance voxel at this level {s_max_world:.4f} "
          f"world units; scale {MIN_SCALE}-1; crops {args.crops} x {args.trials} transforms; {RAYS} rays and {3 * CROP * CROP} "
          f"columns a crop", flush=True)
    rng = np.random.default_rng(args.seed)
    crops = pick_crops(volume, args.crops, rng)
    print("crops by class:", {k: sum(1 for c in crops if c[0] == k) for k in ("rim", "interior", "wispy", "any")}, flush=True)
    jobs = [(name, o, args.volume, args.level, s_max_world, args.seed * 100003 + 7919 * i + t, ARMS, args.synthetic)
            for i, (name, o) in enumerate(crops) for t in range(args.trials)]
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        out = list(pool.map(score, jobs))
    print(f"scored {len(jobs)} crop transforms in {time.time() - start:.0f} s")
    for cls in ("all", "rim", "interior", "wispy", "any"):
        print(f"--- {cls}")
        for name, _, _ in ARMS:
            rows = [r[name] for crop, r in out if cls == "all" or crop == cls]
            if rows:
                print(summary(name, rows))


if __name__ == "__main__":
    main()
