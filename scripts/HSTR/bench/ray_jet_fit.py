"""Ray jet kill test (offline, no Mogwai): can a cache of whole-ray transfer plus its 4D ray-space derivatives predict the next
frame's rays well enough to skip 90% of the sprint's dirty march steps?

The bench cloud's true density (canonical cache, see spectral_fit.py) at mip level 1 is the fine field; a camera D world units
from the cloud flies forward (walk 2, sprint 20 units a frame, the harness motions). The world-fixed beam image keys rays by
direction from the camera, one result per texel; a texel is taken as 1 fine voxel at the cloud (the push probe's measurement:
share8's snap is a median of under 0.5 fine voxels). For each new ray the predictors are:
  oracle    the old camera's exact ray through the new ray's opacity-weighted midpoint (share7: a dense world-keyed cache)
  nearest   that ray snapped to its old texel centre (share8: one result per texel)
  bilinear  the 4 texels around it
  jacobian  the nearest texel's ray + J dr, dr = (transverse offset at the midpoint plane, direction) in 4D, J by central
            differences of the exact transfer (what forward-mode autodiff in the march would output)
  hessian   + 0.5 dr' H dr, H by finite differences
The jets predict (tau, tau * mean source) - the Lie-algebra log of the affine transfer [[T, E], [0, 1]] - and T, E directly;
the better of the two is reported. The source is a smooth stand-in (0.3 + 0.7 height), so E errors are optimistic; T is the
real test. Reuse is weighted by march steps (occupied samples before T < 1e-3), as the push probe's share7/8 were.

    python scripts/HSTR/bench/ray_jet_fit.py [rays per case, default 600]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(__file__))
import spectral_fit

VOLUME = os.path.join(os.environ.get("TEMP", "."), "hstr_jet_level1.npy")
LEVEL = 1
STEP = 0.25  # fine voxels
TOLS = [0.005, 0.01, 0.02]
MOTIONS = [("walk", 2.0), ("sprint", 20.0)]
DISTANCES = [300.0, 800.0]
HALF_FOV = np.radians(37.0)  # 4K 16:9 at the pyscene's 28 mm: +-37 deg across, +-23 deg up
WORKERS = 8
_vol = None


def volume():
    global _vol
    if _vol is None:
        _vol = np.load(VOLUME, mmap_mode="r")
    return _vol


def build_volume():
    dims, vw, chunks = spectral_fit.load()
    k = 2**LEVEL
    lo = np.array(list(chunks)).min(0)
    hi = np.array(list(chunks)).max(0) + 1
    vol = np.zeros(tuple((hi - lo) * 128 // k), np.float32)
    for key, v in chunks.items():
        o = (np.array(key) - lo) * 128 // k
        vol[o[0]:o[0] + 128 // k, o[1]:o[1] + 128 // k, o[2]:o[2] + 128 // k] = spectral_fit.pool(v, k)
    extent = (hi - lo) * 128 * vw
    fit = 0.98 * min(320 / max(extent[0], extent[2]), 160 / extent[1])
    np.save(VOLUME, vol)
    return vw * fit * k


def transfer(o, d, voxel_world):
    """Exact (T, E, steps) of the ray o + t d (fine-voxel units, d unit) through the whole volume."""
    vol = volume()
    n = np.array(vol.shape, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(d != 0, -o / d, -np.inf)
        t1 = np.where(d != 0, (n - o) / d, np.inf)
    a = max(np.max(np.minimum(t0, t1)), 0.0)
    b = np.min(np.maximum(t0, t1))
    if b <= a:
        return 1.0, 0.0, 0
    t = np.arange(a + 0.5 * STEP, b, STEP)
    x = o[None] + t[:, None] * d[None]
    sigma = np.maximum(map_coordinates(vol, (x - 0.5).T, order=1, mode="constant"), 0) * voxel_world
    tau = np.cumsum(sigma) * STEP
    T = np.exp(-tau)
    Tbefore = np.concatenate([[1.0], T[:-1]])
    source = 0.3 + 0.7 * np.clip(x[:, 1] / n[1], 0, 1)
    E = float(np.sum(source * (Tbefore - T)))
    steps = int(np.count_nonzero((sigma > 0) & (Tbefore > 1e-3)))
    return float(T[-1]), E, steps


def midpoint(o, d, voxel_world):
    """Opacity-weighted point along the ray: the world key of the share7 oracle."""
    vol = volume()
    t = np.arange(0.5 * STEP, 4 * max(vol.shape), STEP)
    x = o[None] + t[:, None] * d[None]
    inside = np.all((x >= 0) & (x < np.array(vol.shape)), 1)
    if not inside.any():
        return None
    x, t = x[inside], t[inside]
    sigma = np.maximum(map_coordinates(vol, (x - 0.5).T, order=1, mode="constant"), 0) * voxel_world
    T = np.exp(-np.cumsum(sigma) * STEP)
    w = np.concatenate([[1.0], T[:-1]]) - T
    return None if w.sum() < 1e-4 else (x * w[:, None]).sum(0) / w.sum()


def frame(d):
    """Two unit vectors orthogonal to d."""
    a = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def ray_from(params, base_o, base_d, pivot, span):
    """Ray at 4D offset params = (u, v, p, q) from the base ray: transverse offset at the pivot plane, direction tilt (radians)."""
    e1, e2 = frame(base_d)
    d = base_d + params[2] * e1 + params[3] * e2
    d /= np.linalg.norm(d)
    through = pivot + params[0] * e1 + params[1] * e2
    return through - span * d, d


def ray_params(o, d, base_d, pivot):
    """4D coordinates of the ray (o, d) about the base ray's pivot plane."""
    e1, e2 = frame(base_d)
    t = np.dot(pivot - o, base_d) / np.dot(d, base_d)
    x = o + t * d - pivot
    return np.array([np.dot(x, e1), np.dot(x, e2), np.dot(d, e1) / np.dot(d, base_d), np.dot(d, e2) / np.dot(d, base_d)])


def jets(f0, base_o, base_d, pivot, span, h, voxel_world):
    """Central-difference gradient and Hessian of g = (T, E, tau, tau * mean source) in the 4 ray coordinates."""
    def g(params):
        T, E, _ = transfer(*ray_from(params, base_o, base_d, pivot, span), voxel_world)
        tau = -np.log(max(T, 1e-12))
        return np.array([T, E, tau, tau * E / max(1 - T, 1e-9)])
    g0 = f0
    grad = np.zeros((4, 4))
    hess = np.zeros((4, 4, 4))
    plus, minus = {}, {}
    for i in range(4):
        e = np.zeros(4)
        e[i] = h[i]
        plus[i], minus[i] = g(e), g(-e)
        grad[:, i] = (plus[i] - minus[i]) / (2 * h[i])
        hess[:, i, i] = (plus[i] - 2 * g0 + minus[i]) / h[i] ** 2
    for i in range(4):
        for j in range(i + 1, 4):
            e = np.zeros(4)
            e[i], e[j] = h[i], h[j]
            pp, mm = g(e), g(-e)
            # f(+i+j) + f(-i-j) = 2 f0 + h_i^2 H_ii + h_j^2 H_jj + 2 h_i h_j H_ij
            hess[:, i, j] = hess[:, j, i] = (pp + mm - 2 * g0 - h[i] ** 2 * hess[:, i, i] - h[j] ** 2 * hess[:, j, j]) / (2 * h[i] * h[j])
    return grad, hess


def to_te(g, logspace):
    if not logspace:
        return float(np.clip(g[0], 0, 1)), float(g[1])
    tau = max(g[2], 0.0)
    T = np.exp(-tau)
    return float(T), float(g[3] * (1 - T) / max(tau, 1e-9)) if tau > 1e-9 else 0.0


def one_ray(args):
    seed, distance, step_world, voxel_world = args
    r = np.random.default_rng(seed)
    vol = volume()
    n = np.array(vol.shape, float)
    centre = n / 2
    scale = 1.0 / voxel_world  # world -> fine voxels
    for _ in range(200):
        # A camera 'distance' world units from the cloud centre, looking roughly at it, a ray somewhere in the 4K frustum.
        view = r.normal(size=3)
        view[1] = -abs(view[1]) * 0.3
        view /= np.linalg.norm(view)
        camera = centre - view * distance * scale
        yaw, pitch = r.uniform(-HALF_FOV, HALF_FOV), r.uniform(-0.62 * HALF_FOV, 0.62 * HALF_FOV)
        e1, e2 = frame(view)
        d = view + np.tan(yaw) * e1 + np.tan(pitch) * e2
        d /= np.linalg.norm(d)
        new_o = camera + view * step_world * scale  # forward motion
        mid = midpoint(new_o, d, voxel_world)
        if mid is None:
            continue
        T, E, steps = transfer(new_o, d, voxel_world)
        if steps == 0:
            continue
        break
    else:
        return None
    out = {"steps": steps}
    # The old frame's rays come from the old camera; texel spacing = 1 fine voxel at the midpoint's distance.
    old_o = camera
    dist = np.linalg.norm(mid - old_o)
    texel = 1.0 / dist
    key = (mid - old_o) / dist
    k1, k2 = frame(view)  # texel grid on the old camera's tangent plane (a local stand-in for the octahedral map)
    coords = np.array([np.dot(key, k1), np.dot(key, k2)]) / np.dot(key, view) / texel
    base = np.floor(coords)
    frac = coords - base

    def texel_dir(c):
        v = view + (c[0] * k1 + c[1] * k2) * texel
        return v / np.linalg.norm(v)

    def err(Tp, Ep):
        return max(abs(Tp - T), abs(Ep - E))

    out["oracle"] = err(*transfer(old_o, key, voxel_world)[:2])
    nearest = texel_dir(np.round(coords))
    Tn, En, _ = transfer(old_o, nearest, voxel_world)
    out["nearest"] = err(Tn, En)
    corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
    vals = [transfer(old_o, texel_dir(base + np.array(c)), voxel_world)[:2] for c in corners]
    w = [(1 - frac[0]) * (1 - frac[1]), frac[0] * (1 - frac[1]), (1 - frac[0]) * frac[1], frac[0] * frac[1]]
    out["bilinear"] = err(sum(wi * v[0] for wi, v in zip(w, vals)), sum(wi * v[1] for wi, v in zip(w, vals)))
    # Jets about the nearest texel's ray, pivoting on the point of it nearest the midpoint.
    pivot = old_o + np.dot(mid - old_o, nearest) * nearest
    span = np.dot(pivot - old_o, nearest)
    tauN = -np.log(max(Tn, 1e-12))
    g0 = np.array([Tn, En, tauN, tauN * En / max(1 - Tn, 1e-9)])
    h = np.array([0.25, 0.25, 0.25 / span, 0.25 / span])
    grad, hess = jets(g0, old_o, nearest, pivot, span, h, voxel_world)
    dr = ray_params(new_o, d, nearest, pivot)
    out["snap_voxels"] = float(np.linalg.norm(dr[:2]))
    out["tilt_voxels"] = float(np.linalg.norm(dr[2:]) * 0.5 * steps * STEP)  # rough: tilt x half the occupied depth
    for name, order in (("jacobian", 1), ("hessian", 2)):
        pred = g0 + grad @ dr + (0.5 * np.einsum("kij,i,j->k", hess, dr, dr) if order == 2 else 0)
        out[name] = min(err(*to_te(pred, False)), err(*to_te(pred, True)))
        out[name + "_lin"] = err(*to_te(pred, False))
        out[name + "_log"] = err(*to_te(pred, True))
    return out


def main():
    rays = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    motions = [m for m in MOTIONS if len(sys.argv) < 3 or m[0] in sys.argv[2:]]
    voxel_world = build_volume()
    print(f"level {LEVEL} fine voxel {voxel_world:.3f} world, volume {np.load(VOLUME, mmap_mode='r').shape}")
    names = ["oracle", "nearest", "bilinear", "jacobian", "hessian", "jacobian_lin", "jacobian_log", "hessian_lin", "hessian_log"]
    for distance in DISTANCES:
        for motion, step in motions:
            rng = np.random.default_rng(int(distance) * 100 + int(step))
            jobs = [(int(rng.integers(1 << 30)), distance, step, voxel_world) for _ in range(rays)]
            with ProcessPoolExecutor(WORKERS) as ex:
                rs = [x for x in ex.map(one_ray, jobs) if x is not None]
            steps = np.array([x["steps"] for x in rs], float)
            snap = np.array([x["snap_voxels"] for x in rs])
            print(f"\nD {distance:.0f}, {motion} ({step:g} units/frame): {len(rs)} rays, snap median {np.median(snap):.2f} fine voxels "
                  f"(p90 {np.percentile(snap, 90):.2f}), tilt x half depth median {np.median([x['tilt_voxels'] for x in rs]):.2f}")
            for name in names:
                e = np.array([x[name] for x in rs])
                cells = " / ".join(f"{100 * steps[e <= tol].sum() / steps.sum():.0f}%" for tol in TOLS)
                print(f"  {name:13s} steps reusable at {'/'.join(map(str, TOLS))}: {cells};  rays over 0.02 {100 * np.mean(e > 0.02):.1f}%")
            # Random frustum rays are mostly easy (nearest alone keeps ~92-96% of their steps), unlike the shipping frame's dirty
            # rays (share8: 58% at 0.02). What transfers is the share of the steps nearest loses that a predictor wins back.
            nearest = np.array([x["nearest"] for x in rs])
            for tol in TOLS:
                lost = nearest > tol
                cells = ", ".join(f"{name} {100 * steps[lost & (np.array([x[name] for x in rs]) <= tol)].sum() / max(steps[lost].sum(), 1):.0f}%"
                                  for name in ("oracle", "bilinear", "jacobian", "hessian"))
                print(f"  of the {lost.sum()} rays nearest loses at {tol}, steps recovered: {cells}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
