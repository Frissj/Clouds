"""Spectral node kill test (offline, no Mogwai): can a few spectral modes per macro-node reproduce the chord optical depth of the
real bench cloud to the push probe's per-crossing tolerances? Result recorded beside buildPushOperator (MEASURED and REJECTED).

Reads the canonical cache (HSTRCloudCompiler build, format v1) of the bench library's cloud, cuts occupied nodes of 4^3-64^3
voxels at mip levels 0-3, keeps the best M terms of each node in a DCT, a periodic plane-wave (FFT) and a Haar basis, and scores
|T_exact - T_M| on random chords (uniform point, isotropic direction: length-weighted, like ray crossings) at 0.25-voxel steps.
Extinction is density x 1 per world unit (cloud_sea.pyscene), a source voxel is voxelWorld x the sea's fit scale (instance scale 1).

    python scripts/HSTR/bench/spectral_fit.py           modes needed so every chord of a node is within 0.005 / 0.01 / 0.02
    python scripts/HSTR/bench/spectral_fit.py --soft    share of crossings over 0.02 / 0.005 at M = 1-64 terms
HSTR_CLOUD_CACHE overrides the cache file. Needs numpy and scipy; ~2 GB RAM, ~15 min on 8 workers.
"""
import os
import struct
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.fft import dctn, idctn, irfftn, rfftn
from scipy.ndimage import map_coordinates

CACHE = os.environ.get("HSTR_CLOUD_CACHE", "C:/Users/Friss/Downloads/clouds_hr/cache/cloud_cumulus_1_size_1.hstrcloud")
DOMAIN_VOXEL = 320.0 / 64.0  # sea tileWorld / cloudProxyResolution
LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SOFT_LADDER = [1, 4, 8, 16, 32, 64]
TOLS = [0.005, 0.01, 0.02]
CHORDS = 96
WORKERS = 8  # 30 exhausted the page file
NEVER = 10**9


def load():
    """Level-0 density per non-empty 128^3 chunk, as arrays indexed [x, y z]."""
    f = open(CACHE, "rb")
    h = f.read(264)
    chunks = struct.unpack_from("<I", h, 12)[0]
    dims = np.array(struct.unpack_from("<3I", h, 116))
    vw = struct.unpack_from("<f", h, 128)[0]
    f.seek(struct.unpack_from("<Q", h, 256)[0])
    table = f.read(24 * chunks)
    out = {}
    for c in range(chunks):
        _, _, off, comp, raw = struct.unpack_from("<IIQII", table, 24 * c)
        f.seek(off)
        b = zlib.decompress(f.read(comp))
        assert len(b) == raw
        n = struct.unpack_from("<I", b, 0)[0]
        if n == 0:
            continue
        p = 4
        coords = np.frombuffer(b, np.uint32, n, p)
        p += 4 * n
        ranges = np.frombuffer(b, np.float32, 2 * n, p).reshape(n, 2)
        p += 8 * n
        codes = np.cumsum(np.frombuffer(b, np.uint16, 512 * n, p).reshape(n, 512).astype(np.int64), axis=1) % 65536
        values = (ranges[:, :1] + ranges[:, 1:] * (codes / 65535.0)).reshape(n, 8, 8, 8)  # core index x + 8 (y + 8 z)
        bx, by, bz = coords & 1023, (coords >> 10) & 1023, coords >> 20
        vol = np.zeros((128, 128, 128), np.float32)
        for i in range(n):
            ox, oy, oz = (bx[i] % 16) * 8, (by[i] % 16) * 8, (bz[i] % 16) * 8
            vol[ox:ox + 8, oy:oy + 8, oz:oz + 8] = values[i].transpose(2, 1, 0)
        out[(bx[0] // 16, by[0] // 16, bz[0] // 16)] = vol
    return dims, vw, out


def pool(a, k):
    s = a.shape[0] // k
    return a.reshape(s, k, s, k, s, k).mean((1, 3, 5))


def haar_fwd(a):
    a = a.astype(np.float64).copy()
    n = a.shape[0]
    while n > 1:
        for ax in range(3):
            sl = tuple([slice(0, n)] * 3)
            blk = a[sl]
            e, o = np.take(blk, range(0, n, 2), ax), np.take(blk, range(1, n, 2), ax)
            a[sl] = np.concatenate([(e + o) / np.sqrt(2), (e - o) / np.sqrt(2)], ax)
        n //= 2
    return a


def haar_inv(a):
    a = a.copy()
    n = 2
    while n <= a.shape[0]:
        for ax in reversed(range(3)):
            sl = tuple([slice(0, n)] * 3)
            blk = a[sl]
            s, d = np.take(blk, range(0, n // 2), ax), np.take(blk, range(n // 2, n), ax)
            r = np.empty_like(blk)
            idx = [slice(None)] * 3
            idx[ax] = slice(0, n, 2)
            r[tuple(idx)] = (s + d) / np.sqrt(2)
            idx[ax] = slice(1, n, 2)
            r[tuple(idx)] = (s - d) / np.sqrt(2)
            a[sl] = r
        n *= 2
    return a


def best_m(coef, m):
    flat = np.abs(coef).ravel()
    if m >= flat.size:
        return coef
    threshold = np.partition(flat, flat.size - m)[flat.size - m]
    return np.where(np.abs(coef) >= threshold, coef, 0)


def plane_best_m(F, shape, m):
    """Best m entries of the half spectrum: each conjugate pair is one cos(w.x + phi), the proposal's harmonic (box envelope)."""
    mag = np.abs(F).copy()
    mag[..., 1:(shape[0] + 1) // 2] *= np.sqrt(2)
    flat = mag.ravel()
    keep = np.zeros(flat.size, bool)
    keep[np.argpartition(flat, flat.size - m)[flat.size - m:]] = True
    return irfftn(np.where(keep.reshape(F.shape), F, 0), s=shape)


def chords(n, k, r):
    d = r.normal(size=(k, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    p = r.uniform(0, n, size=(k, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(d != 0, -p / d, -np.inf)
        t1 = np.where(d != 0, (n - p) / d, np.inf)
    return p, d, np.max(np.minimum(t0, t1), 1), np.min(np.maximum(t0, t1), 1)


def transmittance(field, rays, voxel_world, step=0.25):
    out = np.empty(len(rays[0]))
    for i, (p, d, a, b) in enumerate(zip(*rays)):
        m = max(1, int(np.ceil((b - a) / step)))
        t = a + (np.arange(m) + 0.5) * (b - a) / m
        s = map_coordinates(field, (p[None] + t[:, None] * d[None] - 0.5).T, order=1, mode="nearest")
        out[i] = np.maximum(s, 0).sum() * (b - a) / m * voxel_world
    return np.exp(-out)


def reconstructions(field):
    dct = dctn(field, norm="ortho")
    haar = haar_fwd(field)
    F = rfftn(field)
    return {
        "dct": lambda m: idctn(best_m(dct, m), norm="ortho"),
        "plane": lambda m: plane_best_m(F, field.shape, m),
        "haar": lambda m: haar_inv(best_m(haar, m)),
    }


def modes_needed(args):
    field, voxel_world, seed = args
    n = field.shape[0]
    rays = chords(n, CHORDS, np.random.default_rng(seed))
    exact = transmittance(field, rays, voxel_world)
    res = {"tau_max": float(-np.log(exact.min()))}
    for name, rebuild in reconstructions(field).items():
        need = {}
        for m in LADDER:
            if m > n**3 // (2 if name == "plane" else 1):
                break
            err = np.abs(transmittance(rebuild(m), rays, voxel_world) - exact).max()
            need.update({tol: m for tol in TOLS if tol not in need and err <= tol})
            if len(need) == len(TOLS):
                break
        res[name] = need
    return res


def crossing_errors(args):
    field, voxel_world, seed = args
    rays = chords(field.shape[0], CHORDS, np.random.default_rng(seed))
    exact = transmittance(field, rays, voxel_world)
    rec = reconstructions(field)
    return {name: [np.abs(transmittance(rec[name](m), rays, voxel_world) - exact) for m in SOFT_LADDER] for name in ("dct", "haar")}


def sample_nodes(chunks, rng, L, n, count=160):
    span = n * 2**L
    keys = list(chunks)
    nodes = []
    while len(nodes) < count:
        v = chunks[keys[rng.integers(len(keys))]]
        o = rng.integers(0, 128 // span, 3) * span
        blk = v[o[0]:o[0] + span, o[1]:o[1] + span, o[2]:o[2] + span]
        if blk.max() > 0:
            nodes.append(pool(blk, 2**L).astype(np.float64))
    return nodes


def percentile(a, p):
    if not len(a):
        return "-"
    v = np.percentile(a, p)
    return "fail" if v >= NEVER else f"{int(v)}"


def main():
    soft = "--soft" in sys.argv
    rng = np.random.default_rng(1)
    dims, vw, chunks = load()
    occupied = np.array(list(chunks))
    extent = (occupied.max(0) + 1 - occupied.min(0)) * 128 * vw
    fit = 0.98 * min(320 / max(extent[0], extent[2]), 160 / extent[1])  # CloudSea's fit scale (single-asset library)
    keys = list(chunks)
    keep = set(rng.choice(len(keys), 60, replace=False))
    chunks = {k: v for i, (k, v) in enumerate(chunks.items()) if i in keep}
    svw = vw * fit
    print(f"dims {dims}, source voxel {svw:.3f} world = {DOMAIN_VOXEL / svw:.1f} a domain voxel; 60 of {len(keys)} chunks sampled")
    shapes = [(1, 8), (2, 8), (3, 4)] if soft else [(L, n) for L in range(4) for n in (4, 8, 16, 32, 64) if n * 2**L <= 128 and n * 2**L * svw <= 4 * DOMAIN_VOXEL]
    for L, n in shapes:
        lw = svw * 2**L
        jobs = [(f, lw, int(rng.integers(1 << 30))) for f in sample_nodes(chunks, rng, L, n, 200 if soft else 160)]
        with ProcessPoolExecutor(WORKERS) as ex:
            rs = list(ex.map(crossing_errors if soft else modes_needed, jobs))
        print(f"L{L} node {n}^3 ({n**3} dof, {n * lw / DOMAIN_VOXEL:.2f} domain voxels on edge)")
        if soft:
            for name in ("dct", "haar"):
                cells = []
                for i, m in enumerate(SOFT_LADDER):
                    e = np.concatenate([r[name][i] for r in rs])
                    cells.append(f"M={m}: {100 * np.mean(e > 0.02):.1f}%/{100 * np.mean(e > 0.005):.1f}%")
                print(f"  {name} over 0.02/0.005  " + "  ".join(cells))
        else:
            dense = np.array([r["tau_max"] for r in rs]) > 0.05
            print(f"  {len(rs)} nodes, {100 * dense.mean():.0f}% with a chord over tau 0.05")
            for name in ("dct", "plane", "haar"):
                cells = []
                for tol in TOLS:
                    need = np.array([r[name].get(tol, NEVER) for r in rs], float)
                    cells.append(f"{tol}: all med {percentile(need, 50)} p90 {percentile(need, 90)}, dense med {percentile(need[dense], 50)} p90 {percentile(need[dense], 90)}")
                print(f"  {name:5s} " + "; ".join(cells))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
