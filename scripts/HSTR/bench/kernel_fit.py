"""How many analytic kernels a level-0 brick needs at the codec's quality, offline (numpy, no Mogwai).

Question ("Don't Splat Your Gaussians" as HSTR leaves): can a handful of Gaussian or Epanechnikov kernels per 8^3 brick reproduce
the cloud at the package codec's close-up transmittance error (p99 ~0.009 per asset, the compiler's metric)? Stop rule: 1-2 a brick
investigate, 4 interesting, 8 irrelevant, >= 16 closed.

Reads the compiler's canonical cache (<cloud>.hstrcloud: zlib blobs, HSTRCloudCompiler.cpp packCores), every STRIDE-th chunk
(128^3 voxels), and fits K axis-aligned kernels to every brick's 512 voxel values by least squares (Adam, analytic gradients,
weights >= 0, centres and per-axis widths free). Scored exactly as measureTransmittance: per chunk, every voxel column on each
axis, |e^-tau_true - e^-tau_fit| attenuated by the proxy's optical depth in front of the chunk (densityScale 1, the packages'
default). Anchors on the same columns: the atlas's own 8-bit sqrt coding of the truth (what the renderer stores), each brick's
level-1 means (one octave coarser), and zero (sanity: must be far worse). Axis-aligned kernels are a handicap against rotated
ones; per-brick fits cannot share a kernel across bricks.

    python scripts/HSTR/bench/kernel_fit.py [cache file] [chunk stride, default 7] [K list, default 1,2,4,8,16] [max bricks]

MEASURED (intelCloudLib_dense.0.L half, p99 / p99.9 column error; codec package p99 ~0.009):
  kernelfit2 (9 chunks, 2,991 bricks): zero 0.996, atlas 8-bit 0.0005, level-1 means 0.100 / 0.197; Gaussian K1 0.123 / 0.238,
    K2 0.077 / 0.158, K4 0.048 / 0.101, K8 0.031 / 0.065 (K16 / 32 stopped: the CPU fit grows with K).
  kernelfit3 (6 chunks, 975 bricks): level-1 0.091; Gaussian K8 0.029 / 0.055, K16 0.023 / 0.049; Epanechnikov K8 0.038 / 0.080,
    K16 0.027 / 0.055.
  Doubling the kernels takes ~20-35% off the error; 16 a brick are still 2.6x the codec's p99, and four kernels are only twice as
  good as dropping an octave. Stop rule (>= 16: closed) - analytic kernels as HSTR leaves are closed on this content. The fit is
  a local optimum (400 Adam steps, axis-aligned): a better fitter (rotations, more steps) would move the curve by an unmeasured
  amount; 16 kernels would need ~2.6x less error from it to meet the codec.
"""
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "IntelSea/cache_half/intelCloudLib_dense.0.L.hstrcloud"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 7
KS = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "1,2,4,8,16").split(",")]
MAX_BRICKS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000  # numpy fits: ~a minute per 512 bricks at K 16.
N = 128  # Chunk voxels.


def blob(f, offset, compressed, raw):
    f.seek(offset)
    data = zlib.decompress(f.read(compressed))
    assert len(data) == raw
    return data


def read_cache(path):
    with open(path, "rb") as f:
        h = f.read(200)
        magic, version, chunk_count = struct.unpack_from("<QII", h, 0)
        assert magic == 0x314E414352545348 and version == 2, "not a canonical cache"
        dims = struct.unpack_from("<3I", h, 116)
        voxel_world, top_level, proxy_level = struct.unpack_from("<fII", h, 128)
        proxy_dims = struct.unpack_from("<3I", h, 140)
        proxy_ref = struct.unpack_from("<QII", h, 160)
        table = struct.unpack_from("<Q", h, 192)[0]
        proxy = np.frombuffer(blob(f, *proxy_ref), dtype=np.float16)
        f.seek(table)
        chunks = [struct.unpack("<IIQII", f.read(24)) for _ in range(chunk_count)]
        sampled = []
        total = 0
        for i, (chunk, bricks, offset, compressed, raw) in enumerate(chunks):
            # Whole chunks only (a column's error needs every brick on it), every STRIDE-th, up to MAX_BRICKS in all.
            if i % STRIDE or bricks == 0 or total + bricks > MAX_BRICKS:
                continue
            total += bricks
            data = blob(f, offset, compressed, raw)
            count = struct.unpack_from("<I", data, 0)[0]
            coords = np.frombuffer(data, dtype=np.uint32, count=count, offset=4)
            ranges = np.frombuffer(data, dtype=np.float32, count=2 * count, offset=4 + 4 * count).reshape(count, 2)
            codes = np.frombuffer(data, dtype=np.uint16, count=512 * count, offset=4 + 12 * count).reshape(count, 512)
            codes = np.cumsum(codes.astype(np.uint32), axis=1) & 0xFFFF  # Delta-coded, uint16 wrap.
            cores = ranges[:, :1] + ranges[:, 1:] * (codes.astype(np.float32) / 65535.0)
            sampled.append((chunk, coords, cores.reshape(count, 8, 8, 8)))  # [z][y][x]
    return dims, voxel_world, proxy_level, proxy_dims, proxy, sampled


def exposures(dims, voxel_world, proxy_level, proxy_dims, proxy):
    """Per axis, the proxy depth before and after each proxy cell along it (computeVisibility's mDepthBefore / After)."""
    px, py, pz = proxy_dims
    means = np.maximum(proxy[: px * py * pz].astype(np.float32), 0.0).reshape(pz, py, px) * float(1 << proxy_level) * voxel_world
    out = []
    for axis in range(3):  # x, y, z -> numpy axes 2, 1, 0
        a = 2 - axis
        before = np.cumsum(means, axis=a) - means
        after = np.flip(np.cumsum(np.flip(means, a), axis=a), a) - means
        out.append((before, after))
    return out


def chunk_errors(chunk, chunk_dims, coords, truth, fitted, voxel_world, proxy_level, depth):
    """measureTransmittance for one chunk: errors of every non-empty column on each axis."""
    cx, cy, cz = chunk % chunk_dims[0], (chunk // chunk_dims[0]) % chunk_dims[1], chunk // (chunk_dims[0] * chunk_dims[1])
    t = np.zeros((N, N, N), np.float32)
    d = np.zeros((N, N, N), np.float32)
    b = np.stack([coords & 1023, (coords >> 10) & 1023, coords >> 20], 1) % 16 * 8
    for i in range(len(coords)):
        x, y, z = b[i]
        t[z:z + 8, y:y + 8, x:x + 8] = truth[i]
        d[z:z + 8, y:y + 8, x:x + 8] = fitted[i]
    errs = []
    origin = (cx, cy, cz)
    for axis in range(3):
        a = 2 - axis
        tau_t = t.sum(axis=a) * voxel_world
        tau_d = d.sum(axis=a) * voxel_world
        before, after = depth[axis]
        first = (origin[axis] * N) >> proxy_level
        last = ((origin[axis] + 1) * N - 1) >> proxy_level
        o = [ax for ax in range(3) if ax != axis]  # the column's other two axes, increasing (x, y, z order)
        idx = [None, None, None]
        idx[o[0]] = (origin[o[0]] * N + np.arange(N)) >> proxy_level
        idx[o[1]] = (origin[o[1]] * N + np.arange(N)) >> proxy_level

        def at(field, fixed):
            ix = [None, None, None]
            ix[axis] = np.full((N, N), fixed)
            g0, g1 = np.meshgrid(idx[o[0]], idx[o[1]], indexing="ij")
            ix[o[0]], ix[o[1]] = g0, g1
            return field[ix[2], ix[1], ix[0]]

        exposure = np.exp(-np.minimum(at(before, first), at(after, last)))  # [o[0], o[1]]
        tt, dd = tau_t.T, tau_d.T  # numpy (z, y) / (z, x) / (y, x) -> (o[0], o[1]), the lower world axis first
        mask = (tt > 0) | (dd > 0)
        errs.append((exposure * np.abs(np.exp(-tt) - np.exp(-dd)))[mask])
    return np.concatenate(errs)


GRID = np.stack(np.meshgrid(np.arange(8), np.arange(8), np.arange(8), indexing="ij"), -1).reshape(512, 3).astype(np.float32)  # z y x


def fit(cores, K, kind, iterations=400, lr=0.05):
    """Least-squares K-kernel fit per brick, batched. cores [B, 512]. Returns the fitted values [B, 512]."""
    B = cores.shape[0]
    rng = np.random.default_rng(0)
    # Initial centres at density-weighted random voxels, widths ~2 voxels, weights at the local density.
    p = cores.astype(np.float64)
    p /= np.maximum(p.sum(1, keepdims=True), 1e-300)  # float64: choice() rejects float32 sums off 1
    pick = np.array([rng.choice(512, K, p=pi, replace=K > np.count_nonzero(pi)) if pi.sum() > 0 else rng.choice(512, K) for pi in p])
    mu = GRID[pick] + rng.normal(0, 0.3, (B, K, 3)).astype(np.float32)
    logs = np.full((B, K, 3), np.log(2.0 if kind == "gauss" else 3.5), np.float32)
    rawq = np.log(np.expm1(np.maximum(np.take_along_axis(cores, pick, 1), 1e-6)))[..., None].astype(np.float32)
    params = [mu, logs, rawq]
    m = [np.zeros_like(x) for x in params]
    v = [np.zeros_like(x) for x in params]
    for it in range(1, iterations + 1):
        mu, logs, rawq = params
        s = np.exp(logs)
        w = np.log1p(np.exp(rawq))  # softplus: weights >= 0
        dx = (GRID[None, None] - mu[:, :, None]) / s[:, :, None]  # [B, K, 512, 3]
        r2 = (dx * dx).sum(-1)
        if kind == "gauss":
            g = np.exp(-0.5 * r2)
            dg_dr2 = -0.5 * g
        else:  # Epanechnikov: max(0, 1 - r2)
            g = np.maximum(1.0 - r2, 0.0)
            dg_dr2 = -(r2 < 1.0).astype(np.float32)
        f = (w * g).sum(1)  # [B, 512]
        res = f - cores
        gw = 2 * (res[:, None] * g).sum(-1, keepdims=True) * (1 / (1 + np.exp(-rawq)))
        common = 2 * res[:, None] * w * dg_dr2  # dL/dr2 [B, K, 512]
        gmu = (common[..., None] * (-2 * dx / s[:, :, None])).sum(2)
        glogs = (common[..., None] * (-2 * dx * dx)).sum(2)
        grads = [gmu, glogs, gw]
        for j in range(3):
            m[j] = 0.9 * m[j] + 0.1 * grads[j]
            v[j] = 0.999 * v[j] + 0.001 * grads[j] ** 2
            step = lr * (m[j] / (1 - 0.9 ** it)) / (np.sqrt(v[j] / (1 - 0.999 ** it)) + 1e-12)
            params[j] = params[j] - step
        params[1] = np.clip(params[1], np.log(0.3), np.log(16.0))
    mu, logs, rawq = params
    s = np.exp(logs)
    w = np.log1p(np.exp(rawq))
    dx = (GRID[None, None] - mu[:, :, None]) / s[:, :, None]
    r2 = (dx * dx).sum(-1)
    g = np.exp(-0.5 * r2) if kind == "gauss" else np.maximum(1.0 - r2, 0.0)
    return (w * g).sum(1)


def atlas8(cores):
    lo, hi = cores.min(1, keepdims=True), cores.max(1, keepdims=True)
    rng = np.maximum(hi - lo, 1e-30)
    code = np.round(np.sqrt(np.clip((cores - lo) / rng, 0, 1)) * 255.0)
    return lo + rng * (code / 255.0) ** 2


def level1(cores):
    c = cores.reshape(-1, 4, 2, 4, 2, 4, 2).mean((2, 4, 6))
    return np.repeat(np.repeat(np.repeat(c, 2, 1), 2, 2), 2, 3).reshape(-1, 512)


def report(label, errs, bricks, kernels=None):
    e = np.concatenate(errs)
    extra = f", {kernels} kernels a brick" if kernels else ""
    print(f"KERNEL {label}: {len(e)} columns over {bricks} bricks{extra}: mean {e.mean():.5f}, p99 {np.percentile(e, 99):.4f}, "
          f"p99.9 {np.percentile(e, 99.9):.4f}, max {e.max():.4f}", flush=True)


if __name__ == "__main__":
    dims, voxel_world, proxy_level, proxy_dims, proxy, sampled = read_cache(CACHE)
    chunk_dims = [d // N for d in dims]
    depth = exposures(dims, voxel_world, proxy_level, proxy_dims, proxy)
    bricks = sum(len(c) for _, c, _ in sampled)
    print(f"KERNEL {CACHE.name}: dims {dims}, voxel {voxel_world}, {len(sampled)} chunks (every {STRIDE}th), {bricks} bricks", flush=True)
    arms = [("zero (sanity)", lambda c: np.zeros_like(c), None), ("truth (anchor)", lambda c: c, None),
            ("atlas 8-bit sqrt", atlas8, None), ("level-1 means", level1, None)]
    for kind in ("gauss", "epan"):
        for K in KS:
            arms.append((f"{kind} K{K}", (lambda kind, K: lambda c: np.concatenate(
                [fit(c[i:i + 512], K, kind) for i in range(0, len(c), 512)]))(kind, K), K))
    for label, fn, K in arms:
        errs = []
        for chunk, coords, cores in sampled:
            flat = cores.reshape(len(coords), 512).astype(np.float32)
            fitted = fn(flat).reshape(-1, 8, 8, 8)
            errs.append(chunk_errors(chunk, chunk_dims, coords, cores, fitted, voxel_world, proxy_level, depth))
        report(label, errs, bricks, K)
