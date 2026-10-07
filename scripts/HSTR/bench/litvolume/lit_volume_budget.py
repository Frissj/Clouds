"""Lit World Volume gate S1 (offline, no Mogwai): how many world bricks does a de-instanced lit volume hold, and how many new ones
does a moving camera need each frame?

LIT_VOLUME.md section 6. The sea is laid out exactly as CloudSea::makeTile lays it out (hash, asset, quarter turns, mirror, scale
0.65-1, offset; two layers, the second shifted half a tile), around the IntelCloudSeaHalf camera (cloud_sea.pyscene: (0, 140, 0)
looking at (0, 40, 600), 28 mm, 4K), moved along +z as sunset_motion's walk (2 units a frame) and sprint (20). Rays every
--spacing pixels march the sea's coarse density (the assets' level-4 volumes rasterised into a 4-unit world grid, layers summed) to
the near view distance (7 tiles) or transmittance 1e-3 (cloudMinTransmittance), in steps of at most half the finest brick that can
be asked for there. Every sample in a cloud names the brick the march would read there:
  world     level floor(log2(max(t * pixelAngle / W_0, 1))), brick floor(p / (8 W_l))       - the lit volume, W_0 = c * s_max
  instance  (layer tile, its footprintLevel, its asset brick)                                  - per-instance lit bricks
  shared    (asset, footprintLevel, asset brick)                                               - today's shared density atlas
and only bricks whose box holds density (the assets' level-2 occupancy, by summed-area table) count. Reported per frame: bricks
per level, MB at 2 KB a brick (10^3 RG8 texels), bricks not touched in the previous --keep frames (what a cache keeping bricks
that long would bake), and the flight's union.
Stop rule (LIT_VOLUME.md, before any run): close Form A if the walk needs > 2.6 GB or > 4k new bricks a frame at the chosen c.

    python lit_volume_budget.py --assets a0_L4.npy,...,a4_L4.npy --occupancy a0_L2.npy,...,a4_L2.npy --spacing 4
    python lit_volume_budget.py --synthetic          smoke test of the script itself (numbers mean nothing)

Volumes are written by hstrlib_dense (float16, shape (z, y, x), one asset at one level). numpy only; rays split over --workers
processes (fork).
"""
import argparse
import multiprocessing as mp
import time

import numpy as np

TILE_WORLD = 320.0
LAYER_HEIGHT = 160.0
TILES = 16
ORIGIN = np.array([-160.0, -80.0, -160.0])  # the carrier box's minimum (cloud_sea.pyscene: 320 x 160 x 320 about the origin)
COVERAGE = 1.0  # IntelCloudSeaHalf cloudSeaCoverage
SEED = 1
MIN_SCALE = 0.65
CAMERA = np.array([0.0, 140.0, 0.0])
TARGET = np.array([0.0, 40.0, 600.0])
FRAME = (3840, 2160)
PIXEL_ANGLE = 24.0 / 28.0 / FRAME[1]  # cloudPixelAngle: frame height / focal length / pixels
VIEW_DISTANCE = (TILES / 2 - 1) * TILE_WORLD  # HSTRCloud: min(seaViewDistance, (tiles / 2 - 1) * tileWorld)
MIN_TRANSMITTANCE = 1e-3
BRICK_BYTES = 2000
GRID = 4.0  # world units per voxel of the rasterised coarse density
HALF = 9  # tiles each side of the camera's tile in the instance tables
M32 = 0xFFFFFFFF


def hash_uint(v):
    """CloudSea.cpp hashUint, on uint32."""
    v &= M32
    state = (v * 747796405 + 2891336453) & M32
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & M32
    return ((word >> 22) ^ word) & M32


def make_tile(tx, tz, layer, assets, fit):
    """CloudSea::makeTile: the instance of world tile (tx, tz) in a layer, as (occupied, asset, linear / s, offset) with world
    position p mapping to source voxel ((p - offset) @ (linear / s)) + content_min."""
    corner = ORIGIN + (np.array([tx, 0.0, tz]) + np.array([0.5, 0.0, 0.5]) * layer) * TILE_WORLD
    state = hash_uint(((tx & M32) * 73856093) ^ hash_uint(((tz & M32) * 19349663) ^ hash_uint(SEED + 0x9E3779B9 * layer)))

    def nxt():
        nonlocal state
        state = hash_uint(state)
        return float(state >> 8) / 16777216.0

    occupied = nxt() < COVERAGE
    asset_id = min(int(nxt() * len(assets)), len(assets) - 1)
    turns = min(int(nxt() * 4.0), 3)
    mirror = nxt() < 0.5
    scale = MIN_SCALE + (1.0 - MIN_SCALE) * nxt()
    a = assets[asset_id]
    s = a.voxel_world * fit * scale
    content = a.content_max - a.content_min + 1
    extent = content * s
    footprint = np.array([extent[2], extent[1], extent[0]]) if turns & 1 else extent
    offset = np.array([nxt() * max(0.0, TILE_WORLD - footprint[0]),
                       nxt() * max(0.0, LAYER_HEIGHT - footprint[1]),
                       nxt() * max(0.0, TILE_WORLD - footprint[2])])
    linear = np.eye(3)
    if mirror:
        linear[0, 0] = -1.0
    r = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    for _ in range(turns):
        linear = r @ linear
    translation = offset + 0.5 * footprint - (linear * s) @ (0.5 * (content - 1))
    return occupied, asset_id, s, linear / s, corner + translation


class Asset:
    def __init__(self, coarse, level, occupancy, occupancy_level, voxel_world):
        self.coarse = np.asarray(coarse, np.float32)
        self.level = level
        self.voxel_world = voxel_world
        occ = (np.asarray(occupancy, np.float32) > 0).astype(np.int32)
        self.occ_level = occupancy_level
        self.sat = np.zeros(tuple(s + 1 for s in occ.shape), np.int32)
        self.sat[1:, 1:, 1:] = occ.cumsum(0).cumsum(1).cumsum(2)
        # Content bounds as CloudSea takes them (proxy maxima at level 5: 32 source voxels), from the occupancy level: the coarse
        # level's own blur reaches the padded box's edges and would shrink the fit scale.
        nz = np.argwhere(occ > 0)
        s = 2 ** occupancy_level
        self.content_min = ((nz.min(0)[::-1] * s) // 32 * 32).astype(np.float64)
        self.content_max = (-(-((nz.max(0)[::-1] + 1) * s) // 32) * 32 - 1).astype(np.float64)
        self.dims = np.array(occ.shape[::-1]) * s

    def occupied(self, lo, hi):
        """Whether the source-voxel boxes [lo, hi) (n, 3 as x, y, z) hold density, by the occupancy level's summed-area table."""
        s = 2 ** self.occ_level
        top = np.array(self.sat.shape[::-1]) - 1
        a = np.clip(np.floor(lo / s).astype(np.int64), 0, top)
        b = np.clip(np.ceil(hi / s).astype(np.int64), 0, top)
        x0, y0, z0 = a[:, 0], a[:, 1], a[:, 2]
        x1, y1, z1 = b[:, 0], b[:, 1], b[:, 2]
        t = self.sat
        count = (t[z1, y1, x1] - t[z0, y1, x1] - t[z1, y0, x1] - t[z1, y1, x0] + t[z0, y0, x1] + t[z0, y1, x0] + t[z1, y0, x0]
                 - t[z0, y0, x0])
        return count > 0


def trilinear(grid, q):
    """grid (nz, ny, nx), q (n, 3) as (x, y, z) in voxel units (centres at integers); zero outside."""
    nz, ny, nx = grid.shape
    i = np.floor(q).astype(np.int64)
    f = (q - i).astype(np.float32)
    out = np.zeros(q.shape[0], np.float32)
    for corner in range(8):
        dx, dy, dz = corner & 1, (corner >> 1) & 1, corner >> 2
        x, y, z = i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz
        ok = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & (z >= 0) & (z < nz)
        w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
        out[ok] += w[ok] * grid[z[ok], y[ok], x[ok]]
    return out


class Sea:
    """Instance tables around a centre tile, and the coarse density of both layers rasterised on a world grid."""

    def __init__(self, assets, fit, centre):
        self.assets = assets
        self.fit = fit
        n = 2 * HALF + 1
        self.base = []
        self.occupied = np.zeros((2, n, n), bool)
        self.asset = np.zeros((2, n, n), np.int64)
        self.s = np.ones((2, n, n))
        self.lin = np.zeros((2, n, n, 3, 3))
        self.off = np.zeros((2, n, n, 3))
        for layer in range(2):
            b = np.floor((centre[[0, 2]] - ORIGIN[[0, 2]] - 0.5 * layer * TILE_WORLD) / TILE_WORLD).astype(np.int64) - HALF
            self.base.append(b)
            for i in range(n):
                for j in range(n):
                    occ, a, s, lin, off = make_tile(int(b[0] + i), int(b[1] + j), layer, assets, fit)
                    self.occupied[layer, i, j], self.asset[layer, i, j], self.s[layer, i, j] = occ, a, s
                    self.lin[layer, i, j], self.off[layer, i, j] = lin, off
        # The coarse grid over the window's tiles (both layers' squares lie inside layer 0's window shrunk by one tile).
        self.lo = ORIGIN + np.array([self.base[0][0] + 1, 0, self.base[0][1] + 1]) * np.array([TILE_WORLD, 0, TILE_WORLD])
        size = np.array([(n - 2) * TILE_WORLD, LAYER_HEIGHT, (n - 2) * TILE_WORLD])
        dims = np.ceil(size / GRID).astype(int) + 1
        self.grid = np.zeros((dims[2], dims[1], dims[0]), np.float32)
        zz = np.arange(dims[2])
        xx = np.arange(dims[0])
        yy = np.arange(dims[1])
        for layer in range(2):
            for i in range(n):
                for j in range(n):
                    if not self.occupied[layer, i, j]:
                        continue
                    sq = ORIGIN[[0, 2]] + (np.array([self.base[layer][0] + i, self.base[layer][1] + j]) + 0.5 * layer) * TILE_WORLD
                    gx = xx[(self.lo[0] + xx * GRID >= sq[0]) & (self.lo[0] + xx * GRID < sq[0] + TILE_WORLD)]
                    gz = zz[(self.lo[2] + zz * GRID >= sq[1]) & (self.lo[2] + zz * GRID < sq[1] + TILE_WORLD)]
                    if gx.size == 0 or gz.size == 0:
                        continue
                    Z, Y, X = np.meshgrid(gz, yy, gx, indexing="ij")
                    p = self.lo + np.stack([X, Y, Z], -1).reshape(-1, 3) * GRID
                    a = assets[self.asset[layer, i, j]]
                    src = (p - self.off[layer, i, j]) @ self.lin[layer, i, j] + a.content_min
                    v = trilinear(a.coarse, src / (2 ** a.level) - 0.5 + 0.5 / (2 ** a.level))
                    self.grid[gz[0]:gz[-1] + 1, :, gx[0]:gx[-1] + 1] += v.reshape(Z.shape)

    def density(self, p):
        return trilinear(self.grid, (p - self.lo) / GRID)

    def instance(self, p, layer):
        """Table index (i, j) of each world point's tile in a layer, and whether it lies in the table."""
        g = np.floor((p[:, [0, 2]] - ORIGIN[[0, 2]] - 0.5 * layer * TILE_WORLD) / TILE_WORLD).astype(np.int64) - self.base[layer]
        n = 2 * HALF + 1
        ok = np.all((g >= 0) & (g < n), axis=1)
        return np.clip(g[:, 0], 0, n - 1), np.clip(g[:, 1], 0, n - 1), ok


# --- keys (int64-packed) ----------------------------------------------------------------------------------------------------------
BIAS = 1 << 19


def pack_world(level, b):
    return (level << 60) | ((b[:, 0] + BIAS) << 40) | ((b[:, 1] + BIAS) << 20) | (b[:, 2] + BIAS)


def unpack_world(k):
    return k >> 60, np.stack([((k >> 40) & 0xFFFFF) - BIAS, ((k >> 20) & 0xFFFFF) - BIAS, (k & 0xFFFFF) - BIAS], 1)


def pack_asset(head, level, b):
    """head: instance (layer * 1024 + i * 32 + j) or asset id; asset bricks < 2048 an axis."""
    return (head << 36) | (level << 33) | (b[:, 0] << 22) | (b[:, 1] << 11) | b[:, 2]


def unpack_asset(k):
    return k >> 36, (k >> 33) & 7, np.stack([(k >> 22) & 2047, (k >> 11) & 2047, k & 2047], 1)


_SEA = None
_CFG = None


def march_chunk(rays):
    """Marches rays (n, 3 directions) from _CFG['camera']; returns unique world keys per c, instance keys and shared keys."""
    sea, cfg = _SEA, _CFG
    camera, cs, s_max0 = cfg["camera"], cfg["c"], cfg["s_max0"]
    w0min = min(cs) * s_max0 * MIN_SCALE
    d = rays
    n = d.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (ORIGIN[1] - camera[1]) / d[:, 1]
        t1 = (ORIGIN[1] + LAYER_HEIGHT - camera[1]) / d[:, 1]
    enter = np.clip(np.nan_to_num(np.minimum(t0, t1), nan=0.0, posinf=VIEW_DISTANCE, neginf=0.0), 0.0, VIEW_DISTANCE)
    leave = np.clip(np.nan_to_num(np.maximum(t0, t1), nan=0.0, posinf=VIEW_DISTANCE, neginf=0.0), 0.0, VIEW_DISTANCE)
    if ORIGIN[1] <= camera[1] <= ORIGIN[1] + LAYER_HEIGHT:
        enter[:] = 0.0
    t = enter.copy()
    trans = np.ones(n)
    alive = t < leave
    world = {c: [] for c in cs}
    inst, shared = [], []
    while np.any(alive):
        idx = np.nonzero(alive)[0]
        # Half the finest brick that can be asked for here (any c, any instance scale): no brick is stepped over.
        step = 4.0 * np.maximum(w0min, t[idx] * PIXEL_ANGLE)
        tm = t[idx] + 0.5 * step
        p = camera + tm[:, None] * d[idx]
        sigma = sea.density(p)
        hit = sigma > 0
        if np.any(hit):
            ph, th = p[hit], tm[hit]
            for c in cs:
                w0 = c * s_max0
                level = np.floor(np.log2(np.maximum(th * PIXEL_ANGLE / w0, 1.0))).astype(np.int64)
                b = np.floor(ph / (8.0 * w0 * 2.0 ** level)[:, None]).astype(np.int64)
                world[c].append(np.unique(pack_world(level, b)))
            for layer in range(2):
                i, j, ok = sea.instance(ph, layer)
                ok &= sea.occupied[layer, i, j]
                if not np.any(ok):
                    continue
                i, j, q, tq = i[ok], j[ok], ph[ok], th[ok]
                src = np.einsum("ni,nij->nj", q - sea.off[layer, i, j], sea.lin[layer, i, j]) + np.stack(
                    [sea.assets[a].content_min for a in range(len(sea.assets))])[sea.asset[layer, i, j]]
                level = np.floor(np.log2(np.maximum(tq * PIXEL_ANGLE / sea.s[layer, i, j], 1.0))).astype(np.int64)
                level = np.minimum(level, 7)
                b = np.floor(src / (8.0 * 2.0 ** level)[:, None]).astype(np.int64)
                dims = np.stack([sea.assets[a].dims for a in range(len(sea.assets))])[sea.asset[layer, i, j]]
                inside = np.all((src >= 0) & (src < dims), axis=1)
                if not np.any(inside):
                    continue
                head = layer * 1024 + i * 32 + j
                inst.append(np.unique(pack_asset(head[inside], level[inside], b[inside])))
                shared.append(np.unique(pack_asset(sea.asset[layer, i, j][inside], level[inside], b[inside])))
        trans[idx] *= np.exp(-sigma * step)
        t[idx] += step
        alive[idx] = (t[idx] < leave[idx]) & (trans[idx] > MIN_TRANSMITTANCE)
    cat = lambda xs: np.unique(np.concatenate(xs)) if xs else np.zeros(0, np.int64)
    return {c: cat(world[c]) for c in cs}, cat(inst), cat(shared)


def occupied_world(sea, keys, w0):
    """World bricks whose box holds density in either layer's instance (tile under the brick's centre)."""
    level, b = unpack_world(keys)
    size = 8.0 * w0 * 2.0 ** level
    lo = b * size[:, None]
    hi = lo + size[:, None]
    centre = 0.5 * (lo + hi)
    out = np.zeros(keys.size, bool)
    for layer in range(2):
        i, j, ok = sea.instance(centre, layer)
        ok &= sea.occupied[layer, i, j]
        for a in range(len(sea.assets)):
            m = ok & (sea.asset[layer, i, j] == a)
            if not np.any(m):
                continue
            s0 = np.einsum("ni,nij->nj", lo[m] - sea.off[layer, i[m], j[m]], sea.lin[layer, i[m], j[m]]) + sea.assets[a].content_min
            s1 = np.einsum("ni,nij->nj", hi[m] - sea.off[layer, i[m], j[m]], sea.lin[layer, i[m], j[m]]) + sea.assets[a].content_min
            out[np.nonzero(m)[0]] |= sea.assets[a].occupied(np.minimum(s0, s1), np.maximum(s0, s1))
    return keys[out]


def occupied_asset(sea, keys, asset_of_head):
    head, level, b = unpack_asset(keys)
    asset = asset_of_head(head)
    out = np.zeros(keys.size, bool)
    for a in range(len(sea.assets)):
        m = asset == a
        if np.any(m):
            size = (8.0 * 2.0 ** level[m])[:, None]
            out[np.nonzero(m)[0]] = sea.assets[a].occupied(b[m] * size, (b[m] + 1) * size)
    return keys[out]


def rays(camera, spacing):
    forward = (TARGET - CAMERA) / np.linalg.norm(TARGET - CAMERA)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    xs = (np.arange(0, FRAME[0], spacing) + 0.5 * spacing - 0.5 * FRAME[0]) * PIXEL_ANGLE
    ys = (np.arange(0, FRAME[1], spacing) + 0.5 * spacing - 0.5 * FRAME[1]) * PIXEL_ANGLE
    gx, gy = np.meshgrid(xs, ys)
    d = forward[None, :] + gx.reshape(-1, 1) * right[None, :] - gy.reshape(-1, 1) * up[None, :]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def synthetic_assets():
    rng = np.random.default_rng(3)
    assets = []
    for _ in range(5):
        n = (88, 56, 80)
        z, y, x = np.meshgrid(np.arange(n[2]), np.arange(n[1]), np.arange(n[0]), indexing="ij")
        f = np.zeros((n[2], n[1], n[0]), np.float32)
        for _ in range(10):
            c = rng.uniform([15, 10, 15], [n[0] - 15, n[1] - 10, n[2] - 15])
            r = rng.uniform(6, 14)
            f = np.maximum(f, np.clip(1.5 - 1.5 * np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / r, 0, 1))
        assets.append(Asset(f[::4, ::4, ::4] * 0.05, 4, f, 2, 0.25))  # the 88-wide grid stands for level 2 of a 352-wide asset
    return assets


def main():
    global _SEA, _CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", help="comma-separated coarse volumes, one per asset in library order")
    ap.add_argument("--assets-level", type=int, default=4)
    ap.add_argument("--occupancy", help="comma-separated occupancy volumes (same order)")
    ap.add_argument("--occupancy-level", type=int, default=2)
    ap.add_argument("--voxel-world", type=float, default=0.25)
    ap.add_argument("--c", default="1.0,0.8,0.65")
    ap.add_argument("--walk-frames", type=int, default=6)
    ap.add_argument("--sprint-frames", type=int, default=4)
    ap.add_argument("--spacing", type=int, default=4, help="pixels between rays; must stay under half a brick's screen width "
                    "(LOD-matched bricks are 8-16 px), or the touched set flickers and reads as turnover")
    ap.add_argument("--keep", type=int, default=4, help="frames a brick stays resident unused: new = not touched in the last keep frames")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.synthetic:
        assets = synthetic_assets()
    else:
        coarse = [np.load(f) for f in args.assets.split(",")]
        occ = [np.load(f, mmap_mode="r") for f in args.occupancy.split(",")]
        assets = [Asset(c, args.assets_level, o, args.occupancy_level, args.voxel_world) for c, o in zip(coarse, occ)]
    fit = min(0.98 * min(TILE_WORLD / max(e[0], e[2]), LAYER_HEIGHT / e[1])
              for e in ((a.content_max - a.content_min + 1) * a.voxel_world for a in assets))
    s_max0 = assets[0].voxel_world * fit
    cs = [float(v) for v in args.c.split(",")]
    start = time.time()
    _SEA = Sea(assets, fit, CAMERA)
    print(f"{len(assets)} assets, fit scale {fit:.4f}, largest level-0 instance voxel {s_max0:.4f} world units (level-0 reach "
          f"{s_max0 / PIXEL_ANGLE:.0f} units); world W_0 = c x that, c in {cs}; rays every {args.spacing} px; view distance "
          f"{VIEW_DISTANCE:.0f}; bricks kept {args.keep} frames; sea rasterised in {time.time() - start:.0f} s", flush=True)
    for motion, speed, frames in (("walk", 2.0, args.walk_frames), ("sprint", 20.0, args.sprint_frames)):
        history = {}
        union = {}

        def new_count(kind, s):
            past = history.setdefault(kind, [])
            n = len(s - set().union(*past)) if past else len(s)
            past.append(s)
            del past[:-args.keep]
            union.setdefault(kind, set()).update(s)
            return n

        for f in range(frames):
            start = time.time()
            camera = CAMERA + np.array([0.0, 0.0, speed * f])
            _CFG = dict(camera=camera, c=cs, s_max0=s_max0)
            d = rays(camera, args.spacing)
            chunks = np.array_split(d, args.workers * 4)
            with mp.get_context("fork").Pool(args.workers) as pool:
                parts = pool.map(march_chunk, chunks)
            line = [f"{motion} frame {f} (camera z {camera[2]:.0f}, {time.time() - start:.0f} s):"]
            for c in cs:
                keys = occupied_world(_SEA, np.unique(np.concatenate([p[0][c] for p in parts])), c * s_max0)
                level, _ = unpack_world(keys)
                ks = set(keys.tolist())
                new = new_count(("world", c), ks)
                line.append(f"  world c={c}: {len(ks)} bricks ({len(ks) * BRICK_BYTES / 2**20:.0f} MB) by level "
                            f"{np.bincount(level, minlength=1).tolist()}, new {new}, union {len(union[('world', c)])}")
            inst = occupied_asset(_SEA, np.unique(np.concatenate([p[1] for p in parts])),
                                  lambda h: _SEA.asset[h >> 10, (h >> 5) & 31, h & 31])
            shared = occupied_asset(_SEA, np.unique(np.concatenate([p[2] for p in parts])), lambda h: h)
            si, ss = set(inst.tolist()), set(shared.tolist())
            line.append(f"  per-instance: {len(si)} bricks ({len(si) * BRICK_BYTES / 2**20:.0f} MB) by level "
                        f"{np.bincount(unpack_asset(inst)[1], minlength=1).tolist()}, new {new_count('instance', si)}, union "
                        f"{len(union['instance'])}")
            line.append(f"  shared asset bricks: {len(ss)} ({len(ss) * BRICK_BYTES / 2**20:.0f} MB) by level "
                        f"{np.bincount(unpack_asset(shared)[1], minlength=1).tolist()}, new {new_count('shared', ss)}, union "
                        f"{len(union['shared'])}")
            print("\n".join(line), flush=True)


if __name__ == "__main__":
    main()
