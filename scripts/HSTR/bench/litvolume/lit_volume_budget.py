"""Lit World Volume gate S1 (offline, no Mogwai): how many world bricks does a de-instanced lit volume hold, and how many new ones
does a moving camera need each frame?

LIT_VOLUME.md section 6. The sea is laid out exactly as CloudSea::makeTile lays it out (hash, asset, quarter turns, mirror, scale
0.65-1, offset; two layers, the second shifted half a tile), around the IntelCloudSeaHalf camera (cloud_sea.pyscene: (0, 140, 0)
looking at (0, 40, 600), 28 mm, 4K), moved along +z as sunset_motion's walk (2 units a frame) and sprint (20). Rays every 8 pixels
march the sea's coarse density (the assets' level-4 volumes, summed over layers) to the near view distance (7 tiles) or
transmittance 1e-3 (cloudMinTransmittance). Every sample in a cloud names the brick the march would read there:
  world     level floor(log2(max(t * pixelAngle / W_0, 1))), brick floor(p / (8 W_l))       - the lit volume, W_0 = c * s_max
  instance  (instance, its footprintLevel, its asset brick)                                    - per-instance lit bricks
  shared    (asset, footprintLevel, asset brick)                                               - today's shared density atlas
and only bricks whose box holds density (the assets' level-2 occupancy, by summed-area table) count. Reported per frame: bricks
per level, MB at 2 KB a brick (10^3 RG8 texels), bricks new since the previous frame, and the flight's union.
Stop rule (LIT_VOLUME.md, before any run): close if the walk needs > 2.6 GB or > 4k new bricks a frame at the chosen c.

    python lit_volume_budget.py --assets a0_L4.npy,...,a4_L4.npy --occupancy a0_L2.npy,...,a4_L2.npy [--voxel-world 0.25]
    python lit_volume_budget.py --synthetic          smoke test of the script itself (numbers mean nothing)

Volumes are written by hstrlib_dense (float16, shape (z, y, x), one asset at one level). Coarse (default level 4) for the
transmittance, level 2 for occupancy. numpy only.
"""
import argparse
import math
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
RAY_SPACING = 8
BRICK_BYTES = 2000


def hash_uint(v):
    v = np.uint32(v)
    with np.errstate(over="ignore"):
        state = np.uint32(v * np.uint32(747796405) + np.uint32(2891336453))
        word = np.uint32((state >> np.uint32((int(state) >> 28) + 4)) ^ state) * np.uint32(277803737)
        return np.uint32((word >> np.uint32(22)) ^ word)


class Rng:
    def __init__(self, state):
        self.state = np.uint32(state)

    def next(self):
        self.state = hash_uint(self.state)
        return float(int(self.state) >> 8) / 16777216.0


class Asset:
    def __init__(self, coarse, level, occupancy, occupancy_level, voxel_world):
        self.coarse = coarse.astype(np.float32)
        self.level = level
        self.voxel_world = voxel_world
        occ = (occupancy.astype(np.float32) > 0).astype(np.int32)
        self.occ_level = occupancy_level
        self.sat = np.zeros(tuple(s + 1 for s in occ.shape), np.int32)
        self.sat[1:, 1:, 1:] = occ.cumsum(0).cumsum(1).cumsum(2)
        nz = np.argwhere(self.coarse > 0)
        lo = nz.min(0)[::-1] * (2 ** level)
        hi = (nz.max(0)[::-1] + 1) * (2 ** level) - 1
        self.content_min = lo.astype(np.float64)
        self.content_max = hi.astype(np.float64)

    def occupied(self, lo, hi):
        """Whether the source-voxel box [lo, hi) (x, y, z; arrays (n, 3)) holds density, by the occupancy level's table."""
        s = 2 ** self.occ_level
        a = np.clip(np.floor(lo / s).astype(np.int64), 0, np.array(self.sat.shape[::-1]) - 1)
        b = np.clip(np.ceil(hi / s).astype(np.int64), 0, np.array(self.sat.shape[::-1]) - 1)
        x0, y0, z0 = a[:, 0], a[:, 1], a[:, 2]
        x1, y1, z1 = b[:, 0], b[:, 1], b[:, 2]
        t = self.sat
        count = (t[z1, y1, x1] - t[z0, y1, x1] - t[z1, y0, x1] - t[z1, y1, x0] + t[z0, y0, x1] + t[z0, y1, x0] + t[z1, y0, x0]
                 - t[z0, y0, x0])
        return count > 0


def make_tile(world, layer, assets, fit):
    """CloudSea::makeTile: the instance of world tile (tx, tz) in a layer."""
    tx, tz = world
    corner = ORIGIN + (np.array([tx, 0.0, tz]) + np.array([0.5, 0.0, 0.5]) * layer) * TILE_WORLD
    with np.errstate(over="ignore"):
        seed = hash_uint(np.uint32(np.uint32(tx & 0xFFFFFFFF) * np.uint32(73856093)) ^ hash_uint(
            np.uint32(np.uint32(tz & 0xFFFFFFFF) * np.uint32(19349663)) ^ hash_uint(np.uint32(SEED + 0x9E3779B9 * layer & 0xFFFFFFFF))))
    rng = Rng(seed)
    occupied = rng.next() < COVERAGE
    asset_id = min(int(rng.next() * len(assets)), len(assets) - 1)
    turns = min(int(rng.next() * 4.0), 3)
    mirror = rng.next() < 0.5
    scale = MIN_SCALE + (1.0 - MIN_SCALE) * rng.next()
    a = assets[asset_id]
    s = a.voxel_world * fit * scale
    content = a.content_max - a.content_min + 1
    extent = content * s
    footprint = np.array([extent[2], extent[1], extent[0]]) if turns & 1 else extent
    offset = np.array([rng.next() * max(0.0, TILE_WORLD - footprint[0]),
                       rng.next() * max(0.0, LAYER_HEIGHT - footprint[1]),
                       rng.next() * max(0.0, TILE_WORLD - footprint[2])])
    linear = np.eye(3)
    if mirror:
        linear[0, 0] = -1.0
    r = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    for _ in range(turns):
        linear = r @ linear
    forward = linear * s
    translation = offset + 0.5 * footprint - forward @ (0.5 * (content - 1))
    return dict(occupied=occupied, asset=asset_id, linear=linear, s=s, corner=corner, translation=translation, scale=scale)


def to_source(tile, assets, p):
    """World points (n, 3) to the instance's asset source voxels (centres at integers)."""
    q = p - tile["corner"] - tile["translation"]
    c = (q @ tile["linear"]) / tile["s"]  # linear^T q / s, as rows
    return c + assets[tile["asset"]].content_min


def trilinear(grid, p):
    nz, ny, nx = grid.shape
    q = np.clip(p, 0.0, np.array([nx, ny, nz], np.float64) - 1.000001)
    i = np.floor(q).astype(np.int64)
    f = (q - i).astype(np.float32)
    x0, y0, z0 = i[:, 0], i[:, 1], i[:, 2]
    x1, y1, z1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1), np.minimum(z0 + 1, nz - 1)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    c00 = grid[z0, y0, x0] * (1 - fx) + grid[z0, y0, x1] * fx
    c10 = grid[z0, y1, x0] * (1 - fx) + grid[z0, y1, x1] * fx
    c01 = grid[z1, y0, x0] * (1 - fx) + grid[z1, y0, x1] * fx
    c11 = grid[z1, y1, x0] * (1 - fx) + grid[z1, y1, x1] * fx
    inside = np.all((p >= -0.5) & (p <= np.array([nx, ny, nz]) - 0.5), axis=1)
    return ((c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz) * inside


class Sea:
    def __init__(self, assets, fit):
        self.assets = assets
        self.fit = fit
        self.tiles = {}

    def tile(self, world, layer):
        key = (world, layer)
        if key not in self.tiles:
            self.tiles[key] = make_tile(world, layer, self.assets, self.fit)
        return self.tiles[key]

    def tile_keys(self, p, layer):
        g = np.floor((p[:, [0, 2]] - ORIGIN[[0, 2]] - 0.5 * layer * TILE_WORLD) / TILE_WORLD).astype(np.int64)
        return g

    def density(self, p):
        """Summed extinction of both layers at world points (n, 3), from the coarse volumes."""
        out = np.zeros(p.shape[0], np.float32)
        inside_y = (p[:, 1] >= ORIGIN[1]) & (p[:, 1] <= ORIGIN[1] + LAYER_HEIGHT)
        for layer in range(2):
            g = self.tile_keys(p, layer)
            for key in np.unique(g[inside_y], axis=0):
                tile = self.tile((int(key[0]), int(key[1])), layer)
                if not tile["occupied"]:
                    continue
                m = inside_y & (g[:, 0] == key[0]) & (g[:, 1] == key[1])
                a = self.assets[tile["asset"]]
                x = to_source(tile, self.assets, p[m]) / (2 ** a.level) - 0.5 + 0.5 / (2 ** a.level)
                out[m] += trilinear(a.coarse, x)
        return out


def march(sea, camera, view_dir):
    """Rays every RAY_SPACING pixels to the view distance or opacity; returns the in-cloud samples (position, distance)."""
    forward = view_dir / np.linalg.norm(view_dir)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    xs = (np.arange(0, FRAME[0], RAY_SPACING) + 0.5 * RAY_SPACING - 0.5 * FRAME[0]) * PIXEL_ANGLE
    ys = (np.arange(0, FRAME[1], RAY_SPACING) + 0.5 * RAY_SPACING - 0.5 * FRAME[1]) * PIXEL_ANGLE
    gx, gy = np.meshgrid(xs, ys)
    d = forward[None, :] + gx.reshape(-1, 1) * right[None, :] - gy.reshape(-1, 1) * up[None, :]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = d.shape[0]
    # Only the stretch of each ray inside the cloud layer's slab.
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (ORIGIN[1] - camera[1]) / d[:, 1]
        t1 = (ORIGIN[1] + LAYER_HEIGHT - camera[1]) / d[:, 1]
    enter = np.clip(np.nan_to_num(np.minimum(t0, t1), nan=0.0, neginf=0.0, posinf=VIEW_DISTANCE), 0.0, VIEW_DISTANCE)
    leave = np.clip(np.nan_to_num(np.maximum(t0, t1), nan=VIEW_DISTANCE, neginf=0.0, posinf=VIEW_DISTANCE), 0.0, VIEW_DISTANCE)
    inside = (camera[1] >= ORIGIN[1]) & (camera[1] <= ORIGIN[1] + LAYER_HEIGHT)
    if inside:
        enter[:] = 0.0
    t = enter.copy()
    trans = np.ones(n)
    alive = (t < leave)
    keep_p, keep_t = [], []
    while np.any(alive):
        idx = np.nonzero(alive)[0]
        # Steps of half a level-0 world brick near the camera, growing with the footprint: no brick is stepped over.
        step = np.maximum(0.9, 4.0 * t[idx] * PIXEL_ANGLE)
        tm = t[idx] + 0.5 * step
        p = camera[None, :] + tm[:, None] * d[idx]
        sigma = sea.density(p)
        hit = sigma > 0
        keep_p.append(p[hit])
        keep_t.append(tm[hit])
        trans[idx] *= np.exp(-sigma * step)
        t[idx] += step
        alive[idx] = (t[idx] < leave[idx]) & (trans[idx] > MIN_TRANSMITTANCE)
    return np.concatenate(keep_p), np.concatenate(keep_t)


def keys(sea, p, t, c, s_max0):
    """The world, per-instance and shared brick keys of in-cloud samples, filtered to bricks that hold density."""
    w0 = c * s_max0
    level = np.floor(np.log2(np.maximum(t * PIXEL_ANGLE / w0, 1.0))).astype(np.int64)
    size = 8.0 * w0 * 2.0 ** level
    b = np.floor(p / size[:, None]).astype(np.int64)
    world = np.unique(np.concatenate([level[:, None], b], 1), axis=0)
    # Occupancy of each unique world brick: any layer's instance with density inside its box.
    lo = world[:, 1:] * (8.0 * w0 * 2.0 ** world[:, :1])
    hi = lo + 8.0 * w0 * 2.0 ** world[:, :1]
    occupied = np.zeros(world.shape[0], bool)
    centre = 0.5 * (lo + hi)
    for layer in range(2):
        # The tile under the brick's centre (a brick straddling a tile edge sees one side only: rare at <= ~2 units a brick).
        g = sea.tile_keys(centre, layer)
        for key in np.unique(g, axis=0):
            tile = sea.tile((int(key[0]), int(key[1])), layer)
            if not tile["occupied"]:
                continue
            m = (g[:, 0] == key[0]) & (g[:, 1] == key[1])
            a = sea.assets[tile["asset"]]
            s0 = to_source(tile, sea.assets, lo[m])
            s1 = to_source(tile, sea.assets, hi[m])
            occupied[m] |= a.occupied(np.minimum(s0, s1), np.maximum(s0, s1))
    world = world[occupied]
    # Per instance and shared: the asset brick each sample reads (layer 0's instance where it has density, else layer 1's).
    inst_keys, shared_keys = [], []
    for layer in range(2):
        g = sea.tile_keys(p, layer)
        for key in np.unique(g, axis=0):
            tile = sea.tile((int(key[0]), int(key[1])), layer)
            if not tile["occupied"]:
                continue
            m = (g[:, 0] == key[0]) & (g[:, 1] == key[1])
            a = sea.assets[tile["asset"]]
            x = to_source(tile, sea.assets, p[m])
            lvl = np.floor(np.log2(np.maximum(t[m] * PIXEL_ANGLE / tile["s"], 1.0))).astype(np.int64)
            brick = np.floor(x / (8.0 * 2.0 ** lvl[:, None])).astype(np.int64)
            ok = a.occupied(brick * 8.0 * 2.0 ** lvl[:, None], (brick + 1) * 8.0 * 2.0 ** lvl[:, None])
            tid = (layer * 100003 + int(key[0]) * 1009 + int(key[1]))
            inst_keys.append(np.concatenate([np.full((ok.sum(), 1), tid), lvl[ok, None], brick[ok]], 1))
            shared_keys.append(np.concatenate([np.full((ok.sum(), 1), tile["asset"]), lvl[ok, None], brick[ok]], 1))
    inst = np.unique(np.concatenate(inst_keys), axis=0) if inst_keys else np.zeros((0, 5), np.int64)
    shared = np.unique(np.concatenate(shared_keys), axis=0) if shared_keys else np.zeros((0, 5), np.int64)
    return world, inst, shared


def as_set(a):
    return set(map(tuple, a.tolist()))


def synthetic_assets():
    rng = np.random.default_rng(3)
    assets = []
    for k in range(5):
        n = (88, 56, 80)
        z, y, x = np.meshgrid(np.arange(n[2]), np.arange(n[1]), np.arange(n[0]), indexing="ij")
        f = np.zeros((n[2], n[1], n[0]), np.float32)
        for _ in range(10):
            c = rng.uniform([15, 10, 15], [n[0] - 15, n[1] - 10, n[2] - 15])
            r = rng.uniform(6, 14)
            f = np.maximum(f, np.clip(1.5 - 1.5 * np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / r, 0, 1))
        coarse = f[::4, ::4, ::4] * 0.05
        assets.append(Asset(coarse, 4, f, 2, 0.25))  # pretends the 88-wide grid is level 2 of a 352-wide asset
    return assets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", help="comma-separated coarse volumes, one per asset in library order")
    ap.add_argument("--assets-level", type=int, default=4)
    ap.add_argument("--occupancy", help="comma-separated occupancy volumes (same order)")
    ap.add_argument("--occupancy-level", type=int, default=2)
    ap.add_argument("--voxel-world", type=float, default=0.25)
    ap.add_argument("--c", default="1.0,0.8,0.65")
    ap.add_argument("--walk-frames", type=int, default=6)
    ap.add_argument("--sprint-frames", type=int, default=4)
    ap.add_argument("--spacing", type=int, default=8, help="pixels between rays; must stay under half a brick's screen width "
                    "(LOD-matched bricks are 8-16 px), or the touched set flickers and reads as turnover")
    ap.add_argument("--keep", type=int, default=4, help="frames a brick stays resident unused: new = not touched in the last keep frames")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    global RAY_SPACING
    RAY_SPACING = args.spacing
    if args.synthetic:
        assets = synthetic_assets()
    else:
        coarse = [np.load(f) for f in args.assets.split(",")]
        occ = [np.load(f, mmap_mode="r") for f in args.occupancy.split(",")]
        assets = [Asset(c, args.assets_level, np.asarray(o), args.occupancy_level, args.voxel_world) for c, o in zip(coarse, occ)]
    fit = min(0.98 * min(TILE_WORLD / max(e[0], e[2]), LAYER_HEIGHT / e[1])
              for e in ((a.content_max - a.content_min + 1) * a.voxel_world for a in assets))
    s_max0 = assets[0].voxel_world * fit
    print(f"{len(assets)} assets, fit scale {fit:.4f}, largest level-0 instance voxel {s_max0:.4f} world units; level-0 reach "
          f"{s_max0 / PIXEL_ANGLE:.0f} units; rays every {RAY_SPACING} px; view distance {VIEW_DISTANCE:.0f}", flush=True)
    sea = Sea(assets, fit)
    cs = [float(v) for v in args.c.split(",")]
    for motion, speed, frames in (("walk", 2.0, args.walk_frames), ("sprint", 20.0, args.sprint_frames)):
        # Per key kind, the sets of the last `keep` frames: a brick is new when none of them touched it (a cache that keeps
        # bricks `keep` frames unused). The first frame is the settle: everything is new there.
        history = {}
        union = {c: set() for c in cs}

        def new_count(kind, s):
            past = history.setdefault(kind, [])
            seen = set().union(*past) if past else set()
            n = len(s - seen) if past else len(s)
            past.append(s)
            del past[:-args.keep]
            return n

        for f in range(frames):
            start = time.time()
            camera = CAMERA + np.array([0.0, 0.0, speed * f])
            p, t = march(sea, camera, TARGET - CAMERA)
            line = [f"{motion} frame {f}: {p.shape[0]} in-cloud samples ({time.time() - start:.0f} s)"]
            for c in cs:
                world, inst, shared = keys(sea, p, t, c, s_max0)
                ws = as_set(world)
                new = new_count(("world", c), ws)
                union[c] |= ws
                levels = np.bincount(world[:, 0], minlength=1) if world.size else np.zeros(1, int)
                line.append(f"  c={c}: world {len(ws)} bricks ({len(ws) * BRICK_BYTES / 2**20:.0f} MB) new {new} union "
                            f"{len(union[c])} ({len(union[c]) * BRICK_BYTES / 2**20:.0f} MB) by level {levels.tolist()}")
                if c == cs[0]:
                    si, ss = as_set(inst), as_set(shared)
                    line.append(f"  per-instance {len(si)} bricks (new {new_count('instance', si)}), shared asset bricks {len(ss)} "
                                f"(new {new_count('shared', ss)})")
            print("\n".join(line), flush=True)


if __name__ == "__main__":
    main()
