"""Form R gate R0, scene half (offline, numpy): the shared occupancy hulls, the sea's instances and the walk camera's lattice rays,
written to one .npz for form_r_trace.py, which times the hardware ray queries on the GPU.

LIT_VOLUME.md section 6c. A hull is the closed boundary of the octant cells (4 voxels at the level) of one asset's decoded level
that hold any density, dilated by one cell. Vertices are in source voxels (voxel k centred at k, as lit_volume_budget's sampling),
so one hull per (asset, level) serves every instance: each instance's TLAS transform is CloudSea::makeTile's scaled signed
permutation. Triangles wind counter-clockwise seen from outside (right-handed); form_r_trace.py tries both facing conventions and
keeps the one under which no camera ray's first hit is a back face.

Per instance, one level: --level-rule far takes the footprint level of the instance's farthest box corner (the coarsest level any
sample in it reads: what an exact renderer hull must use), near that of its nearest point (finer, heavier hulls: a pessimistic
trace cost). The hull level is the nearest available one on the conservative side; levels without a volume fall back to the
nearest decoded one, so near instances wanting level 0 get the level-1 hull unless a0_L0.npy etc. exist.

Rays: one per --spacing pixels of the 4K walk start camera (the beam lattice is every 4 px), in 8 x 8 tiles of lattice points,
from the camera to the cloud slab's far plane or the near view distance. --check rays (seeded) get a CPU reference first entry:
samples every half cell along each instance box they cross, giving the interval [lo, hi] the first entry must lie in.

    python form_r_scene.py --volumes DIR --out form_r_walk.npz          # DIR holds a{i}_L{l}.npy from hstrlib_dense
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lit_volume_budget import (CAMERA, LAYER_HEIGHT, ORIGIN, PIXEL_ANGLE, TILE_WORLD, VIEW_DISTANCE, Asset, make_tile,  # noqa: E402
                               rays)

CELL = 4  # voxels per hull cell at its level (an octant of an 8^3 brick)


def hull_cells(volume):
    """Occupied octant cells of a (z, y, x) volume, dilated by one cell (3 x 3 x 3)."""
    pad = [(0, -(-d // CELL) * CELL - d) for d in volume.shape]
    v = np.pad(np.asarray(volume) > 0, pad)
    occ = v.reshape(v.shape[0] // CELL, CELL, v.shape[1] // CELL, CELL, v.shape[2] // CELL, CELL).any(axis=(1, 3, 5))
    occ = np.pad(occ, 1)  # room for the dilation (np.roll wraps only these zero planes); cell indices shift by one
    for a in range(3):
        lo = np.roll(occ, 1, axis=a)
        hi = np.roll(occ, -1, axis=a)
        occ = occ | lo | hi
    return occ


def hull_mesh(occ, level):
    """Boundary quads of the cell set as triangles: vertices (n, 3) x, y, z in source voxels, indices (m, 3) uint32."""
    size = CELL * 2 ** level
    p = np.pad(occ, 1).astype(np.int8)  # p[z, y, x]; occ cell (z, y, x) is p[z + 1, y + 1, x + 1]
    tris = []
    # axis in (z, y, x) array order -> (a, b, c) as x, y, z indices with b, c cyclic after a (e_b x e_c = e_a)
    for arr_axis, a in ((2, 0), (1, 1), (0, 2)):
        b, c = (a + 1) % 3, (a + 2) % 3
        d = np.diff(p, axis=arr_axis)  # +1: empty -> occupied going up (outward -a), -1: occupied -> empty (outward +a)
        for sign in (-1, 1):
            idx = np.argwhere(d == sign)  # (z, y, x) in p-diff space
            if idx.size == 0:
                continue
            xyz = idx[:, ::-1].copy()  # x, y, z
            # The face plane lies at the upper side of the lower cell: diff index i compares p[i] and p[i + 1], i.e. occ cells
            # i - 1 and i, plane at corner i (occ-cell corner coordinates, before the pad shift is removed below).
            base = xyz.copy()
            base[:, b] -= 1
            base[:, c] -= 1
            q = np.zeros((len(xyz), 4, 3), np.int64)
            for k, (du, dv) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                q[:, k] = base
                q[:, k, b] += du
                q[:, k, c] += dv
            outward_plus = sign == -1
            order = [(0, 1, 2), (0, 2, 3)] if outward_plus else [(0, 2, 1), (0, 3, 2)]
            for t in order:
                tris.append(np.stack([q[:, t[0]], q[:, t[1]], q[:, t[2]]], 1))
    allc = np.concatenate(tris).reshape(-1, 3)
    uniq, inverse = np.unique(allc, axis=0, return_inverse=True)
    # occ cell j (after hull_cells' pad) starts at source voxel (j - 1) * size - 0.5
    verts = ((uniq - 1) * size - 0.5).astype(np.float32)
    return verts, inverse.reshape(-1, 3).astype(np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", required=True, help="directory holding a{i}_L{l}.npy (hstrlib_dense, float16 (z, y, x))")
    ap.add_argument("--occupancy-level", type=int, default=2, help="level whose volume sets the content bounds (as S1)")
    ap.add_argument("--voxel-world", type=float, default=0.25)
    ap.add_argument("--spacing", type=int, default=4)
    ap.add_argument("--level-rule", choices=("far", "near"), default="far")
    ap.add_argument("--check", type=int, default=2048, help="rays with a CPU reference first entry")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    start = time.time()

    files = {}
    for f in glob.glob(os.path.join(args.volumes, "a*_L*.npy")):
        m = re.fullmatch(r"a(\d+)_L(\d+)\.npy", os.path.basename(f))
        if m:
            files.setdefault(int(m.group(1)), {})[int(m.group(2))] = f
    n_assets = max(files) + 1
    assert all(i in files and args.occupancy_level in files[i] for i in range(n_assets)), "every asset needs its occupancy level"
    assets = [Asset(np.zeros((1, 1, 1)), 0, np.load(files[i][args.occupancy_level], mmap_mode="r"), args.occupancy_level,
                    args.voxel_world) for i in range(n_assets)]
    fit = min(0.98 * min(TILE_WORLD / max(e[0], e[2]), LAYER_HEIGHT / e[1])
              for e in ((a.content_max - a.content_min + 1) * a.voxel_world for a in assets))

    hulls = {}  # (asset, level) -> (occ cells, verts, indices)

    def hull(i, level):
        if (i, level) not in hulls:
            occ = hull_cells(np.load(files[i][level], mmap_mode="r"))
            verts, idx = hull_mesh(occ, level)
            hulls[(i, level)] = (occ, verts, idx)
            print(f"  hull a{i} L{level}: {len(idx)} triangles, {len(verts)} vertices, {100 * occ.mean():.1f}% cells", flush=True)
        return hulls[(i, level)]

    # Instances: both layers' occupied tiles within the near view distance.
    inst = []
    for layer in range(2):
        for tx in range(-9, 10):
            for tz in range(-9, 10):
                occupied, a, s, lin, off = make_tile(tx, tz, layer, assets, fit)
                if not occupied:
                    continue
                m = np.linalg.inv(lin.T)  # world = off + m @ (src - content_min)
                lo = assets[a].content_min - 0.5 - CELL * 2 ** 4  # the coarsest hull's dilation margin, generously
                hi = assets[a].content_max + 0.5 + CELL * 2 ** 4
                box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                w = off + (box - assets[a].content_min) @ m.T
                wlo, whi = w.min(0), w.max(0)
                near = np.linalg.norm(np.clip(CAMERA, wlo, whi) - CAMERA)
                if near > VIEW_DISTANCE:
                    continue
                far = min(np.linalg.norm(w - CAMERA, axis=1).max(), VIEW_DISTANCE)
                d = far if args.level_rule == "far" else max(near, 1.0)
                need = int(np.floor(np.log2(max(d * PIXEL_ANGLE / s, 1.0))))
                have = sorted(files[a])
                if args.level_rule == "far":
                    level = min([l for l in have if l >= need], default=max(have))
                else:
                    level = max([l for l in have if l <= need], default=min(have))
                inst.append(dict(layer=layer, asset=a, level=level, need=need, s=s, lin=lin, off=off,
                                 xform=np.concatenate([m, (off - m @ assets[a].content_min)[:, None]], 1)))
    print(f"{n_assets} assets, fit {fit:.4f}, {len(inst)} instances within {VIEW_DISTANCE:.0f}; levels used "
          f"{np.unique([(i['need'], i['level']) for i in inst], axis=0, return_counts=True)}", flush=True)
    keys = sorted({(i["asset"], i["level"]) for i in inst})
    for k in keys:
        hull(*k)

    # Lattice rays in 8 x 8 tiles of lattice points.
    d = rays(CAMERA, args.spacing)
    ny, nx = -(-2160 // args.spacing), -(-3840 // args.spacing)
    gy, gx = np.divmod(np.arange(ny * nx), nx)
    order = np.lexsort((gx % 8, gy % 8, gx // 8, gy // 8))
    d = d[order]
    with np.errstate(divide="ignore"):
        t_leave = (ORIGIN[1] - CAMERA[1]) / d[:, 1]
        t_enter = (ORIGIN[1] + LAYER_HEIGHT - CAMERA[1]) / d[:, 1]
    keep = (d[:, 1] < 0) & (t_enter < VIEW_DISTANCE)
    d, t_leave = d[keep], np.minimum(t_leave[keep], VIEW_DISTANCE)
    ray = np.zeros((len(d), 8), np.float32)
    ray[:, 0:3] = CAMERA
    ray[:, 3] = t_leave
    ray[:, 4:7] = d
    ray[:, 7] = 0.0
    print(f"{len(ray)} lattice rays (every {args.spacing} px) reach the slab", flush=True)

    # CPU reference first entry for --check rays.
    rng = np.random.default_rng(7)
    chk = np.sort(rng.choice(len(ray), min(args.check, len(ray)), replace=False))
    lo_ref = np.full(len(chk), np.inf)
    hi_ref = np.full(len(chk), np.inf)
    faces = np.zeros(len(chk), np.int64)
    o, dd, tmax = ray[chk, 0:3].astype(np.float64), ray[chk, 4:7].astype(np.float64), ray[chk, 3].astype(np.float64)
    for it in inst:
        occ, _, _ = hulls[(it["asset"], it["level"])]
        size = CELL * 2 ** it["level"]
        cmin = assets[it["asset"]].content_min
        # box of the hull's cells in world: corners of the occ grid
        n = np.array(occ.shape[::-1])
        box = np.array([[x, y, z] for x in (0, n[0]) for y in (0, n[1]) for z in (0, n[2])])
        src = (box - 1) * size - 0.5
        w = it["off"] + (src - cmin) @ np.linalg.inv(it["lin"].T).T
        wlo, whi = w.min(0), w.max(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (wlo - o) / dd, (whi - o) / dd
        a = np.maximum(np.nanmax(np.minimum(ta, tb), 1), 0.0)
        b = np.minimum(np.nanmin(np.maximum(ta, tb), 1), tmax)
        hit = np.nonzero(a < b)[0]
        if hit.size == 0:
            continue
        h = 0.5 * size * it["s"]
        cnt = np.ceil((b[hit] - a[hit]) / h).astype(np.int64) + 1
        r = np.repeat(hit, cnt)
        j = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        t = np.minimum(a[r] + j * h, b[r])
        p = o[r] + t[:, None] * dd[r]
        q = (p - it["off"]) @ it["lin"] + cmin
        c = np.floor((q + 0.5) / size).astype(np.int64) + 1
        ok = np.all((c >= 0) & (c < n), 1)
        filled = np.zeros(len(t), bool)
        filled[ok] = occ[c[ok, 2], c[ok, 1], c[ok, 0]]
        # Hull faces this ray crosses in this instance (filled changes, plus filled at the box's ends): the hits the GPU's
        # counting loop takes, to within the half-cell sampling.
        same = r[1:] == r[:-1]
        np.add.at(faces, r[1:][same & (filled[1:] != filled[:-1])], 1)
        np.add.at(faces, r[(j == 0) & filled], 1)
        np.add.at(faces, r[np.r_[~same, True] & filled], 1)
        first = np.full(len(chk), np.iinfo(np.int64).max)
        np.minimum.at(first, r[filled], np.nonzero(filled)[0])
        got = np.nonzero(first < np.iinfo(np.int64).max)[0]
        k = first[got]
        hi_k = t[k]
        lo_k = np.where(j[k] > 0, t[np.maximum(k - 1, 0)], t[k] - h)
        lo_ref[got] = np.minimum(lo_ref[got], lo_k)
        hi_ref[got] = np.minimum(hi_ref[got], hi_k)
    print(f"CPU reference: {np.isfinite(hi_ref).sum()} of {len(chk)} check rays enter a hull; hull faces crossed a check ray "
          f"mean {faces.mean():.1f}, p50 / p90 / max {np.percentile(faces, 50):.0f} / {np.percentile(faces, 90):.0f} / "
          f"{faces.max()} ({time.time() - start:.0f} s)", flush=True)

    out = dict(rays=ray, check_idx=chk.astype(np.uint32), check_lo=lo_ref, check_hi=hi_ref, check_faces=faces,
               inst_xform=np.array([i["xform"] for i in inst], np.float32),
               inst_blas=np.array([keys.index((i["asset"], i["level"])) for i in inst], np.uint32),
               inst_level=np.array([i["level"] for i in inst]), inst_need=np.array([i["need"] for i in inst]),
               inst_layer=np.array([i["layer"] for i in inst]), blas_keys=np.array(keys),
               meta=np.array([args.spacing, args.level_rule == "far"]))
    for k, key in enumerate(keys):
        _, verts, idx = hulls[key]
        out[f"blas{k}_v"] = verts
        out[f"blas{k}_i"] = idx
    np.savez(args.out, **out)
    print(f"wrote {args.out}: {len(keys)} hulls, {sum(len(hulls[k][2]) for k in keys)} triangles, {len(inst)} instances, "
          f"{len(ray)} rays ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
