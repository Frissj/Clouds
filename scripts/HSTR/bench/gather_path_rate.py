"""Raster-entry batched gather vs the leanest DDA, offline (NVIDIA Warp, no Mogwai).

Question: does the dirty march's ~7-9 G loop iterations/s ceiling (direct_path_rate.py) come from its serial dependency chain
(DDA -> page table -> atlas -> T, one step at a time), and does it go if (1) a ray starts at its first non-empty fine brick - what
rasterising a compile-time hull of each asset's non-empty brick faces would give every pixel, 74% of the walk's steps being empty
bricks - and (2) its samples, whose positions entry + k step are known in advance, are fetched 8 at a time with no dependency
between them (page table + atlas loads unconditional, masked), then composited? A batch that finds no brick at all jumps with the
brick distance field. The entry pass stands in for the raster and is not timed.

Same atlas, view and rays as direct_path_rate.py (level-1 bench cloud, 207 MB packed fp16 (density, sun), 1024^2 perspective
rays, one tap). Kill line: 1M rays in <~0.7 ms at step 4 (the real ~2 samples a brick visit), coherent and shuffled.

    python scripts/HSTR/bench/gather_path_rate.py [camera distance, default 900]

Counters are taken in an untimed launch and time is GPU events: 1M same-address atomics alone cost ~1-2 ms (the error that made
direct_path_rate.py's "7-9 G iterations/s ceiling").
MEASURED (gather_path_rate2/3.log, RTX 4080 Laptop, 1M rays, ray setup floor 0.04 ms):
  distance 900 (16% hit): DDA + distance field 0.14 / 0.62 ms coherent / shuffled, gather 0.12 / 0.44 ms at step 4.
  distance 300 (all hit, 9.2M fine samples ~ the dirty frame's 10.2M): step 4, DDA 1.78 / 2.76 ms, raster entry + gather 8
  0.23 / 1.09 ms (40 G samples/s coherent); step 2, 1.86 / 3.2 vs 0.38 / 1.89 ms. Error against a quarter-voxel step of the same
  field is the same for both (step 4: mean 0.0027 vs 0.0031, p99 0.077 vs 0.078): the gather is exact, the step sets the error.
  Coherence matters 4.7x: dirty rays must stay in screen-tile order when compacted.
"""
import sys
import time

import numpy as np
import warp as wp

import direct_path_rate as dpr
import tile_transport_rate as ttr

BATCH = 8
vec8 = wp.types.vector(length=BATCH, dtype=float)


@wp.func
def ray_dir(p: int, width: int, height: int, pixelAngle: float, fwd: wp.vec3, right: wp.vec3, up: wp.vec3) -> wp.vec3:
    u = (float(p % width) + 0.5 - 0.5 * float(width)) * pixelAngle
    v = (float(p / width) + 0.5 - 0.5 * float(height)) * pixelAngle
    return wp.normalize(fwd + right * u + up * v)


@wp.func
def box_exit(o: wp.vec3, cam: wp.vec3, inv: wp.vec3, size: float) -> float:
    ea = wp.cw_mul(o - cam, inv)
    eb = wp.cw_mul(o + wp.vec3(size, size, size) - cam, inv)
    return wp.min(wp.min(wp.max(ea[0], eb[0]), wp.max(ea[1], eb[1])), wp.max(ea[2], eb[2]))


@wp.kernel
def entry_pass(table: wp.array3d(dtype=wp.int32), dist: wp.array3d(dtype=wp.int32), order: wp.array(dtype=wp.int32), width: int,
               height: int, pixelAngle: float, cam: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, entry: wp.array(dtype=float)):
    """The raster's answer: t at which the ray enters its first non-empty fine brick (1e30 if none)."""
    tid = wp.tid()
    d = ray_dir(order[tid], width, height, pixelAngle, fwd, right, up)
    nb = wp.vec3(float(table.shape[0]), float(table.shape[1]), float(table.shape[2])) * 8.0
    inv = wp.vec3(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])
    ta = wp.cw_mul(wp.vec3(0.0, 0.0, 0.0) - cam, inv)
    tb = wp.cw_mul(nb - cam, inv)
    t = wp.max(wp.max(wp.min(ta[0], tb[0]), wp.min(ta[1], tb[1])), wp.max(wp.min(ta[2], tb[2]), 0.0))
    tEnd = wp.min(wp.min(wp.max(ta[0], tb[0]), wp.max(ta[1], tb[1])), wp.max(ta[2], tb[2]))
    result = float(1.0e30)
    guard = int(0)
    while t < tEnd and guard < 4096:
        guard += 1
        x = cam + d * (t + 1.0e-3)
        bi = int(wp.floor(x[0] / 8.0))
        bj = int(wp.floor(x[1] / 8.0))
        bk = int(wp.floor(x[2] / 8.0))
        if bi < 0 or bj < 0 or bk < 0 or bi >= table.shape[0] or bj >= table.shape[1] or bk >= table.shape[2]:
            break
        if table[bi, bj, bk] >= 0:
            result = t
            break
        o = wp.vec3(float(bi), float(bj), float(bk)) * 8.0
        t = wp.max(box_exit(o, cam, inv, 8.0), t + float(wp.max(dist[bi, bj, bk] - 1, 0)) * 8.0)
    entry[tid] = result


@wp.func
def density_sun(table: wp.array3d(dtype=wp.int32), atlas: wp.array(dtype=dpr.vec2h), x: wp.vec3) -> wp.vec2:
    """Branch-free: a miss (outside, or an empty brick) loads brick 0's texel and is masked to zero."""
    bi = int(wp.floor(x[0] / 8.0))
    bj = int(wp.floor(x[1] / 8.0))
    bk = int(wp.floor(x[2] / 8.0))
    inside = bi >= 0 and bj >= 0 and bk >= 0 and bi < table.shape[0] and bj < table.shape[1] and bk < table.shape[2]
    e = table[wp.clamp(bi, 0, table.shape[0] - 1), wp.clamp(bj, 0, table.shape[1] - 1), wp.clamp(bk, 0, table.shape[2] - 1)]
    q = x - wp.vec3(float(bi), float(bj), float(bk)) * 8.0 + wp.vec3(0.5, 0.5, 0.5)
    i = wp.clamp(int(q[0]), 0, 8)
    j = wp.clamp(int(q[1]), 0, 8)
    k = wp.clamp(int(q[2]), 0, 8)
    v = atlas[wp.max(e, 0) * 1000 + (k * 10 + j) * 10 + i]
    m = float(0.0)
    if inside and e >= 0:
        m = 1.0
    return wp.vec2(float(v[0]) * m, float(v[1]) * m)


@wp.kernel
def gather(table: wp.array3d(dtype=wp.int32), dist: wp.array3d(dtype=wp.int32), atlas: wp.array(dtype=dpr.vec2h),
           order: wp.array(dtype=wp.int32), entry: wp.array(dtype=float), width: int, height: int, pixelAngle: float, cam: wp.vec3,
           fwd: wp.vec3, right: wp.vec3, up: wp.vec3, step: float, voxelWorld: float, out: wp.array(dtype=float),
           counts: wp.array(dtype=wp.int64)):
    tid = wp.tid()
    d = ray_dir(order[tid], width, height, pixelAngle, fwd, right, up)
    nb = wp.vec3(float(table.shape[0]), float(table.shape[1]), float(table.shape[2])) * 8.0
    inv = wp.vec3(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])
    ta = wp.cw_mul(wp.vec3(0.0, 0.0, 0.0) - cam, inv)
    tb = wp.cw_mul(nb - cam, inv)
    tEnd = wp.min(wp.min(wp.max(ta[0], tb[0]), wp.max(ta[1], tb[1])), wp.max(ta[2], tb[2]))
    T = float(1.0)
    L = float(0.0)
    samples = int(0)
    batches = int(0)
    # One global sample phase (samples at 0.5 step + k step from the camera), the first at or past the entry.
    tNext = wp.ceil((entry[tid] - 0.5 * step) / step) * step + 0.5 * step
    while tNext < tEnd and T > 1.0e-3 and batches < 1024:
        batches += 1
        sigma = vec8()
        sun = vec8()
        for k in range(BATCH):  # independent loads: nothing here depends on T
            ds = density_sun(table, atlas, cam + d * (tNext + float(k) * step))
            sigma[k] = ds[0]
            sun[k] = ds[1]
        hit = int(0)
        for k in range(BATCH):
            if sigma[k] > 0.0 and tNext + float(k) * step < tEnd:
                alpha = 1.0 - wp.exp(-sigma[k] * step * voxelWorld)
                L += T * alpha * sun[k]
                T *= 1.0 - alpha
                samples += 1
                hit = 1
        tNext += float(BATCH) * step
        if hit == 0:
            # A batch in a gap: if it ends in an empty brick, jump past it and every brick the distance field clears.
            x = cam + d * tNext
            bi = int(wp.floor(x[0] / 8.0))
            bj = int(wp.floor(x[1] / 8.0))
            bk = int(wp.floor(x[2] / 8.0))
            if bi >= 0 and bj >= 0 and bk >= 0 and bi < table.shape[0] and bj < table.shape[1] and bk < table.shape[2]:
                if table[bi, bj, bk] < 0:
                    o = wp.vec3(float(bi), float(bj), float(bk)) * 8.0
                    tJump = wp.max(box_exit(o, cam, inv, 8.0), tNext + float(wp.max(dist[bi, bj, bk] - 1, 0)) * 8.0)
                    tNext = tNext + wp.ceil(wp.max(tJump - tNext, 0.0) / step) * step
    out[tid] = T + L * 1.0e-9
    if counts.shape[0] > 0:  # counted in an untimed launch (see direct_path_rate.direct)
        wp.atomic_add(counts, 0, wp.int64(samples))
        wp.atomic_add(counts, 2, wp.int64(batches))


def main():
    vol = np.load(dpr.VOLUME)
    atlas, origins, index = ttr.build_atlas(vol)
    atlas_g = wp.array(atlas.view(np.float16).reshape(-1, 2), dtype=dpr.vec2h)
    table_g = wp.array(index, dtype=wp.int32)
    from scipy.ndimage import distance_transform_cdt
    dist_g = wp.array(distance_transform_cdt(index < 0, metric="chessboard").astype(np.int32), dtype=wp.int32)
    centre = np.array(vol.shape) / 2
    distance = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0  # 300: the cloud fills the view, most rays hit
    view = np.array([0.3, -0.25, 1.0])
    view /= np.linalg.norm(view)
    cam = centre - view * distance
    right = np.cross(view, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    width = height = 1024
    # 65 degrees across whatever the distance: a texel is a voxel at 900.
    geo = (width, height, 1.0 / 900.0, wp.vec3(*cam), wp.vec3(*view), wp.vec3(*right), wp.vec3(*up))
    rng = np.random.default_rng(0)
    orders = {"coherent": np.arange(width * height, dtype=np.int32),
              "shuffled": rng.permutation(width * height).astype(np.int32)}
    counts = wp.zeros(3, dtype=wp.int64)
    nocount = wp.zeros(0, dtype=wp.int64)
    out = wp.zeros(width * height, dtype=float)
    entry = wp.zeros(width * height, dtype=float)
    miss = wp.full(width * height, 1.0e30, dtype=float)
    print(f"atlas {len(origins)} bricks, {atlas.nbytes / 1e6:.0f} MB; GPU {wp.get_device().name}")
    for name, order in orders.items():
        order_g = wp.array(order, dtype=wp.int32)
        wp.launch(entry_pass, dim=width * height, inputs=[table_g, dist_g, order_g, *geo, entry])
        hits = int((entry.numpy() < 1e29).sum())
        wp.launch(dpr.direct, dim=width * height, inputs=[table_g, dist_g, atlas_g, order_g, *geo, 0.25, dpr.VOXEL_WORLD, 1, 1, out,
                                                          nocount])
        ref = out.numpy().copy()  # the same field (one tap) at a quarter-voxel step: both arms' error, not their difference
        for step in (4.0, 2.0):
            arms = {
                "floor (ray setup, no hits)": lambda c: wp.launch(gather, dim=width * height, inputs=[
                    table_g, dist_g, atlas_g, order_g, miss, *geo, step, dpr.VOXEL_WORLD, out, c]),
                "DDA + distance field": lambda c: wp.launch(dpr.direct, dim=width * height, inputs=[
                    table_g, dist_g, atlas_g, order_g, *geo, step, dpr.VOXEL_WORLD, 1, 1, out, c]),
                f"raster entry + gather {BATCH}": lambda c: wp.launch(gather, dim=width * height, inputs=[
                    table_g, dist_g, atlas_g, order_g, entry, *geo, step, dpr.VOXEL_WORLD, out, c]),
            }
            for arm, launch in arms.items():
                warm = time.perf_counter()
                while time.perf_counter() - warm < 2.0:  # hold the clocks up (see tile_transport_rate.py)
                    launch(nocount)
                    wp.synchronize()
                start, stop = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                wp.record_event(start)  # GPU time: Python's per-launch overhead must not count
                for _ in range(20):
                    launch(nocount)
                wp.record_event(stop)
                wp.synchronize()
                ms = wp.get_event_elapsed_time(start, stop) / 20
                counts.zero_()
                launch(counts)
                samples, _, iters = (int(c) for c in counts.numpy())
                dT = np.abs(out.numpy() - ref)
                check = f"; |T - T(step 1/4)| mean {dT.mean():.4f}, p99 {np.percentile(dT, 99):.4f}, max {dT.max():.3f}"
                print(f"{name}, step {step:.0f}, {arm:26s}: {ms:6.2f} ms for 1M rays ({hits / 1e6:.2f}M hit), {samples / 1e6:5.1f}M samples, "
                      f"{iters / 1e6:5.1f}M loop iterations (DDA bricks / batches), {samples / ms / 1e6:5.1f} G samples/s{check}")


if __name__ == "__main__":
    main()
