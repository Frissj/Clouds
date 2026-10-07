"""How many volumetric splats a 4K frame can composite in the frame budget, offline (NVIDIA Warp, no Mogwai).

Question: a splat representation of the cloud sea (Gaussians fitted per library asset, lighting baked into each) has no dirty set
and no parallax barrier, but it renders every pixel of every frame from scratch. Before fitting anything: what does a standard
tile rasterizer (3D Gaussian splatting's pipeline) cost at 3840 x 2160 on this GPU, for how many splats, of what size?

Pipeline, all GPU, timed with GPU events over 10 frames: project (radius 3 sigma, tile rectangle, tile count), exclusive scan,
emit one (tile << 16 | depth16, splat) pair per covered 16 x 16 tile, radix sort, tile ranges, then one thread a pixel compositing
its tile's splats front to back (alpha = opacity exp(-d^2 / 2 sigma^2), skipped under 1/255, stop at T < 1e-3). Splats read from
global memory, not staged through shared memory as the CUDA rasterizers do - so a lower bound on speed, not the best case.

Synthetic sea: clumps of splats (clouds) below the horizon (rows 0.3 H - H), depths log-uniform 256 - 2048 world units (where
repairdist1 put the dirty march's work), world sigma SIGMA (a source voxel is ~0.86 units; a 4K pixel ~0.37 units at 1000),
opacity 0.05 - 0.3. Counters (evaluations, pixels saturated) come from an untimed launch.

    python scripts/HSTR/bench/splat_rate.py [N in millions, comma list, default 1,2,4,8] [sigma list, default 1,2.5,6]

Kill line: >= 2M splats of sigma ~2.5 units (a few source voxels) in <= 1.5 ms. The visible sea's residency wants ~243k bricks of
8^3 voxels (124M values): 2M splats is already ~60 voxels a splat.

MEASURED (RTX 4080 Laptop; frame / project + sort / raster ms; evaluations per covered pixel, of them over 1/255):
  opacity 0.05-0.3 (splatrate2): sigma 1: 0.5M 8.2 / 0.7 / 7.5 (130, 37), 1M 16.2 / 1.7 / 14.5 (248, 72), 2M 26.5 / 3.8 / 22.7
    (322, 111), 4M 33.8 / 7.1 / 26.7; sigma 2.5: 0.5M 19.3 / 3.7 / 15.6 (233, 114), 1M 22.3, 2M 32.3 / 16.0 / 16.3 (229, 119),
    4M 46.3 / 30.0 / 16.3.
  opacity 0.5-0.95 (splatrate3, dense interiors): sigma 1: 0.5M 5.8 / 0.7 / 5.0 (67, 32), 2M 8.7 / 3.8 / 4.9, 4M 12.4 / 7.2 / 5.2;
    sigma 2.5: 0.5M 7.1 / 3.6 / 3.5 (49, 36), 1M 11.0, 2M 20.1 / 16.1 / 4.0 (48, 36); 4M's sort timing exceeds its frame (paging).
  The kill line fails ~13x (2M sigma 2.5: 20-32 ms). A translucent volume needs ~35-120 splats over 1/255 a pixel to saturate, at
  every one of 8.3M pixels every frame, and the per-tile sort of 15-116M pairs costs as much again. Even granting a shared-memory
  CUDA rasterizer and CUB sort 3-4x this, 2M splats is ~2.5-8 ms before any lighting or quality question - no better than the
  march's ~5.4 ms moving frame, and it gives up the parked frame's reuse (1.45 ms). Closed for this sea at 4K.
  splatrate1's negative evaluation counts were int32 overflow (now float sums); its times match splatrate2.
"""
import math
import os
import sys

import numpy as np
import warp as wp

W, H, TILE = 3840, 2160, 16
TX, TY = W // TILE, (H + TILE - 1) // TILE
PIXEL_ANGLE = 0.80 / H  # 46-degree vertical field
# HSTR_SPLAT_OPACITY lo,hi: the splats' peak opacity range (default 0.05,0.3: thin rims and haze; dense interiors are near opaque).
OPACITY = [float(v) for v in os.environ.get("HSTR_SPLAT_OPACITY", "0.05,0.3").split(",")]


@wp.kernel
def project(mean: wp.array(dtype=wp.vec2), sigma: wp.array(dtype=float), counts: wp.array(dtype=int), rects: wp.array(dtype=wp.vec4i)):
    i = wp.tid()
    m = mean[i]
    r = 3.0 * sigma[i]
    x0 = wp.max(int(wp.floor((m[0] - r) / 16.0)), 0)
    x1 = wp.min(int(wp.floor((m[0] + r) / 16.0)), 239)
    y0 = wp.max(int(wp.floor((m[1] - r) / 16.0)), 0)
    y1 = wp.min(int(wp.floor((m[1] + r) / 16.0)), 134)
    n = 0
    if x1 >= x0 and y1 >= y0:
        n = (x1 - x0 + 1) * (y1 - y0 + 1)
    counts[i] = n
    rects[i] = wp.vec4i(x0, y0, x1, y1)


@wp.kernel
def emit(rects: wp.array(dtype=wp.vec4i), offsets: wp.array(dtype=int), depth: wp.array(dtype=float),
         keys: wp.array(dtype=wp.uint32), values: wp.array(dtype=int)):
    i = wp.tid()
    rc = rects[i]
    o = offsets[i]
    d = wp.uint32(wp.clamp(depth[i] / 4096.0, 0.0, 1.0) * 65535.0)
    for y in range(rc[1], rc[3] + 1):
        for x in range(rc[0], rc[2] + 1):
            keys[o] = (wp.uint32(y * 240 + x) << wp.uint32(16)) | d
            values[o] = i
            o += 1


@wp.kernel
def ranges(keys: wp.array(dtype=wp.uint32), total: int, start: wp.array(dtype=int), end: wp.array(dtype=int)):
    k = wp.tid()
    tile = int(keys[k] >> wp.uint32(16))
    if k == 0 or int(keys[k - 1] >> wp.uint32(16)) != tile:
        start[tile] = k
    if k == total - 1 or int(keys[k + 1] >> wp.uint32(16)) != tile:
        end[tile] = k + 1


@wp.kernel
def raster(values: wp.array(dtype=int), start: wp.array(dtype=int), end: wp.array(dtype=int), mean: wp.array(dtype=wp.vec2),
           sigma: wp.array(dtype=float), opacity: wp.array(dtype=float), color: wp.array(dtype=wp.vec3),
           image: wp.array(dtype=wp.vec4), evaluations: wp.array(dtype=float), count: int):
    p = wp.tid()
    px = float(p % 3840) + 0.5
    py = float(p // 3840) + 0.5
    tile = (p // 3840 // 16) * 240 + (p % 3840) // 16
    T = float(1.0)
    c = wp.vec3(0.0)
    n = float(0.0)
    hits = float(0.0)
    for k in range(start[tile], end[tile]):
        i = values[k]
        m = mean[i]
        s = sigma[i]
        dx = px - m[0]
        dy = py - m[1]
        a = opacity[i] * wp.exp(-0.5 * (dx * dx + dy * dy) / (s * s))
        n += 1.0
        if a >= 1.0 / 255.0:
            hits += 1.0
            c += T * a * color[i]
            T *= 1.0 - a
            if T < 1.0e-3:
                break
    image[p] = wp.vec4(c[0], c[1], c[2], T)
    if count != 0:
        # Float sums: 8.3M pixels x ~200 evaluations overflow an int32 (splatrate1's negative counts).
        wp.atomic_add(evaluations, 0, n)
        wp.atomic_add(evaluations, 3, hits)
        if T < 1.0e-3:
            wp.atomic_add(evaluations, 1, 1.0)
        if T < 0.999:
            wp.atomic_add(evaluations, 2, 1.0)


def sea(n, sigma_world, rng):
    clumps = max(n // 2000, 1)
    cx = rng.uniform(0, W, clumps)
    cy = rng.uniform(0.3 * H, H, clumps)
    cd = np.exp(rng.uniform(math.log(256.0), math.log(2048.0), clumps))
    which = rng.integers(0, clumps, n)
    d = cd[which] * np.exp(rng.normal(0.0, 0.05, n))
    spread = 40.0 / (cd[which] * PIXEL_ANGLE)  # A clump ~40 world units across.
    x = cx[which] + rng.normal(0.0, 1.0, n) * spread
    y = cy[which] + rng.normal(0.0, 0.5, n) * spread
    s = sigma_world / (d * PIXEL_ANGLE)
    order = np.argsort(which, kind="stable")  # Splats of one cloud stored together, as a per-asset fit would be.
    return (np.stack([x, y], 1)[order].astype(np.float32), s[order].astype(np.float32), d[order].astype(np.float32),
            rng.uniform(*OPACITY, n)[order].astype(np.float32), rng.uniform(0.2, 1.0, (n, 3))[order].astype(np.float32))


def run(n, sigma_world):
    rng = np.random.default_rng(1)
    m, s, d, o, c = sea(n, sigma_world, rng)
    mean, sigma, depth = wp.array(m, dtype=wp.vec2), wp.array(s, dtype=float), wp.array(d, dtype=float)
    opacity, color = wp.array(o, dtype=float), wp.array(c, dtype=wp.vec3)
    counts, offsets = wp.zeros(n, dtype=int), wp.zeros(n, dtype=int)
    rects = wp.zeros(n, dtype=wp.vec4i)
    wp.launch(project, n, inputs=[mean, sigma, counts, rects])
    wp.utils.array_scan(counts, offsets, inclusive=False)
    total = int(offsets.numpy()[-1] + counts.numpy()[-1])
    keys, values = wp.zeros(2 * total, dtype=wp.uint32), wp.zeros(2 * total, dtype=int)
    start, end = wp.zeros(TX * TY, dtype=int), wp.zeros(TX * TY, dtype=int)
    image = wp.zeros(W * H, dtype=wp.vec4)
    evaluations = wp.zeros(4, dtype=float)

    def frame(count):
        wp.launch(project, n, inputs=[mean, sigma, counts, rects])
        wp.utils.array_scan(counts, offsets, inclusive=False)
        wp.launch(emit, n, inputs=[rects, offsets, depth, keys, values])
        wp.utils.radix_sort_pairs(keys, values, total)
        start.zero_()
        end.zero_()
        wp.launch(ranges, total, inputs=[keys, total, start, end])
        wp.launch(raster, W * H, inputs=[values, start, end, mean, sigma, opacity, color, image, evaluations, count])

    frame(0)
    wp.synchronize()
    stages = {}
    for name, fn in (("sort", lambda: (wp.launch(project, n, inputs=[mean, sigma, counts, rects]),
                                       wp.utils.array_scan(counts, offsets, inclusive=False),
                                       wp.launch(emit, n, inputs=[rects, offsets, depth, keys, values]),
                                       wp.utils.radix_sort_pairs(keys, values, total))),
                     ("frame", lambda: frame(0))):
        a, b = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        wp.record_event(a)
        for _ in range(10):
            fn()
        wp.record_event(b)
        wp.synchronize()
        stages[name] = wp.get_event_elapsed_time(a, b) / 10
    evaluations.zero_()
    frame(1)
    e = evaluations.numpy()
    print(f"SPLAT N {n / 1e6:.1f}M sigma {sigma_world:.1f}: frame {stages['frame']:.2f} ms (project+sort {stages['sort']:.2f}, "
          f"raster {stages['frame'] - stages['sort']:.2f}); {total / n:.1f} tiles a splat, {total / 1e6:.0f}M pairs; "
          f"evaluations {e[0] / (W * H):.1f} a pixel, {e[0] / max(e[2], 1):.1f} a covered pixel, of them over 1/255 "
          f"{e[3] / max(e[2], 1):.1f}; covered {100 * e[2] / (W * H):.1f}%, "
          f"opaque {100 * e[1] / (W * H):.1f}%", flush=True)


if __name__ == "__main__":
    wp.init()
    ns = [float(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,2,4,8").split(",")]
    sigmas = [float(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,2.5,6").split(",")]
    for sg in sigmas:
        for nm in ns:
            run(int(nm * 1e6), sg)
