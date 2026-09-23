"""Object-order (shear-warp) push throughput on this GPU, offline (NVIDIA Warp, no Mogwai).

Question: how many fine-voxel x pixel contributions a second can a coherent front-to-back slice compositor push over the real
bench cloud, against the ~50M a full 4K frame of the sea holds (visible occupied cells at pixel-footprint LOD x non-empty share
x share in front of T = 1e-3)? A frame's object-order work fits ~0.9 ms only at ~60 G contributions/s or more.

The cloud is cumulus_1_size_1 at mip level 1 (see ray_jet_fit.py; built there) as a dense float16 array, one voxel a pixel.
Shear-warp: slices along the axis nearest the view, each thread one intermediate-image pixel walking the slices front to back,
bilinear in each slice at its sheared position (neighbouring threads read neighbouring voxels of one slice: coalesced), with a
per-voxel source channel (float16 x channels: 1 grey, 4 = rgb + pad), early out at T < 1e-3. Many views per launch (different
shears) fill the GPU. For contrast, the same volume ray marched per pixel from a perspective camera (trilinear, 1-voxel steps,
same early out). Both are best cases for their kind: dense array, no virtual addressing, no LOD, no sun lookups.

    python scripts/HSTR/bench/shear_warp_rate.py
"""
import os
import time

import numpy as np
import warp as wp

VOLUME = os.path.join(os.environ.get("TEMP", "."), "hstr_jet_level1.npy")
VOXEL_WORLD = 0.4455

wp.init()


@wp.func
def bilinear(vol: wp.array3d(dtype=wp.float16), k: int, x: float, y: float):
    nx = vol.shape[2]
    ny = vol.shape[1]
    fx = wp.floor(x)
    fy = wp.floor(y)
    i = int(fx)
    j = int(fy)
    if i < 0 or j < 0 or i + 1 >= nx or j + 1 >= ny:
        return float(0.0)
    tx = x - fx
    ty = y - fy
    a = float(vol[k, j, i]) * (1.0 - tx) + float(vol[k, j, i + 1]) * tx
    b = float(vol[k, j + 1, i]) * (1.0 - tx) + float(vol[k, j + 1, i + 1]) * tx
    return a * (1.0 - ty) + b * ty


@wp.kernel
def shear_warp(vol: wp.array3d(dtype=wp.float16), src: wp.array4d(dtype=wp.float16), shears: wp.array(dtype=wp.vec2),
               channels: int, voxel_world: float, out: wp.array2d(dtype=wp.vec4), counts: wp.array(dtype=wp.int64)):
    view, p = wp.tid()
    nx = vol.shape[2]
    ny = vol.shape[1]
    nz = vol.shape[0]
    px = p % nx
    py = p / nx
    s = shears[view]
    T = float(1.0)
    L = wp.vec3(0.0, 0.0, 0.0)
    samples = int(0)
    visited = int(0)
    for k in range(nz):
        x = float(px) + s[0] * float(k)
        y = float(py) + s[1] * float(k)
        sigma = bilinear(vol, k, x, y)
        visited += 1
        if sigma > 0.0:
            samples += 1
            a = 1.0 - wp.exp(-sigma * voxel_world)
            c = wp.vec3(0.0, 0.0, 0.0)
            for ch in range(channels):
                v = float(src[ch, k, int(y), int(x)])
                c[wp.min(ch, 2)] = v
            L = L + T * a * c
            T = T * (1.0 - a)
            if T < 1.0e-3:
                break
    out[view, p] = wp.vec4(L[0], L[1], L[2], T)
    wp.atomic_add(counts, 0, wp.int64(samples))
    wp.atomic_add(counts, 1, wp.int64(visited))


@wp.kernel
def ray_march(vol: wp.array3d(dtype=wp.float16), cams: wp.array(dtype=wp.vec3), width: int, height: int, fov: float,
              voxel_world: float, out: wp.array2d(dtype=wp.vec4), counts: wp.array(dtype=wp.int64)):
    view, p = wp.tid()
    nx = float(vol.shape[2])
    ny = float(vol.shape[1])
    nz = float(vol.shape[0])
    cam = cams[view]
    centre = wp.vec3(nx * 0.5, ny * 0.5, nz * 0.5)
    fwd = wp.normalize(centre - cam)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    u = (float(p % width) + 0.5) / float(width) * 2.0 - 1.0
    v = (float(p / width) + 0.5) / float(height) * 2.0 - 1.0
    t = wp.tan(fov * 0.5)
    d = wp.normalize(fwd + right * (u * t * float(width) / float(height)) + up * (v * t))
    # Clip to the volume's box first: the march is charged for the cloud, not the air in front of it.
    inv = wp.vec3(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])
    ta = wp.cw_mul(wp.vec3(1.0, 1.0, 1.0) - cam, inv)
    tb = wp.cw_mul(wp.vec3(nx - 1.0, ny - 1.0, nz - 1.0) - cam, inv)
    t0 = wp.max(wp.max(wp.min(ta[0], tb[0]), wp.min(ta[1], tb[1])), wp.max(wp.min(ta[2], tb[2]), 0.0))
    t1 = wp.min(wp.min(wp.max(ta[0], tb[0]), wp.max(ta[1], tb[1])), wp.max(ta[2], tb[2]))
    T = float(1.0)
    samples = int(0)
    visited = int(0)
    steps = int(wp.max(t1 - t0, 0.0))
    for step in range(steps):
        x = cam + d * (t0 + float(step) + 0.5)
        if x[0] >= 1.0 and x[1] >= 1.0 and x[2] >= 1.0 and x[0] < nx - 1.0 and x[1] < ny - 1.0 and x[2] < nz - 1.0:
            visited += 1
            i = int(x[0] - 0.5)
            j = int(x[1] - 0.5)
            k = int(x[2] - 0.5)
            fx = x[0] - 0.5 - float(i)
            fy = x[1] - 0.5 - float(j)
            fz = x[2] - 0.5 - float(k)
            c00 = float(vol[k, j, i]) * (1.0 - fx) + float(vol[k, j, i + 1]) * fx
            c10 = float(vol[k, j + 1, i]) * (1.0 - fx) + float(vol[k, j + 1, i + 1]) * fx
            c01 = float(vol[k + 1, j, i]) * (1.0 - fx) + float(vol[k + 1, j, i + 1]) * fx
            c11 = float(vol[k + 1, j + 1, i]) * (1.0 - fx) + float(vol[k + 1, j + 1, i + 1]) * fx
            sigma = (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz
            if sigma > 0.0:
                samples += 1
                T = T * wp.exp(-sigma * voxel_world)
                if T < 1.0e-3:
                    break
    out[view, p] = wp.vec4(0.0, 0.0, 0.0, T)
    wp.atomic_add(counts, 0, wp.int64(samples))
    wp.atomic_add(counts, 1, wp.int64(visited))


def timed(launch, repeats=5):
    launch()
    wp.synchronize()
    best = 1e9
    for _ in range(repeats):
        start = time.perf_counter()
        launch()
        wp.synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def densest_block(v, edge):
    """The edge^3 block (stride edge / 2) holding the most non-empty voxels: the cloud's core, where most samples composite."""
    best, at = -1, (0, 0, 0)
    for z in range(0, v.shape[0] - edge + 1, edge // 2):
        for y in range(0, v.shape[1] - edge + 1, edge // 2):
            for x in range(0, v.shape[2] - edge + 1, edge // 2):
                n = np.count_nonzero(v[z:z + edge, y:y + edge, x:x + edge])
                if n > best:
                    best, at = n, (z, y, x)
    z, y, x = at
    return np.ascontiguousarray(v[z:z + edge, y:y + edge, x:x + edge])


def main():
    import sys
    # Slice-major (z, y, x), x fastest: the threads of a slice read consecutive addresses.
    host = np.ascontiguousarray(np.load(VOLUME).astype(np.float16).transpose(2, 1, 0))
    if "--core" in sys.argv:
        host = densest_block(host, 192)
    nz, ny, nx = host.shape
    vol = wp.array(host, dtype=wp.float16)
    height = (0.3 + 0.7 * np.arange(ny) / ny).astype(np.float16)
    print(f"volume {host.shape} ({host.nbytes / 1e6:.0f} MB), GPU {wp.get_device().name}")
    rng = np.random.default_rng(0)
    views = 24
    shears = wp.array(rng.uniform(-0.6, 0.6, size=(views, 2)).astype(np.float32), dtype=wp.vec2)
    for channels in (1, 4):
        src = wp.array(np.broadcast_to(height[None, None, :, None], (channels, nz, ny, nx)).copy(), dtype=wp.float16)
        out = wp.zeros((views, nx * ny), dtype=wp.vec4)
        counts = wp.zeros(2, dtype=wp.int64)

        def launch():
            counts.zero_()
            wp.launch(shear_warp, dim=(views, nx * ny), inputs=[vol, src, shears, channels, VOXEL_WORLD, out, counts])

        seconds = timed(launch)
        n, visited = (int(c) for c in counts.numpy())
        print(f"shear-warp, {channels} source channel(s): {n / 1e6:.0f}M contributions (non-empty samples) in {seconds * 1e3:.2f} ms "
              f"= {n / seconds / 1e9:.1f} G/s; {visited / seconds / 1e9:.1f} G visited slice samples/s incl. empty ({n / visited:.0%} non-empty)")
        del src
    width, height_px = 960, 540
    cams = wp.array((np.array([nx, ny, nz]) / 2 + rng.normal(size=(views, 3)) * 0 + np.array([[np.cos(a), 0.07, np.sin(a)] for a in rng.uniform(0, 6.28, views)]) * 1.3 * max(nx, ny, nz)).astype(np.float32), dtype=wp.vec3)
    out = wp.zeros((views, width * height_px), dtype=wp.vec4)
    counts = wp.zeros(2, dtype=wp.int64)

    def march():
        counts.zero_()
        wp.launch(ray_march, dim=(views, width * height_px), inputs=[vol, cams, width, height_px, 0.9, VOXEL_WORLD, out, counts])

    seconds = timed(march)
    n, visited = (int(c) for c in counts.numpy())
    print(f"perspective ray march (dense, trilinear): {n / 1e6:.0f}M non-empty samples in {seconds * 1e3:.2f} ms = {n / seconds / 1e9:.1f} G/s; "
          f"{visited / seconds / 1e9:.1f} G visited samples/s ({n / max(visited, 1):.0%} non-empty)")


if __name__ == "__main__":
    main()
