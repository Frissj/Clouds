"""Brick x tile cooperative transport throughput on this GPU, offline (NVIDIA Warp native CUDA, no Mogwai).

Question: once a workgroup owns a BeamTile and walks its front-to-back fine bricks, staging each one into shared memory for all
its rays, how many exact march steps a second does the GPU sustain - against the shipping dirty march's ~7.3 G steps/s, and the
~50-70 G steps/s a sub-2 ms frame needs (26.8M dirty chord steps)?

One persistent workgroup per tile (rays' T and L stay in registers across its bricks). Per brick: resolve it, stage its 10^3
(8^3 core + apron) density and sun visibility - fp16 pairs in the atlas (4 KB), converted to float2 in shared memory - barrier,
every live ray marches its chord through the brick from shared memory (trilinear density and sun, exp, composite), barrier, and
the tile stops once every ray is opaque. The atlas is the bench cloud at mip 1 (every non-empty brick, far bigger than L2, so
stages come from DRAM as they would); the sun channel is transmittance straight down (a stand-in with the right storage and
access, not the right lighting). Tiles are a camera D voxels out with a texel of one voxel at the cloud (share8's measurement),
their brick lists the union of the bricks their corner and centre rays cross, sorted by depth.

Arms (each one question): staged / resident (the first brick stays in shared memory: the inner loop's ceiling) / direct (the
same brick lists and samples, texels read from the atlas through L1 / L2 with no staging or barriers: staging's one controlled
difference), tile of 64 / 128 / 256 rays, step 1 / 0.5 voxels (samples per stage). Reports G steps/s, samples per stage and
staging GB/s.

The first version timed one same-address atomicAdd per thread (~1M serialised atomics a launch) with perf_counter, as
direct_path_rate.py did; its numbers are void. Counting now runs in an untimed launch and the timing is GPU events.
MEASURED (tile_transport_rate2, RTX 4080 Laptop, 51,762 bricks / 207 MB, 1024^2): G steps/s staged / resident / direct -
tile 64: step 1 8.0 / 32.2 / 15.8, step 0.5 13.5 / 35.7 / 17.8; tile 128: 10.1 / 28.0 / 14.2, 15.6 / 30.4 / 15.5; tile 256:
9.9 / 21.5 / 13.6, 14.5 / 24.3 / 15.1. Staging runs at 205-334 GB/s for 96-282 steps a stage: it is bandwidth-bound, and direct
reads of the very same bricks match or beat it in every configuration. The resident ceiling (no DRAM at all) is only ~2x direct.
Cooperative brick x tile transport does not pay on this GPU, now on clean timing: a tile's rays do not share enough samples per
brick to amortise staging it, and L1 / L2 already give per-ray reads most of the reuse.

    python scripts/HSTR/bench/tile_transport_rate.py
"""
import os
import time

import numpy as np
import warp as wp

VOLUME = os.path.join(os.environ.get("TEMP", "."), "hstr_jet_level1.npy")  # Built by ray_jet_fit.py.
VOXEL_WORLD = 0.4455
wp.init()

SNIPPET = r"""
    __shared__ float2 stage[1000];
    // mode 0 staged, 1 resident (the first brick stays staged: the inner loop's ceiling), 2 direct (no staging or barriers: every
    // lane reads its texels from the atlas through L1 / L2, the same bricks and samples - the staging's one controlled difference).
    #define H2F(h) (((h) & 0x7C00u) ? __uint_as_float(((((h) >> 10) & 31u) + 112u) << 23 | ((h) & 0x3FFu) << 13) : 0.f)
    #define TEX(q) (mode == 2 ? make_float2(H2F(atlas[b * 1000 + (q)] & 0xFFFFu), H2F(atlas[b * 1000 + (q)] >> 16)) : stage[q])
    const int lane = tid % block;
    const int tile = tid / block;
    const int tx = tile % tilesX, ty = tile / tilesX;
    const int px = tx * tileW + lane % tileW, py = ty * (block / tileW) + lane / tileW;
    const float u = ((float)px + 0.5f - 0.5f * (float)width) * pixelAngle;
    const float v = ((float)py + 0.5f - 0.5f * (float)height) * pixelAngle;
    float dx = fwd[0] + u * right[0] + v * up[0];
    float dy = fwd[1] + u * right[1] + v * up[1];
    float dz = fwd[2] + u * right[2] + v * up[2];
    const float inv = rsqrtf(dx * dx + dy * dy + dz * dz);
    dx *= inv; dy *= inv; dz *= inv;
    float T = 1.f, L = 0.f;
    int samples = 0, stages = 0;
    bool live = true;
    const int first = listStart[tile], count = listCount[tile];
    for (int k = 0; k < count; ++k)
    {
        const int b = listBricks[first + k];
        if (mode == 0 || (mode == 1 && k == 0))
        {
            for (int i = lane; i < 1000; i += block)
            {
                const unsigned int packed = atlas[b * 1000 + i];
                const unsigned int lo = packed & 0xFFFFu, hi = packed >> 16;
                const float d = (lo & 0x7C00u) ? __uint_as_float((((lo >> 10) & 31u) + 112u) << 23 | (lo & 0x3FFu) << 13) : 0.f;
                const float s = (hi & 0x7C00u) ? __uint_as_float((((hi >> 10) & 31u) + 112u) << 23 | (hi & 0x3FFu) << 13) : 0.f;
                stage[i] = make_float2(d, s);
            }
            ++stages;
        }
        if (mode != 2)
            __syncthreads();
        if (live)
        {
            // The brick's core box [o, o + 8) in voxels; the stage holds [o - 1, o + 9).
            const float ox = (float)brickOrigin[b][0], oy = (float)brickOrigin[b][1], oz = (float)brickOrigin[b][2];
            const float ix = 1.f / dx, iy = 1.f / dy, iz = 1.f / dz;
            const float ax = (ox - cam[0]) * ix, bx = (ox + 8.f - cam[0]) * ix;
            const float ay = (oy - cam[1]) * iy, by = (oy + 8.f - cam[1]) * iy;
            const float az = (oz - cam[2]) * iz, bz = (oz + 8.f - cam[2]) * iz;
            const float t0 = fmaxf(fmaxf(fminf(ax, bx), fminf(ay, by)), fmaxf(fminf(az, bz), 0.f));
            const float t1 = fminf(fminf(fmaxf(ax, bx), fmaxf(ay, by)), fmaxf(az, bz));
            for (float t = t0 + 0.5f * step; t < t1; t += step)
            {
                const float x = cam[0] + t * dx - ox + 0.5f, y = cam[1] + t * dy - oy + 0.5f, z = cam[2] + t * dz - oz + 0.5f;
                const float fx = floorf(x), fy = floorf(y), fz = floorf(z);
                const int i = min(max((int)fx, 0), 8), j = min(max((int)fy, 0), 8), l = min(max((int)fz, 0), 8);
                const float wx = x - fx, wy = y - fy, wz = z - fz;
                const int o = (l * 10 + j) * 10 + i;
                const float2 c000 = TEX(o), c100 = TEX(o + 1), c010 = TEX(o + 10), c110 = TEX(o + 11);
                const float2 c001 = TEX(o + 100), c101 = TEX(o + 101), c011 = TEX(o + 110), c111 = TEX(o + 111);
                const float d00 = c000.x + wx * (c100.x - c000.x), d10 = c010.x + wx * (c110.x - c010.x);
                const float d01 = c001.x + wx * (c101.x - c001.x), d11 = c011.x + wx * (c111.x - c011.x);
                const float s00 = c000.y + wx * (c100.y - c000.y), s10 = c010.y + wx * (c110.y - c010.y);
                const float s01 = c001.y + wx * (c101.y - c001.y), s11 = c011.y + wx * (c111.y - c011.y);
                const float d0 = d00 + wy * (d10 - d00), d1 = d01 + wy * (d11 - d01);
                const float s0 = s00 + wy * (s10 - s00), s1 = s01 + wy * (s11 - s01);
                const float sigma = d0 + wz * (d1 - d0);
                const float sun = s0 + wz * (s1 - s0);
                const float alpha = 1.f - __expf(-sigma * step * voxelWorld);
                L += T * alpha * sun;
                T *= 1.f - alpha;
                ++samples;
                if (T < 1e-3f) { live = false; break; }
            }
        }
        if (mode == 2)
        {
            if (!live)
                break;
        }
        else if (__syncthreads_or(live ? 1 : 0) == 0)
            break;
    }
    out[tid] = T + L * 1e-9f;
    // Counted in an untimed launch only: one same-address atomic per thread serialises into milliseconds, which is what made the
    // first version of this benchmark (and direct_path_rate.py) report the atomics rather than the transport.
    if (counting)
    {
        atomicAdd(&counts[0], (unsigned long long)samples);
        if (lane == 0) atomicAdd(&counts[1], (unsigned long long)stages);
    }
    #undef TEX
    #undef H2F
"""


@wp.func_native(SNIPPET)
def tile_body(tid: int, block: int, tileW: int, tilesX: int, width: int, height: int, pixelAngle: float, cam: wp.vec3,
              fwd: wp.vec3, right: wp.vec3, up: wp.vec3, listStart: wp.array(dtype=wp.int32), listCount: wp.array(dtype=wp.int32),
              listBricks: wp.array(dtype=wp.int32), brickOrigin: wp.array(dtype=wp.vec3i), atlas: wp.array(dtype=wp.uint32),
              mode: int, step: float, voxelWorld: float, out: wp.array(dtype=float), counts: wp.array(dtype=wp.uint64), counting: int):
    ...


@wp.kernel
def transport(block: int, tileW: int, tilesX: int, width: int, height: int, pixelAngle: float, cam: wp.vec3, fwd: wp.vec3,
              right: wp.vec3, up: wp.vec3, listStart: wp.array(dtype=wp.int32), listCount: wp.array(dtype=wp.int32),
              listBricks: wp.array(dtype=wp.int32), brickOrigin: wp.array(dtype=wp.vec3i), atlas: wp.array(dtype=wp.uint32),
              mode: int, step: float, voxelWorld: float, out: wp.array(dtype=float), counts: wp.array(dtype=wp.uint64), counting: int):
    tid = wp.tid()
    tile_body(tid, block, tileW, tilesX, width, height, pixelAngle, cam, fwd, right, up, listStart, listCount, listBricks,
              brickOrigin, atlas, mode, step, voxelWorld, out, counts, counting)


def build_atlas(vol):
    """Every non-empty 8^3 brick with its one-voxel apron, as fp16 (density, sun) pairs packed in a uint32."""
    n = np.array(vol.shape) // 8
    vol = vol[:n[0] * 8, :n[1] * 8, :n[2] * 8]
    # Sun visibility straight down (+y is up): transmittance from the top of the volume.
    tau = np.flip(np.cumsum(np.flip(vol, 1), 1), 1) * VOXEL_WORLD
    sun = np.exp(-tau).astype(np.float32)
    pad = lambda a: np.pad(a, 1, mode="edge")
    vp, sp = pad(vol), pad(sun)
    occupied = vol.reshape(n[0], 8, n[1], 8, n[2], 8).max((1, 3, 5)) > 0
    index = -np.ones(n, np.int32)
    coords = np.argwhere(occupied)
    index[tuple(coords.T)] = np.arange(len(coords))
    atlas = np.empty((len(coords), 10, 10, 10), np.uint32)
    for b, (i, j, k) in enumerate(coords):
        d = vp[i * 8:i * 8 + 10, j * 8:j * 8 + 10, k * 8:k * 8 + 10].astype(np.float16).view(np.uint16).astype(np.uint32)
        s = sp[i * 8:i * 8 + 10, j * 8:j * 8 + 10, k * 8:k * 8 + 10].astype(np.float16).view(np.uint16).astype(np.uint32)
        atlas[b] = (d | (s << 16)).transpose(2, 1, 0)  # Stage layout (z, y, x), x fastest.
    return atlas.reshape(-1), (coords * 8).astype(np.int32), index


def tile_lists(index, cam, fwd, right, up, width, height, tileW, tileH, pixelAngle):
    """Per tile, the bricks its four corner rays and centre ray cross, front to back by depth along the view."""
    nb = np.array(index.shape)
    tilesX, tilesY = width // tileW, height // tileH
    starts, counts, bricks = [], [], []
    ts = np.arange(0.5, 3000, 1.0)
    for ty in range(tilesY):
        for tx in range(tilesX):
            found = {}
            for cx, cy in ((0, 0), (tileW, 0), (0, tileH), (tileW, tileH), (tileW / 2, tileH / 2)):
                u = (tx * tileW + cx - 0.5 * width) * pixelAngle
                v = (ty * tileH + cy - 0.5 * height) * pixelAngle
                d = fwd + u * right + v * up
                d /= np.linalg.norm(d)
                p = cam[None] + ts[:, None] * d[None]
                b = np.floor(p / 8).astype(int)
                ok = np.all((b >= 0) & (b < nb), 1)
                b, t = b[ok], ts[ok]
                ids = index[b[:, 0], b[:, 1], b[:, 2]]
                hit = ids >= 0
                unique, first = np.unique(ids[hit], return_index=True)  # t increases along the ray: first is the entry
                for i, tt in zip(unique.tolist(), t[hit][first].tolist()):
                    found[i] = min(found.get(i, 1e30), tt)
            order = sorted(found, key=found.get)
            starts.append(len(bricks))
            counts.append(len(order))
            bricks += order
    return np.array(starts, np.int32), np.array(counts, np.int32), np.array(bricks if bricks else [0], np.int32), tilesX * tilesY, tilesX


def main():
    vol = np.load(VOLUME)
    atlas, origins, index = build_atlas(vol)
    print(f"atlas: {len(origins)} bricks, {atlas.nbytes / 1e6:.0f} MB (fp16 density + sun), GPU {wp.get_device().name}")
    atlas_g = wp.array(atlas, dtype=wp.uint32)
    origins_g = wp.array(origins, dtype=wp.vec3i)
    centre = np.array(vol.shape) / 2
    distance = 900.0
    view = np.array([0.3, -0.25, 1.0])
    view /= np.linalg.norm(view)
    cam = centre - view * distance
    right = np.cross(view, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    width = height = 1024
    pixelAngle = 1.0 / distance  # a texel is one voxel at the cloud's centre
    counts = wp.zeros(2, dtype=wp.uint64)
    for block, tileW in ((64, 8), (128, 16), (256, 16)):
        tileH = block // tileW
        starts, lens, bricks, tiles, tilesX = tile_lists(index, cam, view, right, up, width, height, tileW, tileH, pixelAngle)
        lists = [wp.array(a, dtype=wp.int32) for a in (starts, lens, bricks)]
        out = wp.zeros(tiles * block, dtype=float)
        for step in (1.0, 0.5):
            for mode, name in ((0, "staged  "), (1, "resident"), (2, "direct  ")):
                def launch(counting):
                    wp.launch(transport, dim=tiles * block, block_dim=block,
                              inputs=[block, tileW, tilesX, width, height, pixelAngle, wp.vec3(*cam), wp.vec3(*view), wp.vec3(*right),
                                      wp.vec3(*up), *lists, origins_g, atlas_g, mode, step, VOXEL_WORLD, out, counts, counting])
                # The laptop GPU idles at P8 (210 MHz) between short launches and ramps slowly: keep it busy ~2 s, then time
                # 20 launches back to back with GPU events (Python's per-launch overhead must not count).
                warm = time.perf_counter()
                while time.perf_counter() - warm < 2.0:
                    launch(0)
                    wp.synchronize()
                start, stop = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                wp.record_event(start)
                for _ in range(20):
                    launch(0)
                wp.record_event(stop)
                wp.synchronize()
                best = wp.get_event_elapsed_time(start, stop) / 20 / 1e3
                counts.zero_()
                launch(1)
                samples, stages = (int(c) for c in counts.numpy())
                print(f"tile {block:3d} ({tileW}x{tileH}), step {step}, {name}: {samples / 1e6:6.1f}M steps in "
                      f"{best * 1e3:6.3f} ms = {samples / best / 1e9:6.1f} G steps/s; {stages} stages, {samples / max(stages, 1):6.0f} steps a stage, "
                      f"staging {stages * 4000 / best / 1e9:5.0f} GB/s; brick lists {lens.mean():.1f} a tile")


if __name__ == "__main__":
    main()
