"""Direct resolved-brick path throughput on this GPU, offline (NVIDIA Warp, no Mogwai).

Question: if a dirty ray walked a direct page table brick by brick (DDA; empty bricks jumped at their exit; a fine brick's atlas
base resolved once on entry) and sampled a packed (density, sun) atlas from global memory, how many fine samples a second would
the GPU sustain - against the ~23 G/s at which the dirty frame's ~10.2M fine samples (brickshare1) fit a 2 ms frame?

The atlas is the bench cloud at mip 1 (every non-empty 8^3 brick with a one-voxel apron, fp16 density + sun packed in a uint32,
207 MB: far past L2, as the real 256 MB atlas is). Trilinear is 8 loads from global memory through L1 - hardware texture filtering
would do one fetch, so this bounds the fetch cost from above. Perspective rays from a camera D voxels out, a texel one voxel at
the cloud. Arms: step 4 / 2 / 1 voxels (the dirty march takes ~2 samples a brick visit: step 4 in an 8-voxel brick is the real
pattern), rays coherent (warps are pixel rows, as a full frame) or shuffled (each warp's rays from anywhere on screen: scattered
dirty rays at worst).

    python scripts/HSTR/bench/direct_path_rate.py

MEASURED (RTX 4080 Laptop, 1M rays, 1.32M fine visits, 16.4M DDA brick steps): 2.3-3.2 ms whatever the step - 1.8M / 3.6M /
7.1M fine samples at step 4 / 2 / 1 all cost the same, so the traversal, not the sampling, sets the time: 5-7 G DDA brick steps/s,
0.4-0.6 G fine visits/s, coherent and shuffled alike. Counting brick steps and samples together, ~8-9 G loop iterations/s - the
shipping march's 7.3 G steps/s, the cooperative kernel's 5.6-11 G/s staged. A divergent ray loop with a dependent global load per
iteration runs at ~7-9 G iterations/s on this GPU however little the iteration does; only on-chip data (17-30 G/s) beats it.
MEASURED (arms, direct_path_rate2.log): one tap instead of eight changes nothing (2.64 vs 2.77 ms); a chessboard brick distance
field cuts the DDA 16.4M -> 5.0M brick steps and the time only 2.7 -> 1.6-2.0 ms, whatever the step (1.8-7.1M samples). What is
left is per ray, not per iteration: ~1.6-2 ms per 1M rays, the longest ray of each warp (divergence) setting it. 742k dirty rays
would cost ~1.2-1.5 ms in the leanest exact loop before any lighting.
"""
import os
import time

import numpy as np
import warp as wp

import tile_transport_rate as ttr  # build_atlas: the same bricks, apron and packing

VOLUME = os.path.join(os.environ.get("TEMP", "."), "hstr_jet_level1.npy")
VOXEL_WORLD = 0.4455
wp.init()


vec2h = wp.types.vector(length=2, dtype=wp.float16)


@wp.func
def corner(atlas: wp.array(dtype=vec2h), base: int, o: int) -> wp.vec2:
    v = atlas[base + o]
    return wp.vec2(float(v[0]), float(v[1]))


@wp.kernel
def direct(table: wp.array3d(dtype=wp.int32), dist: wp.array3d(dtype=wp.int32), atlas: wp.array(dtype=vec2h),
           order: wp.array(dtype=wp.int32), width: int, height: int, pixelAngle: float, cam: wp.vec3, fwd: wp.vec3, right: wp.vec3,
           up: wp.vec3, step: float, voxelWorld: float, taps: int, skip: int, out: wp.array(dtype=float),
           counts: wp.array(dtype=wp.int64)):
    tid = wp.tid()
    p = order[tid]
    u = (float(p % width) + 0.5 - 0.5 * float(width)) * pixelAngle
    v = (float(p / width) + 0.5 - 0.5 * float(height)) * pixelAngle
    d = wp.normalize(fwd + right * u + up * v)
    nb = wp.vec3(float(table.shape[0]), float(table.shape[1]), float(table.shape[2])) * 8.0
    inv = wp.vec3(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])
    ta = wp.cw_mul(wp.vec3(0.0, 0.0, 0.0) - cam, inv)
    tb = wp.cw_mul(nb - cam, inv)
    t = wp.max(wp.max(wp.min(ta[0], tb[0]), wp.min(ta[1], tb[1])), wp.max(wp.min(ta[2], tb[2]), 0.0))
    tEnd = wp.min(wp.min(wp.max(ta[0], tb[0]), wp.max(ta[1], tb[1])), wp.max(ta[2], tb[2]))
    T = float(1.0)
    L = float(0.0)
    samples = int(0)
    visits = int(0)
    tNext = t + 0.5 * step  # one sample phase for the whole ray, continuous across bricks
    guard = int(0)
    while t < tEnd and T > 1.0e-3 and guard < 4096:
        guard += 1
        x = cam + d * (t + 1.0e-3)  # past the last exit: at t ~ 900, 1e-4 is under an fp32 ulp and re-enters the same brick
        bi = int(wp.floor(x[0] / 8.0))
        bj = int(wp.floor(x[1] / 8.0))
        bk = int(wp.floor(x[2] / 8.0))
        if bi < 0 or bj < 0 or bk < 0 or bi >= table.shape[0] or bj >= table.shape[1] or bk >= table.shape[2]:
            break
        o = wp.vec3(float(bi), float(bj), float(bk)) * 8.0
        ea = wp.cw_mul(o - cam, inv)
        eb = wp.cw_mul(o + wp.vec3(8.0, 8.0, 8.0) - cam, inv)
        tExit = wp.min(wp.min(wp.max(ea[0], eb[0]), wp.max(ea[1], eb[1])), wp.max(ea[2], eb[2]))
        entry = table[bi, bj, bk]
        if entry >= 0:
            visits += 1
            base = entry * 1000
            while tNext < tExit and T > 1.0e-3:
                q = cam + d * tNext - o + wp.vec3(0.5, 0.5, 0.5)
                fx = wp.floor(q[0])
                fy = wp.floor(q[1])
                fz = wp.floor(q[2])
                i = wp.clamp(int(fx), 0, 8)
                j = wp.clamp(int(fy), 0, 8)
                k = wp.clamp(int(fz), 0, 8)
                w = q - wp.vec3(fx, fy, fz)
                c = (k * 10 + j) * 10 + i
                ds = corner(atlas, base, c)
                if taps == 8:
                    c100 = corner(atlas, base, c + 1)
                    c010 = corner(atlas, base, c + 10)
                    c110 = corner(atlas, base, c + 11)
                    c001 = corner(atlas, base, c + 100)
                    c101 = corner(atlas, base, c + 101)
                    c011 = corner(atlas, base, c + 110)
                    c111 = corner(atlas, base, c + 111)
                    e0 = wp.lerp(wp.lerp(ds, c100, w[0]), wp.lerp(c010, c110, w[0]), w[1])
                    e1 = wp.lerp(wp.lerp(c001, c101, w[0]), wp.lerp(c011, c111, w[0]), w[1])
                    ds = wp.lerp(e0, e1, w[2])
                alpha = 1.0 - wp.exp(-ds[0] * step * voxelWorld)
                L += T * alpha * ds[1]
                T *= 1.0 - alpha
                samples += 1
                tNext += step
        else:
            # Empty: jump to the exit, or with the brick distance field past every brick within (dist - 1) bricks (chessboard
            # distance bounds the Euclidean one), keeping the sample phase.
            if skip != 0:
                tExit = wp.max(tExit, t + float(wp.max(dist[bi, bj, bk] - 1, 0)) * 8.0)
            tNext = tNext + wp.ceil(wp.max(tExit - tNext, 0.0) / step) * step
        t = tExit
    out[tid] = T + L * 1.0e-9
    wp.atomic_add(counts, 0, wp.int64(samples))
    wp.atomic_add(counts, 1, wp.int64(visits))
    wp.atomic_add(counts, 2, wp.int64(guard))


def main():
    vol = np.load(VOLUME)
    atlas, origins, index = ttr.build_atlas(vol)
    print(f"atlas {len(origins)} bricks, {atlas.nbytes / 1e6:.0f} MB; page table {index.shape} = {index.nbytes / 1e6:.1f} MB; GPU {wp.get_device().name}")
    atlas_g = wp.array(atlas.view(np.float16).reshape(-1, 2), dtype=vec2h)  # (density, sun) per texel, 4 bytes
    table_g = wp.array(index, dtype=wp.int32)
    from scipy.ndimage import distance_transform_cdt
    dist_g = wp.array(distance_transform_cdt(index < 0, metric="chessboard").astype(np.int32), dtype=wp.int32)
    centre = np.array(vol.shape) / 2
    distance = 900.0
    view = np.array([0.3, -0.25, 1.0])
    view /= np.linalg.norm(view)
    cam = centre - view * distance
    right = np.cross(view, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    width = height = 1024
    rng = np.random.default_rng(0)
    orders = {"coherent": np.arange(width * height, dtype=np.int32),
              "shuffled": rng.permutation(width * height).astype(np.int32)}
    counts = wp.zeros(3, dtype=wp.int64)
    out = wp.zeros(width * height, dtype=float)
    # (step, taps, distance-field skip): the real ~2 samples a visit is step 4; one tap stands for a hardware-filtered fetch.
    arms = [(4.0, 8, 0), (4.0, 1, 0), (4.0, 8, 1), (4.0, 1, 1), (2.0, 1, 1), (1.0, 8, 0), (1.0, 1, 1)]
    for step, taps, skip in arms:
        for name, order in orders.items():
            order_g = wp.array(order, dtype=wp.int32)
            name = f"{name}, {taps} tap{'s' if taps > 1 else ''}, skip {skip}"

            def launch():
                counts.zero_()
                wp.launch(direct, dim=width * height, inputs=[table_g, dist_g, atlas_g, order_g, width, height, 1.0 / distance,
                                                              wp.vec3(*cam), wp.vec3(*view), wp.vec3(*right), wp.vec3(*up), step,
                                                              VOXEL_WORLD, taps, skip, out, counts])

            warm = time.perf_counter()
            while time.perf_counter() - warm < 2.0:  # hold the clocks up (see tile_transport_rate.py)
                launch()
                wp.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                launch()
            wp.synchronize()
            seconds = (time.perf_counter() - start) / 20
            samples, visits, bricks = (int(c) for c in counts.numpy())
            print(f"step {step:.0f}, {name:28s}: {samples / 1e6:6.1f}M fine samples, {visits / 1e6:5.2f}M visits ({samples / max(visits, 1):.1f} a visit) "
                  f"in {seconds * 1e3:6.2f} ms = {samples / seconds / 1e9:5.1f} G samples/s, {visits / seconds / 1e9:5.2f} G visits/s; "
                  f"{bricks / 1e6:.1f}M DDA brick steps (empty included) = {bricks / seconds / 1e9:5.2f} G/s")


if __name__ == "__main__":
    main()
