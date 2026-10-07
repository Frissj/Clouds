"""Form R gate R0, GPU half (Windows, slangpy on D3D12): what do the hardware ray queries that cross empty space between the shared
occupancy hulls cost for the 4K walk frame's lattice rays?

LIT_VOLUME.md section 6c. Loads form_r_scene.py's .npz, builds one BLAS per (asset, level) hull and one TLAS of the sea's
instances (CloudSea::makeTile's scaled signed permutations as instance transforms), then runs form_r_trace.slang: per lattice
ray, one query per hull face crossed (the union's inside is a front / back count), to the slab's far plane or the near view
distance. Arms, in one process, each timed by GPU timestamps around its dispatch (median of --reps after 2 warm-ups):
  full          every face crossing, rays in 8 x 8 lattice tiles                       - the number the stop rule reads
  shuffled      the same rays in a random order                     (sanity: must be slower; counts must equal full's)
  first         one query a ray, its first entry                    (what rasterising the hulls would give)
  full again    anchor                                              (must repeat full within ~5%)
Before timing: both facing conventions are tried and the one under which no ray's first hit is a back face is kept (the camera
is outside every hull); the first union entry of the --check rays is compared with form_r_scene.py's CPU reference interval.

Stop rule (LIT_VOLUME.md, set before any run): stop Form R if `full` costs more than 0.5 ms for every lattice ray of the frame.

    pip install slangpy numpy
    python form_r_trace.py form_r_walk.npz [--reps 20] [--device d3d12|vulkan]
"""
import argparse
import os

import numpy as np
import slangpy as spy

HERE = os.path.dirname(os.path.abspath(__file__))


def build(device, desc, kind):
    sizes = device.get_acceleration_structure_sizes(desc)
    scratch = device.create_buffer(size=sizes.scratch_size, usage=spy.BufferUsage.unordered_access)
    accel = device.create_acceleration_structure(kind=kind, size=sizes.acceleration_structure_size)
    enc = device.create_command_encoder()
    enc.build_acceleration_structure(desc=desc, dst=accel, src=None, scratch_buffer=scratch)
    device.submit_command_buffer(enc.finish())
    device.wait_for_idle()
    return accel, sizes.acceleration_structure_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--device", choices=("d3d12", "vulkan"), default="d3d12")
    ap.add_argument("--max-queries", type=int, default=1024, help="cap per ray (reported if any ray reaches it)")
    args = ap.parse_args()
    z = np.load(args.scene)
    device = spy.Device(type=getattr(spy.DeviceType, args.device), enable_debug_layers=False, enable_rhi_validation=False,
                        compiler_options={"include_paths": [HERE]})
    if not device.has_feature(spy.Feature.acceleration_structure):
        raise SystemExit("no acceleration structure support on this device")
    print(f"{device.info.api_name} {device.info.adapter_name}")

    keys = z["blas_keys"]
    blas, blas_bytes, tris = [], 0, 0
    for k in range(len(keys)):
        v, i = z[f"blas{k}_v"], z[f"blas{k}_i"]
        vb = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=np.ascontiguousarray(v.reshape(-1)))
        ib = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=np.ascontiguousarray(i.reshape(-1)))
        tri = spy.AccelerationStructureBuildInputTriangles({
            "vertex_buffers": [vb], "vertex_format": spy.Format.rgb32_float, "vertex_count": len(v), "vertex_stride": 12,
            "index_buffer": ib, "index_format": spy.IndexFormat.uint32, "index_count": i.size,
            "flags": spy.AccelerationStructureGeometryFlags.opaque})
        a, size = build(device, spy.AccelerationStructureBuildDesc(
            {"inputs": [tri], "flags": spy.AccelerationStructureBuildFlags.prefer_fast_trace}),
            spy.AccelerationStructureKind.bottom_level)
        blas.append(a)
        blas_bytes += size
        tris += len(i)

    xform, which = z["inst_xform"], z["inst_blas"]

    def tlas_for(flags):
        lst = device.create_acceleration_structure_instance_list(len(xform))
        for n in range(len(xform)):
            lst.write(n, {"transform": spy.float3x4(xform[n].reshape(-1).tolist()), "instance_id": n, "instance_mask": 0xFF,
                          "instance_contribution_to_hit_group_index": 0, "flags": flags,
                          "acceleration_structure": blas[int(which[n])].handle})
        return build(device, spy.AccelerationStructureBuildDesc(
            {"inputs": [lst.build_input_instances()], "flags": spy.AccelerationStructureBuildFlags.prefer_fast_trace}),
            spy.AccelerationStructureKind.top_level)[0]

    rays = np.ascontiguousarray(z["rays"], np.float32)
    n = len(rays)
    ray_buf = device.create_buffer(element_count=n, struct_size=32, usage=spy.BufferUsage.shader_resource, data=rays)
    ident = device.create_buffer(element_count=n, struct_size=4, usage=spy.BufferUsage.shader_resource,
                                 data=np.arange(n, dtype=np.uint32))
    shuf = device.create_buffer(element_count=n, struct_size=4, usage=spy.BufferUsage.shader_resource,
                                data=np.random.default_rng(1).permutation(n).astype(np.uint32))
    out = device.create_buffer(element_count=n, struct_size=16,
                               usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource)
    kernel = device.create_compute_kernel(device.load_program("form_r_trace.slang", ["traceRuns"]))
    print(f"{len(keys)} hulls ({tris} triangles, BLAS {blas_bytes / 2**20:.1f} MB), {len(xform)} instances, {n} rays")

    def run(tlas, order, max_queries, qp=None, k=0):
        kw = dict(query_pool=qp, query_index_before=2 * k, query_index_after=2 * k + 1) if qp is not None else {}
        kernel.dispatch(thread_count=[n, 1, 1], vars={"g_tlas": tlas, "g_rays": ray_buf, "g_order": order, "g_out": out,
                                                      "g_count": n, "g_maxQueries": max_queries}, **kw)

    def result():
        device.wait_for_idle()
        return out.to_numpy().view(np.uint32).reshape(n, 4).copy()

    # Facing convention: the camera is outside every hull, so the right one has no first hit on a back face.
    tried = []
    for name, flags in (("default", spy.AccelerationStructureInstanceFlags.none),
                        ("front counter-clockwise", spy.AccelerationStructureInstanceFlags.triangle_front_counter_clockwise)):
        t = tlas_for(flags)
        run(t, ident, 1)
        r = result()
        back = int(r[:, 2].sum())
        hit = back + int((r.view(np.float32)[:, 3] >= 0).sum())
        print(f"facing {name}: {back} of {hit} first hits on a back face")
        tried.append((back, hit, name, t))
    back, hit, chosen, tlas = min(tried, key=lambda x: x[0])
    if back > 0.01 * max(hit, 1):
        raise SystemExit("neither facing convention keeps first hits off back faces: the hulls are not closed or not oriented")

    # Counts and the CPU reference.
    run(tlas, ident, args.max_queries)
    full = result()
    q, ent, ft = full[:, 0], full[:, 1], full.view(np.float32)[:, 3]
    capped = int((q >= args.max_queries).sum())
    chk, lo, hi, faces = z["check_idx"], z["check_lo"], z["check_hi"], z["check_faces"]
    g = ft[chk].astype(np.float64)
    gpu_hit, cpu_hit = g >= 0, np.isfinite(hi)
    tol = 1e-2 + 1e-5 * np.where(gpu_hit, g, 0)
    late = gpu_hit & cpu_hit & (g > hi + tol)
    early = gpu_hit & cpu_hit & (g < lo - tol)
    print(f"check rays ({len(chk)}): both miss {int((~gpu_hit & ~cpu_hit).sum())}, both hit {int((gpu_hit & cpu_hit).sum())} "
          f"(inside the CPU interval {int((gpu_hit & cpu_hit & ~late & ~early).sum())}, late {int(late.sum())}, early "
          f"{int(early.sum())}), GPU only {int((gpu_hit & ~cpu_hit).sum())}, CPU only {int((~gpu_hit & cpu_hit).sum())}; "
          f"queries a check ray {q[chk].mean():.1f} against CPU faces crossed {faces.mean():.1f} (+1 for the last miss)")
    print(f"queries: total {int(q.sum())}, a ray mean {q.mean():.1f}, p50 / p90 / p99 / max {np.percentile(q, 50):.0f} / "
          f"{np.percentile(q, 90):.0f} / {np.percentile(q, 99):.0f} / {q.max()}, rays at the cap {capped}; union entries a ray "
          f"{ent.mean():.2f} (empty runs crossed by one query each), rays entering any hull {int((ent > 0).sum())}")

    # Timing: one process, arms interleaved per repetition, GPU timestamps around each dispatch.
    arms = [("full", ident, args.max_queries), ("shuffled", shuf, args.max_queries), ("first", ident, 1),
            ("full again", ident, args.max_queries)]
    qp = device.create_query_pool(spy.QueryType.timestamp, 2 * len(arms) * args.reps)
    for _ in range(2):
        for _, order, mq in arms:
            run(tlas, order, mq)
    device.wait_for_idle()
    k = 0
    for _ in range(args.reps):
        for _, order, mq in arms:
            run(tlas, order, mq, qp, k)
            k += 1
    device.wait_for_idle()
    ts = np.array(qp.get_timestamp_results(0, 2 * k)).reshape(args.reps, len(arms), 2)
    ms = (ts[:, :, 1] - ts[:, :, 0]) * 1e3
    total = float(q.sum())
    print(f"timing ({args.reps} reps, facing {chosen}):")
    for a, (name, _, mq) in enumerate(arms):
        nq = n if mq == 1 else total
        print(f"  {name:11s} median {np.median(ms[:, a]):.3f} ms (min {ms[:, a].min():.3f}, max {ms[:, a].max():.3f}), "
              f"{np.median(ms[:, a]) / nq * 1e6:.3f} ms per 1M queries")
    verdict = "PASS" if np.median(ms[:, 0]) <= 0.5 else "STOP"
    print(f"R0 stop rule (full <= 0.5 ms for every lattice ray): {verdict}")


if __name__ == "__main__":
    main()
