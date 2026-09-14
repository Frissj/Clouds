import json
import math
import os
import time
from falcor import *

# Quality ceiling of a world-space SH radiance cache for the smooth transport term (sun scattered two or more times plus
# scattered sky), against the path-traced reference and against the current HST sky + fitted sun octaves 1-3.
# The cache is converged offline here (no frame budget): this measures what the representation can hold, not its cost.
# Writes %TEMP%/hstr_bench/<tag>_cache.txt.

OUT = "C:/Users/Friss/AppData/Local/Temp/hstr_bench"
TAG = os.environ.get("HSTR_TAG", "cache")
SPP = int(os.environ.get("HSTR_SPP", "1024"))
W = int(os.environ.get("HSTR_W", "960"))
H = int(os.environ.get("HSTR_H", "540"))
# (cell voxels, gather passes, passes per frame); HSTR_CACHE_SCALE shortens the convergence for smoke tests.
SCALE = float(os.environ.get("HSTR_CACHE_SCALE", "1"))
CELLS = [(c, max(1, int(n * SCALE)), k) for c, n, k in [(8, 4096, 64), (4, 4096, 16), (2, 2048, 4)]]
# HSTR_CACHE_CURVE=1: instead of the cell sweep, error of 4-voxel cells against GPU time for each estimator:
# (name, worldCacheEstimator, updates per frame while converging, checkpoints in estimator passes).
CURVE = os.environ.get("HSTR_CACHE_CURVE", "0") == "1"
PHOTONS = int(os.environ.get("HSTR_CACHE_PHOTONS", "262144"))
ESTIMATORS = [
    (label, estimator, per_frame, sorted({max(1, int(c * SCALE)) for c in checkpoints}))
    for label, estimator, per_frame, checkpoints in [
        ("gather", 0, 16, [4, 16, 64, 256, 1024, 4096]),
        ("light tracing", 1, 4, [int(c) for c in os.environ.get("HSTR_CACHE_CHECKPOINTS", "1,4,16,64,256,1024").split(",")]),
    ]
    if label in os.environ.get("HSTR_CACHE_ESTIMATORS", "gather,light tracing").split(",")
]
# HSTR_CACHE_LIGHTSWEEP=1: converged light-tracing caches over (cell voxels, deposited SH bands), every evaluation order.
LIGHTSWEEP = os.environ.get("HSTR_CACHE_LIGHTSWEEP", "0") == "1"
LIGHTSWEEP_BATCHES = max(1, int(32 * SCALE))
# HSTR_CACHE_FULL=1: the complete split frame (background + fine-sun single scattering + cache) against the current frame.
FULL = os.environ.get("HSTR_CACHE_FULL", "0") == "1"
FULL_VARIANTS = json.loads(os.environ.get("HSTR_FULL_VARIANTS", "[]"))
FULL_DEFAULTS = {"stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "maxStepVoxels": 4.0, "lightingStride": 1, "worldCacheOrder": 2,
                 "worldCacheWindow": 1, "worldCacheTextured": 1}
# HSTR_CACHE_CONFIGS="4:2,2:2" restricts the sweep to (cell voxels, bands) pairs.
LIGHTSWEEP_CONFIGS = [tuple(int(x) for x in c.split(":")) for c in os.environ.get("HSTR_CACHE_CONFIGS", "8:1,4:1,4:2,2:1,2:2").split(",")]


def update_cost_ms(estimator, per_frame):
    """Mean GPU time of one estimator pass, from a profiled run of frames doing per_frame passes each."""
    hstr.set_properties({"debugView": 8, "compareReference": False, "worldCacheEstimator": estimator, "worldCacheUpdates": per_frame})
    for i in range(4):
        m.renderFrame()
    m.profiler.enabled = True
    m.profiler.start_capture()
    for i in range(16):
        m.renderFrame()
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    times = [lane["stats"]["mean"] for name, lane in capture["events"].items() if name.endswith("/worldCache/gpu_time")]
    return (times[0] if times else float("nan")) / per_frame
ONLY = [c for c in os.environ.get("HSTR_CASES", "").split(",") if c]

target = float3(-10.0, 73.0, -43.0)
radius = 510.0
BACKGROUND, SUN1, SUNN, SKY, ALL = 1, 2, 4, 8, 15
SMOOTH = SUNN | SKY


def normalized(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


FIT_SUN = (0.4319, 0.8639, 0.2699)
BACK_SUN = normalized((-0.5, 0.6, 0.62))


def orbit(azimuth, height, distance=0.8):
    return (target.x + distance * radius * math.cos(azimuth), height, target.z + distance * radius * math.sin(azimuth))


def opposite_sun(sun, distance=1.1):
    return (target.x - distance * radius * sun[0], target.y - distance * radius * sun[1], target.z - distance * radius * sun[2])


CASES = [
    ("static", (target.x + radius, 60.0, target.z), FIT_SUN),
    ("backlit", opposite_sun(FIT_SUN), FIT_SUN),
    ("orbit60", orbit(math.radians(60.0), 60.0), FIT_SUN),
    ("oblique_backsun", orbit(2.2, 180.0), BACK_SUN),
]
if ONLY:
    CASES = [c for c in CASES if c[0] in ONLY]

os.makedirs(OUT, exist_ok=True)


def trace(message):
    with open(f"{OUT}/{TAG}_trace.txt", "a") as f:
        f.write(message + "\n")


g = RenderGraph("WorldCache")
g.addPass(createPass("HSTRCloud", {"cutTransmittanceTolerance": 0.05}), "HSTRCloud")
g.addPass(createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0}), "ToneMapper")
g.addEdge("HSTRCloud.color", "ToneMapper.src")
g.markOutput("ToneMapper.dst")
m.addGraph(g)
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(W, H)
hstr = g.getPass("HSTRCloud")
cam = m.scene.camera
m.frameCapture.outputDir = OUT
trace("scene loaded")


def shoot(name):
    m.frameCapture.baseFilename = f"{TAG}_{name}"
    m.frameCapture.capture()


def measure(hst, substitute, target_mask, debug_view=0, extra=None):
    props = {"debugView": debug_view, "compareReference": True, "hstComponents": hst, "compareSubstitute": substitute, "compareTarget": target_mask}
    props.update(extra or {})
    hstr.set_properties(props)
    for i in range(2):
        m.renderFrame()
    p = hstr.properties
    return (float(p["referenceError"]), float(p["referenceLogError"]), float(p["referenceNoiseError"]), float(p["referenceNoiseLogError"]))


def fmt(e):
    return f"L1 {e[0]:.4f} log {e[1]:.4f}"


lines = []
for name, position, sun in CASES:
    hstr.set_properties({"sunDirection": float3(*sun), "debugView": 6, "compareReference": False, "referenceShow": ALL})
    cam.position = float3(*position)
    cam.target = target
    for i in range(SPP):
        m.renderFrame()
    hstr.set_properties({"referenceShow": SMOOTH})
    m.renderFrame()
    shoot(f"{name}_pt_smooth")
    trace(f"reference {name} done")

    lines.append(f"== {name}")
    term = measure(SMOOTH, 0, SMOOTH)
    total = measure(ALL, 0, ALL)
    lines.append(f"  smooth term: reference noise L1 {term[2]:.4f} log {term[3]:.4f}; total noise L1 {total[2]:.4f} log {total[3]:.4f}")
    lines.append(f"  current HST (sky + sun octaves 1-3)      term {fmt(term)}   total {fmt(total)}")
    lines.append(f"  current HST + exact background/single    total {fmt(measure(SMOOTH, BACKGROUND | SUN1, ALL))}")

    if FULL:
        # Complete frame, no reference components: transmitted background + fine-sun-field single scattering (residual
        # octave 0 at strength 1) + light-traced 2-voxel cache, SH order 2 windowed.
        split = {"residualStrength": 1.0, "worldCacheOrder": 2, "worldCacheWindow": 1}
        hstr.set_properties({"worldCacheCellVoxels": 2, "worldCacheBands": 2, "worldCachePhotons": 262144, "worldCacheEstimator": 0})
        hstr.set_properties({"debugView": 8, "compareReference": False, "worldCacheUpdates": 0})
        m.renderFrame()  # Restart the cache.
        hstr.set_properties({"worldCacheEstimator": 1})
        while int(hstr.properties["worldCacheSampleCount"]) < 32:
            hstr.set_properties({"worldCacheUpdates": 4})
            m.renderFrame()
        hstr.set_properties({"worldCacheUpdates": 0})
        lines.append(f"  split frame (fine-sun single + cache + background)  total {fmt(measure(ALL, 0, ALL, 8, split))}")
        lines.append(f"  split frame with exact single scattering            total {fmt(measure(ALL & ~SUN1, SUN1, ALL, 8, split))}")
        lines.append(f"  single scattering alone: fine sun field {fmt(measure(SUN1, 0, SUN1, 8, split))}"
                     f"   current octave 0 {fmt(measure(SUN1, 0, SUN1, 0, {'residualStrength': 1.4}))}")
        # Cost/quality variants of the split frame; every key a variant sets is restored from DEFAULTS afterwards.
        for label, props in FULL_VARIANTS:
            lines.append(f"  variant {label:40s} total {fmt(measure(ALL, 0, ALL, 8, dict(split, **props)))}")
            hstr.set_properties({key: FULL_DEFAULTS[key] for key in props})
        hstr.set_properties(dict(split, debugView=8, hstComponents=ALL, compareReference=False))
        m.renderFrame()
        shoot(f"{name}_split")
        hstr.set_properties({"debugView": 0, "residualStrength": 1.4, "worldCacheWindow": 0})
        m.renderFrame()
        m.renderFrame()
        shoot(f"{name}_hst")
        hstr.set_properties({"debugView": 6, "referenceShow": ALL})
        m.renderFrame()
        shoot(f"{name}_pt")
        trace(f"{name} full done")
        with open(f"{OUT}/{TAG}_cache.txt", "w") as f:
            f.write("Split frame vs current HST frame vs reference (log error of the whole image).\n\n" + "\n".join(lines) + "\n")
        continue

    if LIGHTSWEEP:
        for cell_voxels, bands in LIGHTSWEEP_CONFIGS:
            hstr.set_properties({"worldCacheCellVoxels": cell_voxels, "worldCacheBands": bands, "worldCachePhotons": PHOTONS, "worldCacheOrder": 0})
            cost = update_cost_ms(1, 4)
            hstr.set_properties({"worldCacheEstimator": 0, "worldCacheUpdates": 0})
            m.renderFrame()  # Restart the cache.
            hstr.set_properties({"worldCacheEstimator": 1})
            while int(hstr.properties["worldCacheSampleCount"]) < LIGHTSWEEP_BATCHES:
                hstr.set_properties({"worldCacheUpdates": min(4, LIGHTSWEEP_BATCHES - int(hstr.properties["worldCacheSampleCount"]))})
                m.renderFrame()
            hstr.set_properties({"worldCacheUpdates": 0})
            evaluations = [(str(order), order, 0) for order in range(bands + 1)] + [(f"{order}w", order, 1) for order in range(1, bands + 1)]
            errors = [(label, measure(SMOOTH, 0, SMOOTH, 8, {"worldCacheOrder": order, "worldCacheWindow": window})[1]) for label, order, window in evaluations]
            hstr.set_properties({"worldCacheWindow": 0})
            lines.append(
                f"  {cell_voxels}-voxel cells, bands 0-{bands}: {cost:.2f} ms per batch; after {LIGHTSWEEP_BATCHES} batches log error by order "
                + ", ".join(f"{label}: {e:.4f}" for label, e in errors)
            )
            trace(f"{name} {cell_voxels}/{bands} done")
        with open(f"{OUT}/{TAG}_cache.txt", "w") as f:
            f.write("Light-tracing cache: cell size and SH bands (smooth term).\n\n" + "\n".join(lines) + "\n")
        continue

    if CURVE:
        reference_mean = measure(0, 0, SMOOTH)[0]
        curve_cell = int(os.environ.get("HSTR_CACHE_CELL", "4"))
        curve_bands = int(os.environ.get("HSTR_CACHE_BANDS", "1"))
        hstr.set_properties({"worldCacheCellVoxels": curve_cell, "worldCacheBands": curve_bands, "worldCachePhotons": PHOTONS, "worldCacheOrder": 1})
        lines.append(f"  {curve_cell}-voxel cells, SH bands 0-{curve_bands}")
        for label, estimator, per_frame, checkpoints in ESTIMATORS:
            cost = update_cost_ms(estimator, per_frame)
            # Switching the estimator away and back restarts the cache.
            hstr.set_properties({"worldCacheEstimator": 1 - estimator, "worldCacheUpdates": 0})
            m.renderFrame()
            hstr.set_properties({"worldCacheEstimator": estimator})
            unit = "paths per cell" if estimator == 0 else f"batches of {PHOTONS} photons"
            lines.append(f"  {label}: {cost:.2f} ms GPU per pass ({unit})")
            for checkpoint in checkpoints:
                while int(hstr.properties["worldCacheSampleCount"]) < checkpoint:
                    hstr.set_properties({"worldCacheUpdates": min(per_frame, checkpoint - int(hstr.properties["worldCacheSampleCount"]))})
                    m.renderFrame()
                hstr.set_properties({"worldCacheUpdates": 0})
                samples = int(hstr.properties["worldCacheSampleCount"])
                evaluations = [("order 0", 0, 0), ("order 1", 1, 0)] + ([("order 2w", 2, 1)] if curve_bands >= 2 and estimator == 1 else [])
                errors = [(label, measure(SMOOTH, 0, SMOOTH, 8, {"worldCacheOrder": order, "worldCacheWindow": window})) for label, order, window in evaluations]
                hstr.set_properties({"worldCacheWindow": 0})
                cache_mean = measure(SMOOTH, 0, 0, 8, {"worldCacheOrder": 0})[0]
                lines.append(
                    f"    {samples:5d} passes = {samples * cost:8.1f} ms GPU: " + ", ".join(f"{label} log {e[1]:.4f}" for label, e in errors)
                    + f"  (cache mean {cache_mean:.4f} vs reference {reference_mean:.4f})"
                )
            trace(f"{name} {label} curve done")
        with open(f"{OUT}/{TAG}_cache.txt", "w") as f:
            f.write("World cache error against paths per cell (smooth term).\n\n" + "\n".join(lines) + "\n")
        continue

    for cell_voxels, passes, per_frame in CELLS:
        hstr.set_properties({"debugView": 8, "compareReference": False, "worldCacheCellVoxels": cell_voxels, "worldCacheUpdates": per_frame, "worldCacheOrder": 2})
        start = time.time()
        frames = 0
        while int(hstr.properties["worldCacheSampleCount"]) < passes:
            m.renderFrame()
            frames += 1
        seconds = time.time() - start
        samples = int(hstr.properties["worldCacheSampleCount"])
        lines.append(f"  cache {cell_voxels}-voxel cells: {samples} paths per cell in {seconds:.1f} s ({1000.0 * seconds / max(samples, 1):.1f} ms per pass incl. camera frames)")
        for order in (0, 1, 2):
            extra = {"worldCacheOrder": order}
            cache_term = measure(SMOOTH, 0, SMOOTH, 8, extra)
            cache_total = measure(SMOOTH, BACKGROUND | SUN1, ALL, 8, extra)
            lines.append(f"    SH order {order}: term {fmt(cache_term)}   total with exact background/single {fmt(cache_total)}")
        if cell_voxels == 4:
            hstr.set_properties({"worldCacheOrder": 2, "compareReference": False})
            m.renderFrame()
            shoot(f"{name}_cache4")
        trace(f"{name} cache {cell_voxels} done")
    with open(f"{OUT}/{TAG}_cache.txt", "w") as f:
        f.write("World-space SH cache vs path-traced reference (smooth term = sun scattered 2+ times + scattered sky).\n\n")
        f.write("\n".join(lines) + "\n")

trace("done")
exit()
