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
# HSTR_CACHE_CURVE=1: instead of the cell sweep, error of 4-voxel order-1 cells against the number of paths per cell.
CURVE = os.environ.get("HSTR_CACHE_CURVE", "0") == "1"
CHECKPOINTS = [4, 16, 64, 256, 1024, 4096]
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

    if CURVE:
        hstr.set_properties({"debugView": 8, "compareReference": False, "worldCacheCellVoxels": 8})
        m.renderFrame()  # A cell-size change restarts the cache.
        hstr.set_properties({"worldCacheCellVoxels": 4, "worldCacheUpdates": 1, "worldCacheOrder": 1})
        for checkpoint in CHECKPOINTS:
            while int(hstr.properties["worldCacheSampleCount"]) < checkpoint:
                per_frame = min(16, checkpoint - int(hstr.properties["worldCacheSampleCount"]))
                hstr.set_properties({"worldCacheUpdates": per_frame})
                m.renderFrame()
            hstr.set_properties({"worldCacheUpdates": 0})
            samples = int(hstr.properties["worldCacheSampleCount"])
            for order in (0, 1):
                cache_term = measure(0, 0, SMOOTH, 8, {"worldCacheOrder": order})
                lines.append(f"  4-voxel order {order}, {samples:5d} paths per cell: term {fmt(cache_term)}")
        trace(f"{name} curve done")
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
            cache_term = measure(0, 0, SMOOTH, 8, extra)
            cache_total = measure(0, BACKGROUND | SUN1, ALL, 8, extra)
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
