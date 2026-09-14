import math
import os
from falcor import *

# Photon efficiency of the world cache: GPU time per light-tracing batch and the frame's error against the saved 4K path-traced
# reference after a given number of batches (8x8-block log error, per-pixel split view with the fixed march).
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "photon")
W, H = int(os.environ.get("HSTR_W", "3840")), int(os.environ.get("HSTR_H", "2160"))
g = RenderGraph("PhotonStudy")
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
target = float3(-10.0, 73.0, -43.0)
radius = 510.0
base = {"hstComponents": 15, "residualStrength": 1.0, "worldCacheCellVoxels": 2, "worldCacheBands": 2, "worldCacheOrder": 2,
        "worldCacheWindow": 1, "worldCacheEstimator": 1, "worldCacheTextured": 1, "worldCacheSegments": 0, "worldCachePhotons": 65536,
        "worldCacheBakeInterval": 1, "worldCacheModulation": -1.0, "adaptiveMarch": False, "stepOpticalDepth": 0.5,
        "minStepVoxels": 1.0, "lightingStride": 1, "debugView": 8, "worldCacheSimilarity": 0}
default_sun = (0.4319, 0.8639, 0.2699)
back_sun = (-0.5, 0.6, 0.62)
n = math.sqrt(sum(c * c for c in back_sun))
back_sun = tuple(c / n for c in back_sun)
views = [
    ("side", (target.x + radius, 60.0, target.z), default_sun),
    ("sunbehind", (target.x + 0.8 * radius * math.cos(2.2), 180.0, target.z + 0.8 * radius * math.sin(2.2)), back_sun),
]
if os.environ.get("HSTR_VIEWS"):
    views = [v for v in views if v[0] in os.environ["HSTR_VIEWS"].split(",")]
# label, cache properties. Whole paths per batch (worldCacheSegments 0) against the persistent pool, where every thread advances
# worldCacheSegments flights per update and respawns dead photons, so no warp waits on its longest path.
configs = [
    ("paths", {}),
    ("pool 16 x 64k", {"worldCacheSegments": 16}),
    ("pool 64 x 64k", {"worldCacheSegments": 64}),
    ("pool 16 x 256k", {"worldCacheSegments": 16, "worldCachePhotons": 262144}),
    ("pool 64 x 256k", {"worldCacheSegments": 64, "worldCachePhotons": 262144}),
    # Similarity: exact for the first N scatterings, then the similar isotropic medium.
    ("similar 2", {"worldCacheSimilarity": 2}),
    ("similar 4", {"worldCacheSimilarity": 4}),
    ("similar 8", {"worldCacheSimilarity": 8}),
    ("similar 16", {"worldCacheSimilarity": 16}),
]
if os.environ.get("HSTR_CONFIGS"):
    keep = os.environ["HSTR_CONFIGS"].split(",")
    configs = [c for c in configs if c[0] in keep]
BUDGETS = [float(b) for b in os.environ.get("HSTR_BUDGETS", "200,800").split(",")]  # Cumulative tracing GPU ms at each check.
UPDATES = 4  # Cache updates per frame while accumulating.


def measure():
    hstr.set_properties({"compareTarget": 15, "compareSubstitute": 0, "compareBlock": 8})
    m.renderFrame()
    hstr.set_properties({"compareReference": True})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareBlock": 1})
    return float(p["referenceLogError"])


lines = []
for name, position, sun in views:
    cam.position = float3(*position)
    cam.target = target
    hstr.set_properties(dict(base, sunDirection=float3(*sun), worldCacheUpdates=0))
    m.renderFrame()
    hstr.set_properties({"loadReference": f"{OUT}/references/{name}_{W}x{H}"})
    m.renderFrame()
    for label, props in configs:
        settings = dict(dict(base, worldCacheSegments=0), **props)

        def restart():
            # A cell-size toggle restarts the cache (and the pool).
            hstr.set_properties(dict(settings, worldCacheUpdates=0, worldCacheCellVoxels=settings["worldCacheCellVoxels"] + 1))
            m.renderFrame()
            hstr.set_properties(dict(settings, worldCacheUpdates=UPDATES))

        # Time per update after the GPU's clocks settle, then restart and accumulate to each tracing-time budget.
        restart()
        for i in range(24):
            m.renderFrame()
        # Laptop clocks wander under sustained load: the fastest of three captures.
        per_update = 1e9
        for attempt in range(3):
            m.profiler.enabled = True
            m.profiler.start_capture()
            for i in range(8):
                m.renderFrame()
            capture = m.profiler.end_capture()
            m.profiler.enabled = False
            ms = [lane["stats"]["mean"] for key, lane in capture["events"].items() if key.endswith("/worldCache/gpu_time")]
            if ms:
                per_update = min(per_update, ms[0] / UPDATES)
        restart()
        results = []
        for budget in BUDGETS:
            while int(hstr.properties["worldCacheSampleCount"]) * per_update < budget:
                m.renderFrame()
            hstr.set_properties({"worldCacheUpdates": 0})
            results.append((budget, measure()))
            hstr.set_properties({"worldCacheUpdates": UPDATES})
        errors = "  ".join(f"{b:.0f} ms: {e:.4f}" for b, e in results)
        lines.append(f"{name:10s} {label:18s} {per_update:6.3f} ms/update  log err after tracing {errors}")
        with open(f"{OUT}/{TAG}_test.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
exit()
