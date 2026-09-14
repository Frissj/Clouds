import math
import os
from falcor import *

# Whole-frame GPU cost of the cloud at 4K: beam view (h16x3, tolerance 0.05) with the cache static, and with a world-cache
# update of N photons every frame (similarity tracing, followed by the texture bake).
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "frame")
W, H = int(os.environ.get("HSTR_W", "3840")), int(os.environ.get("HSTR_H", "2160"))
g = RenderGraph("FrameCost")
g.addPass(createPass("HSTRCloud", {"cutTransmittanceTolerance": 0.05}), "HSTRCloud")
g.addPass(createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0}), "ToneMapper")
g.addEdge("HSTRCloud.color", "ToneMapper.src")
g.markOutput("ToneMapper.dst")
m.addGraph(g)
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(W, H)
hstr = g.getPass("HSTRCloud")
cam = m.scene.camera
target = float3(-10.0, 73.0, -43.0)
radius = 510.0
base = {"hstComponents": 15, "residualStrength": 1.0, "worldCacheCellVoxels": 2, "worldCacheBands": 2, "worldCacheOrder": 2,
        "worldCacheWindow": 1, "worldCacheEstimator": 1, "worldCacheTextured": 1, "worldCacheSegments": 0, "worldCacheBakeInterval": 1,
        "stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "lightingStride": 1, "beamSegments": 1, "beamEdgeDepth": 0.0,
        "debugView": 9, "beamTileSize": 16, "beamLevels": 3, "beamTolerance": 0.05}
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
# label, photons per frame (0: static cache)
# label, photons per frame (0: static cache; with segments: pool slots), frames between bakes, pool flights per frame (0: whole
# paths, whose dispatch lasts as long as its longest photon path)
configs = [("static cache", 0, 1, 0), ("16k bake1", 16384, 1, 0), ("65k bake1", 65536, 1, 0), ("1k bake8", 1024, 8, 0),
           ("4k bake8", 4096, 8, 0), ("16k bake8", 16384, 8, 0), ("32k bake8", 32768, 8, 0), ("65k bake8", 65536, 8, 0),
           ("pool 64k x 4 bake8", 65536, 8, 4), ("64 bake8", 64, 8, 0)]
if os.environ.get("HSTR_BASE"):
    import json
    base.update(json.loads(os.environ["HSTR_BASE"]))
if os.environ.get("HSTR_CONFIGS"):
    keep = os.environ["HSTR_CONFIGS"].split(",")
    configs = [c for c in configs if c[0] in keep]


def gpu_times(frames):
    for i in range(32):  # Warm-up: the laptop GPU's clocks settle.
        m.renderFrame()
    best = None
    for attempt in range(3):
        m.profiler.enabled = True
        m.profiler.start_capture()
        for i in range(frames):
            m.renderFrame()
        capture = m.profiler.end_capture()
        m.profiler.enabled = False
        t = {name[:-len("/gpu_time")].split("HSTRCloud", 1)[-1].strip("/") or "HSTRCloud": lane["stats"]["mean"]
             for name, lane in capture["events"].items() if name.endswith("gpu_time") and "HSTRCloud" in name}
        if best is None or t.get("HSTRCloud", 1e9) < best.get("HSTRCloud", 1e9):
            best = t
    return best


lines = []
for name, position, sun in views:
    cam.position = float3(*position)
    cam.target = target
    hstr.set_properties(dict(base, sunDirection=float3(*sun), worldCachePhotons=65536, worldCacheUpdates=4))
    while int(hstr.properties["worldCacheSampleCount"]) < 32:
        m.renderFrame()
    for label, photons, bake_interval, segments in configs:
        if photons:
            hstr.set_properties({"worldCachePhotons": photons, "worldCacheUpdates": 1, "worldCacheBakeInterval": bake_interval,
                                 "worldCacheSegments": segments})
        else:
            hstr.set_properties({"worldCacheUpdates": 0})
        t = gpu_times(16)
        bake = t.get("worldCache/worldCacheBake", 0)
        parts = {"photons": t.get("worldCache", 0) - bake, "bake": bake, "beam": t.get("beamQueries", 0) + t.get("resolve", 0)}
        lines.append(f"{name:10s} {label:18s} frame {t.get('HSTRCloud', 0):5.2f} ms  (photons {parts['photons']:4.2f}, bake {parts['bake']:4.2f}, "
                     f"beam {parts['beam']:4.2f})")
        with open(f"{OUT}/{TAG}_test.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
    # A picture: the beam view after 256 updates of 65536 photons.
    if os.environ.get("HSTR_NOPICTURE"):
        continue
    hstr.set_properties({"worldCachePhotons": 65536, "worldCacheUpdates": 8, "worldCacheBakeInterval": 1, "worldCacheSegments": 0})
    while int(hstr.properties["worldCacheSampleCount"]) < 256:
        m.renderFrame()
    hstr.set_properties({"worldCacheUpdates": 0})
    m.renderFrame()
    m.frameCapture.outputDir = OUT
    m.frameCapture.baseFilename = f"{TAG}_{name}_picture"
    m.frameCapture.capture()
exit()
