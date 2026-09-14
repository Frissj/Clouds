import math
import os
from falcor import *

# Beam view (per-tile exact queries, reconstructed pixels) against the exact per-pixel split frame at 4K: error, share of
# pixels still marched, and GPU time.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "beam")
W, H = int(os.environ.get("HSTR_W", "3840")), int(os.environ.get("HSTR_H", "2160"))
g = RenderGraph("BeamTest")
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
        "worldCacheBakeInterval": 1, "stepOpticalDepth": 1.0, "minStepVoxels": 2.0, "lightingStride": 2}
default_sun = (0.4319, 0.8639, 0.2699)
back_sun = (-0.5, 0.6, 0.62)
n = math.sqrt(sum(c * c for c in back_sun))
back_sun = tuple(c / n for c in back_sun)
views = [
    ("side", (target.x + radius, 60.0, target.z), default_sun),
    ("far", (target.x + 3.0 * radius, 60.0, target.z), default_sun),
    ("sunbehind", (target.x + 0.8 * radius * math.cos(2.2), 180.0, target.z + 0.8 * radius * math.sin(2.2)), back_sun),
]
configs = [
    # label, tile, levels, segments, tolerance, edge depth
    ("flat 8", 8, 1, 4, 0.02, 0.0),
    ("flat 16", 16, 1, 4, 0.02, 0.0),
    ("h16x3", 16, 3, 4, 0.02, 0.0),
    ("h16x3 seg1", 16, 3, 1, 0.02, 0.0),
    ("h16x3 seg8", 16, 3, 8, 0.02, 0.0),
    ("h16x4", 16, 4, 4, 0.02, 0.0),
    ("h32x4", 32, 4, 4, 0.02, 0.0),
    ("h16x3 tol.05", 16, 3, 4, 0.05, 0.0),
    ("h32x4 tol.05", 32, 4, 4, 0.05, 0.0),
]
if os.environ.get("HSTR_BASE"):
    import json
    base.update(json.loads(os.environ["HSTR_BASE"]))
if os.environ.get("HSTR_VIEWS"):
    views = [v for v in views if v[0] in os.environ["HSTR_VIEWS"].split(",")]
if os.environ.get("HSTR_CONFIGS"):
    keep = os.environ["HSTR_CONFIGS"].split(",")
    configs = [c for c in configs if c[0] in keep]


def gpu_times(frames):
    m.profiler.enabled = True
    m.profiler.start_capture()
    for i in range(frames):
        m.renderFrame()
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    return {name.split("/")[-2]: lane["stats"]["mean"] for name, lane in capture["events"].items()
            if name.endswith("gpu_time") and "HSTRCloud" in name}


def compare():
    hstr.set_properties({"compareReference": True, "compareExact": True})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False})
    return float(p["referenceLogError"]), float(p["referenceNoiseError"]), float(p["referenceNoiseLogError"]), float(p["beamMarchedFraction"])


lines = []
current_sun = None
for name, position, sun in views:
    cam.position = float3(*position)
    cam.target = target
    if sun != current_sun:
        hstr.set_properties(dict(base, debugView=8, sunDirection=float3(*sun), worldCacheUpdates=4))
        while int(hstr.properties["worldCacheSampleCount"]) < 32:
            m.renderFrame()
        current_sun = sun
    hstr.set_properties(dict(base, debugView=8, worldCacheUpdates=0))
    m.renderFrame()
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    t = gpu_times(16)
    lines.append(f"{name:10s} {'exact per pixel':18s} resolve {t.get('resolve', 0):5.2f} ms  total {t.get('HSTRCloud', 0):5.2f} ms")
    if name == "side":
        m.frameCapture.baseFilename = f"{TAG}_{name}_exact"
        m.frameCapture.capture()
    for label, tile, levels, segments, tolerance, edge in configs:
        hstr.set_properties({"debugView": 9, "beamTileSize": tile, "beamLevels": levels, "beamSegments": segments, "beamTolerance": tolerance,
                             "beamEdgeDepth": edge, "compareExact": False})
        m.renderFrame()
        m.renderFrame()
        log_error, over02, over10, marched = compare()
        t = gpu_times(16)
        lines.append(f"{name:10s} {label:18s} resolve {t.get('resolve', 0):5.2f} ms  queries {t.get('beamQueries', 0):5.2f} ms  "
                     f"total {t.get('HSTRCloud', 0):5.2f} ms  log err {log_error:.4f}  >0.02 {100 * over02:5.2f}%  "
                     f">0.1 {100 * over10:5.2f}%  marched {100 * marched:5.1f}%")
        if name == "side" and label in ("h16x3", "h32x4 tol.05"):
            m.frameCapture.baseFilename = f"{TAG}_{name}_{label.replace(' ', '_')}"
            m.frameCapture.capture()
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
exit()
