import json
import math
import os
from falcor import *

# Per-pixel marching and hierarchical beam tiles against an accurate per-pixel march of the same split model (ground truth)
# at 4K: error, share of pixels still marched, GPU time.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "truth")
W, H = int(os.environ.get("HSTR_W", "3840")), int(os.environ.get("HSTR_H", "2160"))
g = RenderGraph("BeamTruth")
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
        "worldCacheBakeInterval": 1, "beamSegments": 1, "beamEdgeDepth": 0.0}
TRUTH = {"stepOpticalDepth": 0.25, "minStepVoxels": 0.5, "lightingStride": 1, "adaptiveMarch": False}
FULL = {"stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "lightingStride": 1, "adaptiveMarch": False}
CHEAP = {"stepOpticalDepth": 1.0, "minStepVoxels": 2.0, "lightingStride": 2}
MID = {"stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "lightingStride": 2}
default_sun = (0.4319, 0.8639, 0.2699)
back_sun = (-0.5, 0.6, 0.62)
n = math.sqrt(sum(c * c for c in back_sun))
back_sun = tuple(c / n for c in back_sun)
views = [
    ("side", (target.x + radius, 60.0, target.z), default_sun),
    ("far", (target.x + 3.0 * radius, 60.0, target.z), default_sun),
    ("sunbehind", (target.x + 0.8 * radius * math.cos(2.2), 180.0, target.z + 0.8 * radius * math.sin(2.2)), back_sun),
]


def beam(tile, levels, tolerance, march):
    return dict(march, debugView=9, beamTileSize=tile, beamLevels=levels, beamTolerance=tolerance)


configs = [
    ("flat8 full", beam(8, 1, 0.02, FULL)),
    ("h16x3 full", beam(16, 3, 0.02, FULL)),
    ("h16x3 full .05", beam(16, 3, 0.05, FULL)),
    ("h32x4 full", beam(32, 4, 0.02, FULL)),
    ("h32x4 full .05", beam(32, 4, 0.05, FULL)),
    ("h16x3 mid", beam(16, 3, 0.02, MID)),
    ("h16x3 mid .05", beam(16, 3, 0.05, MID)),
    ("h32x4 mid .05", beam(32, 4, 0.05, MID)),
    ("h16x3 truth", beam(16, 3, 0.02, TRUTH)),
    # Changes since the beam commit, one at a time (properties unknown to older builds are ignored with a warning).
    ("h16x3 full .05 nomod", dict(beam(16, 3, 0.05, FULL), worldCacheModulation=0.0)),
    ("h16x3 full .05 nomod 512", dict(beam(16, 3, 0.05, FULL), worldCacheModulation=0.0, maxMarchSteps=512)),
    ("h16x3 adaptive .05", dict(beam(16, 3, 0.05, FULL), adaptiveMarch=True, marchTolerance=0.01, marchCoarseVoxels=2.0)),
    # Analytic steps between sample points, no refinement: one sample per step like the fixed march.
    ("h16x3 analytic1 .05", dict(beam(16, 3, 0.05, FULL), adaptiveMarch=True, marchTolerance=0.0, marchCoarseVoxels=1.0)),
    ("h16x3 analytic2 .05", dict(beam(16, 3, 0.05, FULL), adaptiveMarch=True, marchTolerance=0.0, marchCoarseVoxels=2.0)),
    # Per-pixel marching last: its heavy frames throttle the laptop GPU for the configurations after them.
    ("pixel cheap", dict(CHEAP, debugView=8)),
    ("pixel mid", dict(MID, debugView=8)),
    ("pixel full", dict(FULL, debugView=8)),
]
if os.environ.get("HSTR_VIEWS"):
    views = [v for v in views if v[0] in os.environ["HSTR_VIEWS"].split(",")]
if os.environ.get("HSTR_CONFIGS"):
    keep = os.environ["HSTR_CONFIGS"].split(",")
    configs = [c for c in configs if c[0] in keep]
CAPTURE = os.environ.get("HSTR_CAPTURE", "").split(",")


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
    hstr.set_properties(dict(base, **TRUTH, debugView=8, worldCacheUpdates=0))
    m.renderFrame()
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    if name in CAPTURE:
        m.frameCapture.baseFilename = f"{TAG}_{name}_truth"
        m.frameCapture.capture()
    for label, props in configs:
        hstr.set_properties(dict(props, compareExact=False))
        m.renderFrame()
        m.renderFrame()
        log_error, over02, over10, marched = compare()
        # Warm-up: the laptop GPU's clocks settle over a few dozen frames after a change of load.
        for i in range(32):
            m.renderFrame()
        t = gpu_times(32)
        marched_text = f"marched {100 * marched:5.1f}%" if props["debugView"] == 9 else ""
        levels = " ".join(f"{t[k]:4.2f}" for k in ("queries0", "queries1", "queries2", "queries3") if k in t)
        lines.append(f"{name:10s} {label:16s} total {t.get('HSTRCloud', 0):5.2f} ms  (resolve {t.get('resolve', 0):5.2f} of which march "
                     f"{t.get('march', 0):4.2f}; beam {t.get('beamQueries', 0):5.2f}, queries by level {levels})  log err {log_error:.4f}  >0.02 {100 * over02:5.2f}%  >0.1 {100 * over10:5.2f}%  "
                     f"{marched_text}")
        if name in CAPTURE:
            m.frameCapture.baseFilename = f"{TAG}_{name}_{label.replace(' ', '_')}"
            m.frameCapture.capture()
        with open(f"{OUT}/{TAG}_test.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
exit()
