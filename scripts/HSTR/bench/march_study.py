import math
import os
from falcor import *

# Camera integration of the cache frame against the 4K path tracer: fixed majorant-bounded steps against the adaptive analytic
# march (integrateCachedSegment), for the per-pixel split view (8) and the beam view (9). One cache (auto modulation) serves
# every configuration; errors are 8x8-block log errors of the full frame and its components, times are GPU means.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "march")
W, H = int(os.environ.get("HSTR_W", "3840")), int(os.environ.get("HSTR_H", "2160"))
BLOCK = 8
g = RenderGraph("MarchStudy")
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
        "worldCacheBakeInterval": 1, "lightingStride": 1, "worldCacheModulation": -1.0, "maxMarchSteps": 8192,
        "sunNearVoxels": 0.0, "beamTileSize": 16, "beamLevels": 3, "beamSegments": 1, "beamTolerance": 0.05}
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
fixed = {"adaptiveMarch": False, "stepOpticalDepth": 0.5, "maxStepVoxels": 4.0}
adaptive = {"adaptiveMarch": True, "stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "maxStepVoxels": 4.0, "marchMinVoxels": 0.03,
            "marchCoarseVoxels": 2.0, "marchTolerance": 0.005}
# label, march properties, views to run (8 split per pixel, 9 beam)
configs = [
    ("fixed 1", dict(fixed, minStepVoxels=1.0), (8, 9)),
    ("fixed 0.1", dict(fixed, minStepVoxels=0.1), (8, 9)),
    ("adaptive", dict(adaptive, marchTolerance=0.01, sunNearVoxels=0.0), (8, 9)),
    ("adaptive tol.005", dict(adaptive, sunNearVoxels=0.0), (8, 9)),
    ("adaptive tol.002", dict(adaptive, marchTolerance=0.002, sunNearVoxels=0.0), (8, 9)),
    ("adaptive exact sun", dict(adaptive, marchTolerance=0.01, sunNearVoxels=2.0), (8, 9)),
    ("adaptive coarse4", dict(adaptive, marchTolerance=0.01, marchCoarseVoxels=4.0, maxStepVoxels=8.0, sunNearVoxels=0.0), (8, 9)),
]
if os.environ.get("HSTR_CONFIGS"):
    keep = os.environ["HSTR_CONFIGS"].split(",")
    configs = [c for c in configs if c[0] in keep]


def gpu_times(frames):
    for i in range(8):  # Warm-up: the laptop GPU clocks settle.
        m.renderFrame()
    m.profiler.enabled = True
    m.profiler.start_capture()
    for i in range(frames):
        m.renderFrame()
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    times = {name[:-len("/gpu_time")].split("HSTRCloud", 1)[-1].strip("/") or "HSTRCloud": lane["stats"]["mean"]
             for name, lane in capture["events"].items() if name.endswith("gpu_time") and "HSTRCloud" in name}
    return times


def measure(mask):
    hstr.set_properties({"hstComponents": mask, "compareTarget": mask, "compareSubstitute": 0, "compareBlock": BLOCK})
    m.renderFrame()
    hstr.set_properties({"compareReference": True})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareBlock": 1, "hstComponents": 15})
    return float(p["referenceLogError"])


lines = []
for name, position, sun in views:
    path = f"{OUT}/references/{name}_{W}x{H}"
    cam.position = float3(*position)
    cam.target = target
    hstr.set_properties(dict(base, debugView=8, sunDirection=float3(*sun), worldCacheUpdates=0))
    m.renderFrame()
    hstr.set_properties({"loadReference": path})
    m.renderFrame()
    # One cache for every configuration: a cell-size toggle restarts it, then 512 batches of 65536 photons.
    hstr.set_properties({"worldCacheCellVoxels": 3})
    m.renderFrame()
    hstr.set_properties({"worldCacheCellVoxels": 2, "worldCacheUpdates": 8})
    while int(hstr.properties["worldCacheSampleCount"]) < 512:
        m.renderFrame()
    hstr.set_properties({"worldCacheUpdates": 0})
    lines.append(f"{name}: reference {hstr.properties['referenceSampleCount']} spp, cache {hstr.properties['worldCacheSampleCount']} batches")
    for label, props, debug_views in configs:
        for view in debug_views:
            hstr.set_properties(dict(props, debugView=view))
            t = gpu_times(16)
            full, cache, single = measure(15), measure(12), measure(2)
            lines.append(f"{name:10s} view {view} {label:22s} {t.get('HSTRCloud', 0):6.2f} ms  log err full {full:.4f}  cache {cache:.4f}  "
                         f"single {single:.4f}")
            if view == 9:
                parts = {k: v for k, v in t.items() if k != "HSTRCloud" and v > 0.05}
                lines.append(f"{'':10s} {'':30s} " + "  ".join(f"{k} {v:.2f}" for k, v in sorted(parts.items())) +
                             f"  marched {100 * float(hstr.properties['beamMarchedFraction']):.1f}%")
            m.renderFrame()
            m.frameCapture.baseFilename = f"{TAG}_{name}_v{view}_{label.replace(' ', '_')}"
            m.frameCapture.capture()
            with open(f"{OUT}/{TAG}_test.txt", "w") as f:
                f.write("\n".join(lines) + "\n")
exit()
