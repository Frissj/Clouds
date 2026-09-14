import math
from falcor import *

# Pictures of the current split frame (single cloud) and its 4K GPU cost, converging and converged.
OUT = "C:/Users/Friss/Documents/HSTR_results"
g = RenderGraph("PictureNow")
g.addPass(createPass("HSTRCloud", {"cutTransmittanceTolerance": 0.05}), "HSTRCloud")
g.addPass(createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0}), "ToneMapper")
g.addEdge("HSTRCloud.color", "ToneMapper.src")
g.markOutput("ToneMapper.dst")
m.addGraph(g)
m.loadScene("data/HSTR/wdas_cloud.pyscene")
hstr = g.getPass("HSTRCloud")
cam = m.scene.camera
m.frameCapture.outputDir = OUT
target = float3(-10.0, 73.0, -43.0)
radius = 510.0
base = {"debugView": 8, "hstComponents": 15, "residualStrength": 1.0, "worldCacheCellVoxels": 2, "worldCacheBands": 2,
        "worldCacheOrder": 2, "worldCacheWindow": 1, "worldCacheEstimator": 1, "worldCacheTextured": 1, "worldCacheSegments": 0,
        "worldCachePhotons": 65536, "worldCacheBakeInterval": 1, "stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "lightingStride": 1}
cheap = {"stepOpticalDepth": 1.0, "minStepVoxels": 2.0, "lightingStride": 2}
back_sun = (-0.5, 0.6, 0.62)
n = math.sqrt(sum(c * c for c in back_sun))
back_sun = tuple(c / n for c in back_sun)
lines = []
m.resizeFrameBuffer(1920, 1080)
views = [
    ("side", (target.x + radius, 60.0, target.z), (0.4319, 0.8639, 0.2699)),
    ("backlit", (target.x + 0.8 * radius * math.cos(2.2), 180.0, target.z + 0.8 * radius * math.sin(2.2)), back_sun),
]
for name, position, sun in views:
    cam.position = float3(*position)
    cam.target = target
    hstr.set_properties(dict(base, sunDirection=float3(*sun), worldCacheUpdates=4))
    while int(hstr.properties["worldCacheSampleCount"]) < 32:
        m.renderFrame()
    hstr.set_properties({"worldCacheUpdates": 0})
    m.renderFrame()
    m.frameCapture.baseFilename = f"now_{name}"
    m.frameCapture.capture()


def profile(frames, orbit):
    m.profiler.enabled = True
    m.profiler.start_capture()
    for i in range(frames):
        if orbit:
            a = 0.02 * i
            cam.position = float3(target.x + radius * math.cos(a), 60.0, target.z + radius * math.sin(a))
            cam.target = target
        m.renderFrame()
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    times = {name.split("/")[-2]: lane["stats"]["mean"] for name, lane in capture["events"].items()
             if name.endswith("gpu_time") and ("HSTRCloud" in name or "ToneMapper" in name)}
    return times


m.resizeFrameBuffer(3840, 2160)
hstr.set_properties(dict(base, sunDirection=float3(0.4319, 0.8639, 0.2699)))
for label, props in (("full quality", {}), ("cheap camera", cheap)):
    # Converging: persistent pool 32k x 8 segments per frame, bake every 8 frames.
    hstr.set_properties(dict(base, **props, worldCacheUpdates=1, worldCachePhotons=32768, worldCacheSegments=8, worldCacheBakeInterval=8))
    for i in range(8):
        m.renderFrame()
    t = profile(32, True)
    lines.append(f"4K {label}, converging: HSTRCloud {t.get('HSTRCloud', 0):.2f} ms  " +
                 "  ".join(f"{k} {v:.2f}" for k, v in sorted(t.items()) if k != "HSTRCloud"))
    # Converged: no photon updates, no bake; only the camera resolve.
    hstr.set_properties({"worldCacheUpdates": 0})
    m.renderFrame()
    t = profile(32, True)
    lines.append(f"4K {label}, converged:  HSTRCloud {t.get('HSTRCloud', 0):.2f} ms  " +
                 "  ".join(f"{k} {v:.2f}" for k, v in sorted(t.items()) if k != "HSTRCloud"))
with open(OUT + "/picture_now.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
exit()
