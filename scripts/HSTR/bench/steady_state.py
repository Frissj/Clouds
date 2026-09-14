import math
from falcor import *

# Per-frame GPU cost of the split frame at 4K with an orbiting camera and a fixed photon budget per frame.
OUT = "C:/Users/Friss/Documents/HSTR_results/steady_state.txt"
m.script("scripts/HSTR/HSTRCloud.py")
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(3840, 2160)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
target = float3(-10.0, 73.0, -43.0)
radius = 510.0
base = {"debugView": 8, "residualStrength": 1.0, "worldCacheCellVoxels": 2, "worldCacheBands": 2, "worldCacheOrder": 2, "worldCacheWindow": 1,
        "worldCacheEstimator": 1, "worldCacheTextured": 1, "hstComponents": 15, "stepOpticalDepth": 0.5, "minStepVoxels": 1.0,
        "lightingStride": 1, "worldCacheBakeInterval": 1}
cheap = {"stepOpticalDepth": 1.0, "minStepVoxels": 2.0, "lightingStride": 2}
configs = [
    ("whole paths 16k/frame, bake every 8", {"worldCachePhotons": 16384, "worldCacheBakeInterval": 8, "worldCacheSegments": 0}),
    ("pool 32k x 8 segments, bake every 8", {"worldCachePhotons": 32768, "worldCacheSegments": 8, "worldCacheBakeInterval": 8}),
    ("pool 65k x 8 segments, bake every 8", {"worldCachePhotons": 65536, "worldCacheSegments": 8, "worldCacheBakeInterval": 8}),
    ("pool 65k x 16 segments, bake every 8", {"worldCachePhotons": 65536, "worldCacheSegments": 16, "worldCacheBakeInterval": 8}),
    ("pool 65k x 8, bake every 8, cheap camera", dict(cheap, worldCachePhotons=65536, worldCacheSegments=8, worldCacheBakeInterval=8)),
]
lines = []
for label, props in configs:
    hstr.set_properties(dict(base, worldCacheUpdates=1, **props))
    for i in range(8):
        m.renderFrame()
    m.profiler.enabled = True
    m.profiler.start_capture()
    for i in range(48):
        a = 0.02 * i
        cam.position = float3(target.x + radius * math.cos(a), 60.0, target.z + radius * math.sin(a))
        cam.target = target
        m.renderFrame()
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    times = {name.split("/")[-2]: lane["stats"]["mean"] for name, lane in capture["events"].items() if name.endswith("gpu_time") and "HSTRCloud" in name}
    parts = "  ".join(f"{k} {v:.2f}" for k, v in sorted(times.items()) if k != "HSTRCloud")
    lines.append(f"{label:50s} total {times.get('HSTRCloud', float('nan')):6.2f} ms  ({parts})")
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
exit()
