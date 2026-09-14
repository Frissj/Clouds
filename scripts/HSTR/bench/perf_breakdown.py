import json
import os
from falcor import *

# 4K timing of the split-frame resolve under variants, with a converged cache and no cache updates.
OUT = "C:/Users/Friss/Documents/HSTR_results/perf_breakdown.txt"
m.script("scripts/HSTR/HSTRCloud.py")
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(3840, 2160)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
cam.position = float3(500.0, 60.0, -43.0)
cam.target = float3(-10.0, 73.0, -43.0)
base = {"debugView": 8, "residualStrength": 1.0, "worldCacheCellVoxels": 2, "worldCacheBands": 2, "worldCacheOrder": 2,
        "worldCacheWindow": 1, "worldCacheEstimator": 1, "worldCachePhotons": 262144, "hstComponents": 15, "worldCacheTextured": 1,
        "stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "maxStepVoxels": 4.0}
hstr.set_properties(dict(base, worldCacheUpdates=4))
while int(hstr.properties["worldCacheSampleCount"]) < 32:
    m.renderFrame()

base["lightingStride"] = 1
variants = [
    ("full (textured, order 2w)", {}),
    ("stride 2", {"lightingStride": 2}),
    ("stride 4", {"lightingStride": 4}),
    ("coarse steps", {"stepOpticalDepth": 1.0, "minStepVoxels": 2.0}),
    ("coarse steps + stride 2", {"stepOpticalDepth": 1.0, "minStepVoxels": 2.0, "lightingStride": 2}),
    ("order 1 windowed", {"worldCacheOrder": 1}),
    ("order 0", {"worldCacheOrder": 0}),
    ("background only (transmittance march)", {"hstComponents": 1}),
    ("coarse steps, background only", {"stepOpticalDepth": 1.0, "minStepVoxels": 2.0, "hstComponents": 1}),
    ("current HST frame (debugView 0)", {"debugView": 0, "residualStrength": 1.4}),
    ("full, repeated for clock drift", {}),
]
lines = []
for label, props in variants:
    hstr.set_properties(dict(base, worldCacheUpdates=0, **props))
    for i in range(4):
        m.renderFrame()
    m.profiler.enabled = True
    m.profiler.start_capture()
    for i in range(20):
        m.renderFrame()
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    times = {name.split("/")[-2]: lane["stats"]["mean"] for name, lane in capture["events"].items() if name.endswith("gpu_time") and "HSTRCloud" in name}
    lines.append(f"{label:45s} total {times.get('HSTRCloud', float('nan')):7.2f} ms  resolve {times.get('resolve', float('nan')):7.2f} ms")
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
exit()
