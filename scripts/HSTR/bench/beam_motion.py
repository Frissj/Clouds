import math
import os
from falcor import *

# Beam tiles with an orbiting camera at 4K: per-level queries against temporal queries (last frame's refinement), error of
# every frame against the accurate per-pixel split frame of the same camera, and GPU time while moving and while static.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "motion")
W, H = int(os.environ.get("HSTR_W", "3840")), int(os.environ.get("HSTR_H", "2160"))
STEP = float(os.environ.get("HSTR_ORBIT_STEP", "0.01"))  # Radians per frame.
g = RenderGraph("BeamMotion")
g.addPass(createPass("HSTRCloud", {"cutTransmittanceTolerance": 0.05}), "HSTRCloud")
g.markOutput("HSTRCloud.color")
m.addGraph(g)
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(W, H)
hstr = g.getPass("HSTRCloud")
cam = m.scene.camera
target = float3(-10.0, 73.0, -43.0)
radius = 510.0
FULL = {"stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "lightingStride": 1}
base = dict(FULL, hstComponents=15, residualStrength=1.0, worldCacheCellVoxels=2, worldCacheBands=2, worldCacheOrder=2, worldCacheWindow=1,
            worldCacheEstimator=1, worldCacheTextured=1, worldCacheSegments=0, worldCachePhotons=65536, worldCacheBakeInterval=1,
            beamSegments=1, beamTileSize=16, beamLevels=3, beamTolerance=0.05, compareExact=False)


def place(angle):
    cam.position = float3(target.x + radius * math.cos(angle), 60.0, target.z + radius * math.sin(angle))
    cam.target = target


def gpu_times(frames, moving, start):
    m.profiler.enabled = True
    m.profiler.start_capture()
    for i in range(frames):
        if moving:
            place(start + STEP * i)
        m.renderFrame()
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    return {name.split("/")[-2]: lane["stats"]["mean"] for name, lane in capture["events"].items()
            if name.endswith("gpu_time") and "HSTRCloud" in name}


place(0.0)
hstr.set_properties(dict(base, debugView=8, worldCacheUpdates=4))
while int(hstr.properties["worldCacheSampleCount"]) < 32:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0})
lines = []
for label, temporal in (("per-level", False), ("temporal", True)):
    hstr.set_properties({"debugView": 9, "beamTemporal": temporal})
    place(0.0)
    for i in range(4):
        m.renderFrame()
    # Error of every frame of a 24-frame orbit against the accurate per-pixel frame of the same camera.
    errors = []
    for i in range(1, 25):
        place(STEP * i)
        hstr.set_properties({"debugView": 8})
        m.renderFrame()
        hstr.set_properties({"storeExact": True})
        m.renderFrame()
        hstr.set_properties({"debugView": 9, "compareReference": True, "compareExact": True})
        m.renderFrame()
        p = hstr.properties
        hstr.set_properties({"compareReference": False, "compareExact": False})
        errors.append((float(p["referenceLogError"]), float(p["referenceNoiseLogError"]), float(p["beamMarchedFraction"])))
    mean = [sum(e[k] for e in errors) / len(errors) for k in range(3)]
    worst = max(e[0] for e in errors)
    lines.append(f"{label:10s} orbit {STEP} rad/frame: log err mean {mean[0]:.4f} worst {worst:.4f}  >0.1 {100 * mean[1]:.3f}%  "
                 f"marched {100 * mean[2]:.2f}%")
    for i in range(32):
        place(STEP * i)
        m.renderFrame()
    t = gpu_times(48, True, 0.32)
    lines.append(f"{label:10s} moving:  total {t.get('HSTRCloud', 0):5.2f} ms  beam {t.get('beamQueries', 0):5.2f}  queries0 "
                 f"{t.get('queries0', 0):4.2f}  resolve {t.get('resolve', 0):4.2f}  march {t.get('march', 0):4.2f}")
    for i in range(8):
        m.renderFrame()
    t = gpu_times(48, False, 0.0)
    lines.append(f"{label:10s} static:  total {t.get('HSTRCloud', 0):5.2f} ms  beam {t.get('beamQueries', 0):5.2f}  resolve "
                 f"{t.get('resolve', 0):4.2f}  march {t.get('march', 0):4.2f}")
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
exit()
