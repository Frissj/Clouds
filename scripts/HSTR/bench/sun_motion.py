import math
import os
from falcor import *

# Baked sun depth while things move, per-pixel view at 4K from the settled near view: GPU ms and bake backlog through a slow and a
# fast sun sweep and a camera flight, with the cached frame compared against the live sun march of the same frame every few frames
# (the live frame renders first, so the residency is one frame older). Also a block error map of the settled view, for contours at
# brick level-of-detail boundaries.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "sun_motion")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(3840, 2160)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
m.frameCapture.outputDir = OUT
lines = []


def log(line):
    lines.append(line)
    print(line)
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def stats():
    return hstr.properties.get("cloudStats", {})


def settle():
    quiet = 0
    for i in range(1500):
        m.renderFrame()
        s = stats()
        busy = int(s.get("pendingTiles", 0)) + int(s.get("pending", 0)) + int(s.get("sunBakesFrame", 0))
        quiet = quiet + 1 if busy == 0 else 0
        if i > 32 and quiet > 20:
            return i
    return 1500


def compare():
    hstr.set_properties({"cloudSunCache": False})
    m.renderFrame()
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    hstr.set_properties({"cloudSunCache": True, "compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareExact": False})
    return float(p["referenceLogError"]), float(p["referenceLogP999"]), float(p["referenceLogMax"])


def run(label, frames, step, every=24):
    # GPU ms over the moving frames (compare frames excluded), and the error at every `every` frames.
    total, count, errors, waiting = 0.0, 0, [], 0
    for start in range(0, frames, every):
        m.profiler.enabled = True
        m.profiler.start_capture()
        for i in range(start, min(frames, start + every)):
            step(i)
            m.renderFrame()
        capture = m.profiler.end_capture()
        m.profiler.enabled = False
        lane = next((l for n, l in capture["events"].items() if n.endswith("HSTRCloud/gpu_time")), None)
        bake = next((l for n, l in capture["events"].items() if n.endswith("bakeCloudSun/gpu_time")), None)
        if lane:
            total += lane["stats"]["mean"]
            count += 1
        waiting = max(waiting, int(stats().get("sunWaiting", 0)))
        errors.append(compare())
    mean = sum(e[0] for e in errors) / len(errors)
    log(f"{label}: GPU {total / max(count, 1):7.2f} ms, bake pass {bake['stats']['mean'] if bake else 0:5.2f} ms; most bakes waiting {waiting}; "
        f"cached vs live log mean {mean:.2e}, worst p99.9 {max(e[1] for e in errors):.2e}, worst max {max(e[2] for e in errors):.2e}")


cam.position = float3(340, -10, 46)
cam.target = float3(6, -15, 46)
hstr.set_properties({"debugView": 8, "hstComponents": 15, "worldCacheUpdates": 1, "cloudSunCache": True})
settle()
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0})
s = stats()
log(f"settled: {s.get('sunBaked')} bakes, {s.get('sunWaiting')} waiting, sun bake angle {hstr.properties['cloudSunBakeAngle']} degrees, "
    f"{hstr.properties['cloudSunBakesPerFrame']} bakes per frame")
err = compare()
log(f"static: cached vs live log mean {err[0]:.2e}, p99.9 {err[1]:.2e}, max {err[2]:.2e}")
hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 2, "compareMapScale": 0.02})
m.renderFrame()
m.frameCapture.baseFilename = f"{TAG}_static_error_map"
m.frameCapture.capture()
hstr.set_properties({"compareReference": False, "compareExact": False, "compareMapScale": 0.0})

base = [float(c) for c in hstr.properties["sunDirection"]]
azimuth = math.atan2(base[2], base[0])
horizontal = math.hypot(base[0], base[2])
for label, degrees in (("sun sweep 0.05 deg/frame", 0.05), ("sun sweep 0.5 deg/frame", 0.5)):
    def sweep(i, d=degrees):
        a = azimuth + math.radians(d * i)
        hstr.set_properties({"sunDirection": float3(math.cos(a) * horizontal, base[1], math.sin(a) * horizontal)})
    run(label, 120, sweep)
hstr.set_properties({"sunDirection": float3(*base)})
settle()

for label, speed in (("flight 1/frame", 1.0), ("flight 4/frame", 4.0)):
    def fly(i, v=speed):
        cam.position = float3(340 - v * i, -10, 46)
        cam.target = float3(6 - v * i, -15, 46)
    run(label, 120, fly)
exit()
