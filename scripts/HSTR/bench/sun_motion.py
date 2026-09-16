import json
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
PROPS = json.loads(os.environ.get("HSTR_PROPS", "{}"))  # HSTRCloud properties held through the whole run (the path under test).
LANE = float(os.environ.get("HSTR_LANE", "0.5"))  # Profiler lanes at or above this mean are broken out of the total.
RUNS = os.environ.get("HSTR_RUNS", "static,sun,flight").split(",")  # Which motions to measure, for narrowing a result down.
VIEW = os.environ.get("HSTR_VIEW", "near")
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
    # The frame of the path under test against the accurate per-pixel march of the same camera and sun.
    hstr.set_properties({"debugView": 8, "minStepVoxels": 1})
    m.renderFrame()
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    hstr.set_properties(dict(PROPS, compareReference=True, compareExact=True, compareBlock=1))
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareExact": False})
    return float(p["referenceLogError"]), float(p["referenceLogP999"]), float(p["referenceLogMax"])


def run(label, frames, step):
    # GPU ms over the moving frames, then the error of where the motion ended (the compare frames switch views, which would cost the
    # beam path its temporal history mid-run).
    waiting = 0
    m.profiler.enabled = True
    m.profiler.start_capture()
    for i in range(frames):
        step(i)
        m.renderFrame()
        waiting = max(waiting, int(stats().get("sunWaiting", 0)))
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    lane = next((l for n, l in capture["events"].items() if n.endswith("HSTRCloud/gpu_time")), None)
    bake = next((l for n, l in capture["events"].items() if n.endswith("bakeCloudSun/gpu_time")), None)
    parts = " ".join(f"{n[:-len('/gpu_time')].split('HSTRCloud/', 1)[-1]} {l['stats']['mean']:.1f}"
                     for n, l in capture["events"].items()
                     if n.endswith("gpu_time") and "HSTRCloud/" in n and l["stats"]["mean"] >= LANE)
    error = compare()
    # The comparison frame reads back the beam lists, so the tile counts and marched share describe the frame just compared.
    s = stats()
    tiles = " ".join(f"{k} {s[k]}" for k in sorted(s) if k.startswith("beam"))
    log(f"{label}: GPU {lane['stats']['mean'] if lane else 0:7.2f} ms ({parts}), bake pass {bake['stats']['mean'] if bake else 0:5.2f} ms; "
        f"most bakes waiting {waiting}; marched {100 * float(hstr.properties['beamMarchedFraction']):.1f}% ({tiles}); "
        f"vs the per-pixel march at the end: log mean {error[0]:.2e}, p99.9 {error[1]:.2e}, max {error[2]:.2e}")


# Camera, its target, and the direction the flights travel in (the near view closes on one cloud, the sea flies over the field).
VIEWS = {"near": ((340, -10, 46), (6, -15, 46), (-1, 0, 0)), "sea": ((0, 140, 0), (0, 40, 600), (0, 0, 1))}
EYE, AIM, HEADING = VIEWS[VIEW]
cam.position = float3(*EYE)
cam.target = float3(*AIM)
hstr.set_properties(dict(PROPS, debugView=8, hstComponents=15, worldCacheUpdates=1, cloudSunCache=True))
settle()
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0})
s = stats()
log(f"settled: {s.get('sunBaked')} bakes, {s.get('sunWaiting')} waiting, sun bake angle {hstr.properties['cloudSunBakeAngle']} degrees, "
    f"{hstr.properties['cloudSunBakesPerFrame']} bakes per frame")
if "static" in RUNS:
    err = compare()
    log(f"static: vs the per-pixel march: log mean {err[0]:.2e}, p99.9 {err[1]:.2e}, max {err[2]:.2e}")
    run("static", 48, lambda i: None)
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 2, "compareMapScale": 0.02})
    m.renderFrame()
    m.frameCapture.baseFilename = f"{TAG}_static_error_map"
    m.frameCapture.capture()
    hstr.set_properties({"compareReference": False, "compareExact": False, "compareMapScale": 0.0})

if "sun" in RUNS:
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

if "flight" in RUNS:
    for label, speed in (("flight 1/frame", 1.0), ("flight 4/frame", 4.0)):
        def fly(i, v=speed):
            cam.position = float3(EYE[0] + HEADING[0] * v * i, EYE[1] + HEADING[1] * v * i, EYE[2] + HEADING[2] * v * i)
            cam.target = float3(AIM[0] + HEADING[0] * v * i, AIM[1] + HEADING[1] * v * i, AIM[2] + HEADING[2] * v * i)
        run(label, 120, fly)
exit()
