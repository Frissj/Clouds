import json
import os
from falcor import *

# Beam view of the cloud sea with a moving camera at 4K: GPU ms while flying, and the error of every step of a flight against the
# exact per-pixel frame of the same camera (the stored frame of HSTR_BASE, as in ab_test). A parked camera hides everything a
# temporal beam gets wrong, so this is the benchmark for beamRefresh. Every test flies the same path from the same start: a warm-up
# that builds its history, a timed flight, and a flight of compared steps. HSTR_MOTIONS lists the camera motions as
# [label, forward units per frame, yaw radians per frame]. Results go to <tag>_test.txt and <tag>_test.jsonl.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "sea_motion")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
BASE = json.loads(os.environ["HSTR_BASE"])
TESTS = json.loads(os.environ["HSTR_TESTS"])
MOTIONS = json.loads(os.environ.get("HSTR_MOTIONS", '[["fly 2", 2.0, 0.0], ["fly 20", 20.0, 0.0], ["yaw", 0.0, 0.004]]'))
WARM, TIMED, STEPS = 16, 48, int(os.environ.get("HSTR_STEPS", "12"))
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(3840, 2160)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
START_POSITION, START_TARGET = float3(0, 140, 0), float3(0, 40, 600)
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


def pose(frame, forward, yaw):
    # Frame 0 is the settled view; the path runs through it, so every flight starts WARM frames before it.
    import math
    offset = float3(0, 0, forward * frame)
    angle = yaw * frame
    view = START_TARGET - START_POSITION
    turned = float3(view.x * math.cos(angle) - view.z * math.sin(angle), view.y, view.x * math.sin(angle) + view.z * math.cos(angle))
    cam.position = START_POSITION + offset
    cam.target = START_POSITION + offset + turned


def timed(frames, first, forward, yaw):
    import time
    m.profiler.enabled = True
    m.profiler.start_capture()
    wall = time.perf_counter()
    for i in range(frames):
        pose(first + i, forward, yaw)
        m.renderFrame()
    wall = (time.perf_counter() - wall) / frames * 1000.0
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    t = {name[:-len("/gpu_time")].split("HSTRCloud", 1)[-1].strip("/") or "HSTRCloud": lane["stats"]["mean"]
         for name, lane in capture["events"].items() if name.endswith("gpu_time") and "HSTRCloud" in name}
    t["wall"] = wall
    t["cpu"] = {name[:-len("/cpu_time")].split("cloudSea", 1)[-1].strip("/") or "cloudSea": round(lane["stats"]["mean"], 2)
                for name, lane in capture["events"].items() if name.endswith("cpu_time") and "cloudSea" in name}
    s = stats()
    t["residency"] = {k: s.get(k, 0) for k in ("loaded", "mapped", "desired", "pending", "cutMs", "sunWaiting", "sunBakesFrame", "cutPops",
                                               "cutOrdered", "cutTotalMs")}
    return t


pose(0, 0.0, 0.0)
hstr.set_properties(dict(BASE, debugView=9, hstComponents=15, worldCacheUpdates=1))
frames = settle()
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
# The resident set is frozen once settled: otherwise every camera change re-runs the CPU residency cut (100+ ms), the GPU idles into a
# lower power state and the timings measure that instead of the frame.
hstr.set_properties({"worldCacheUpdates": 0, "cloudResidencyFrozen": os.environ.get("HSTR_FREEZE", "1") != "0"})
log(f"sea: settled in {frames} frames")
for motion, forward, yaw in MOTIONS:
    for test, properties in TESTS:
        hstr.set_properties(dict(BASE, **properties))
        for i in range(WARM):
            pose(i - WARM - TIMED, forward, yaw)
            m.renderFrame()
        t = timed(TIMED, -TIMED, forward, yaw)
        errors = []
        for step in range(STEPS):
            pose(step, forward, yaw)
            hstr.set_properties(BASE)
            m.renderFrame()
            hstr.set_properties({"storeExact": True})
            m.renderFrame()
            hstr.set_properties(dict(properties, compareReference=True, compareExact=True, compareBlock=1))
            m.renderFrame()
            p = hstr.properties
            hstr.set_properties({"compareReference": False, "compareExact": False})
            s = p.get("cloudStats", {})
            errors.append({"over02": float(p["referenceNoiseError"]), "p999": float(p["referenceLogP999"]),
                           "max": float(p["referenceLogMax"]), "marched": float(p["beamMarchedFraction"]),
                           "carriedPoints": int(s.get("beamCarriedPoints", 0)), "carriedPixels": int(s.get("beamCarriedPixels", 0)),
                           "marchTiles": int(s.get("beamMarchTiles", 0)), "debug": int(s.get("beamRefreshDebugCount", 0))})
        mean = {k: sum(e[k] for e in errors) / len(errors) for k in errors[0]}
        worst = max(e["over02"] for e in errors)
        with open(f"{OUT}/{TAG}_test.jsonl", "a") as f:
            f.write(json.dumps({"motion": motion, "test": test, "gpu": t, "errors": errors}) + "\n")
        parts = " ".join(f"{k} {v:.2f}" for k, v in t.items() if k not in ("HSTRCloud", "residency", "cpu") and v >= 0.3)
        parts += " " + json.dumps(t["residency"]) + " cpu " + json.dumps({k: v for k, v in t["cpu"].items() if v >= 0.2})
        log(f"{motion:7s} {test:16s} {t.get('HSTRCloud', 0):6.2f} ms ({parts}); marched {100 * mean['marched']:4.1f}%; "
            f"carried {mean['carriedPoints']:.0f} points {mean['carriedPixels']:.0f} pixels (debug {mean['debug']:.0f}); "
            f">0.02 mean {100 * mean['over02']:.3f}% worst {100 * worst:.3f}%, p99.9 {mean['p999']:.2e}, max {max(e['max'] for e in errors):.2e}")
exit()
