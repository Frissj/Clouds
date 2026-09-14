import json
import math
import os
import time
from falcor import *

# Cloud sea benchmark: GPU and CPU frame cost with the camera static, flying forward fast (tiles and bricks streaming), and with
# the sun sweeping; pictures of each. Results go to <tag>_test.json as one object per configuration.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "cloud_sea")
FLIGHT_SPEED = float(os.environ.get("HSTR_FLIGHT_SPEED", "20"))  # World units per frame (1200 per second at 60 fps).
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(int(os.environ.get("HSTR_W", "1920")), int(os.environ.get("HSTR_H", "1080")))
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
m.frameCapture.outputDir = OUT
results = []


def stats():
    s = hstr.properties.get("cloudStats", {})
    return {k: (float(v) if isinstance(v, float) else int(v)) for k, v in s.items()}


def capture(frames, step=None):
    m.profiler.enabled = True
    m.profiler.start_capture()
    wall = time.perf_counter()
    for i in range(frames):
        if step:
            step(i)
        m.renderFrame()
    wall = (time.perf_counter() - wall) / frames * 1000.0
    events = m.profiler.end_capture()["events"]
    m.profiler.enabled = False
    gpu = {}
    cpu = {}
    for name, lane in events.items():
        if "HSTRCloud" not in name:
            continue
        key = name.split("HSTRCloud", 1)[-1].strip("/").rsplit("/", 1)[0] or "HSTRCloud"
        if name.endswith("gpu_time"):
            gpu[key] = round(lane["stats"]["mean"], 3)
        elif name.endswith("cpu_time"):
            cpu[key] = round(lane["stats"]["mean"], 3)
    return {"wall_ms": round(wall, 2), "gpu_ms": gpu, "cpu_ms": cpu}


def record(label, frames, step=None):
    entry = {"label": label, **capture(frames, step), "cloud": stats()}
    results.append(entry)
    with open(f"{OUT}/{TAG}_test.json", "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(entry))


def picture(name):
    m.frameCapture.baseFilename = f"{TAG}_{name}"
    m.frameCapture.capture()


def settle(limit=600):
    # Tiles must all be resident; bricks stream until nothing is pending or the limit (the cut keeps refining within its budget).
    for i in range(limit):
        m.renderFrame()
        s = stats()
        if s.get("pendingTiles", 0) == 0 and s.get("pending", 0) == 0 and i > 16:
            return i
    return limit


# Startup: the first window of tiles and the bricks around the camera, then a converged world cache.
settled = settle()
while int(hstr.properties["worldCacheSampleCount"]) < 96:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0})
for i in range(8):
    m.renderFrame()
results.append({"label": "startup", "settle_frames": settled, "cloud": stats()})
record("static (cache converged)", 64)
picture("static")

# Static camera with the cache still converging (one 65k-photon update every frame).
hstr.set_properties({"worldCacheUpdates": 1})
record("static (cache updating)", 64)

# Fast forward flight over the sea: new tiles rasterize, their cache restarts, bricks stream.
direction = float3(0.0, 0.0, 1.0)


def fly(i):
    cam.position = cam.position + direction * FLIGHT_SPEED
    cam.target = cam.target + direction * FLIGHT_SPEED


record(f"forward flight {FLIGHT_SPEED:g}/frame", 300, fly)
picture("flight")
hstr.set_properties({"beamTemporal": False})
record(f"forward flight {FLIGHT_SPEED:g}/frame, beam per level", 120, fly)
picture("flight_levels")
hstr.set_properties({"cloudVirtual": False})
record(f"forward flight {FLIGHT_SPEED:g}/frame, proxy only", 120, fly)
picture("flight_proxy")
hstr.set_properties({"cloudVirtual": True, "beamTemporal": True})
settle(240)

# Sun sweep: the sun pages and the world cache restart with every change.
base = hstr.properties["sunDirection"]


def sweep(i):
    a = 0.6 + 0.01 * i
    hstr.set_properties({"sunDirection": float3(math.cos(a) * 0.6, 0.75, math.sin(a) * 0.6)})


record("sun sweep", 64, sweep)
picture("sunsweep")
hstr.set_properties({"sunDirection": base})
exit()
