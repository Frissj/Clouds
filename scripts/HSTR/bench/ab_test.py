import json
import os
from falcor import *

# A/B of HSTRCloud properties at 4K on settled views: GPU ms of the base (A) and test (B) properties per configuration, and B's frame
# against A's (mean, share over 0.02 and 0.1, 99.9th percentile and max of the per-pixel log error). Defaults: the baked sun depth
# (A the live sun march, B the cache). HSTR_BASE must set every property HSTR_TEST sets: the views settle with the base, and nothing
# puts a property back afterwards.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "ab")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
BASE = json.loads(os.environ.get("HSTR_BASE", '{"cloudSunCache": false}'))
TEST = json.loads(os.environ.get("HSTR_TEST", '{"cloudSunCache": true}'))
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(3840, 2160)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
m.frameCapture.outputDir = OUT
views = [("near", (340, -10, 46), (6, -15, 46)), ("farside", (-330, -10, 46), (6, -15, 46)), ("sea", (0, 140, 0), (0, 40, 600))]
views = [v for v in views if v[0] in os.environ.get("HSTR_VIEWS", "near,farside,sea").split(",")]
configs = json.loads(os.environ.get("HSTR_CONFIGS", '[["pixel c15", {"debugView": 8, "hstComponents": 15}], ["pixel c3", {"debugView": 8, "hstComponents": 3}], ["beam c15", {"debugView": 9, "hstComponents": 15, "beamTemporal": false}]]'))
CAPTURE = os.environ.get("HSTR_CAPTURE", "1") != "0"
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


def gpu_times(frames=12):
    for i in range(16):
        m.renderFrame()
    best = None
    for attempt in range(3):
        m.profiler.enabled = True
        m.profiler.start_capture()
        for i in range(frames):
            m.renderFrame()
        capture = m.profiler.end_capture()
        m.profiler.enabled = False
        t = {name[:-len("/gpu_time")].split("HSTRCloud", 1)[-1].strip("/") or "HSTRCloud": lane["stats"]["mean"]
             for name, lane in capture["events"].items() if name.endswith("gpu_time") and "HSTRCloud" in name}
        if best is None or t.get("HSTRCloud", 1e9) < best.get("HSTRCloud", 1e9):
            best = t
    return best


def capture(name):
    if CAPTURE:
        m.frameCapture.baseFilename = name
        m.frameCapture.capture()


for name, position, target in views:
    cam.position = float3(*position)
    cam.target = float3(*target)
    hstr.set_properties(dict(BASE, debugView=9, hstComponents=15, worldCacheUpdates=1))
    frames = settle()
    while int(hstr.properties["worldCacheSampleCount"]) < 64:
        m.renderFrame()
    hstr.set_properties({"worldCacheUpdates": 0})
    s = stats()
    log(f"{name}: settled in {frames} frames; {s.get('sunBaked', 0)} sun bakes, {s.get('sunWaiting', 0)} waiting, {s.get('mapped', 0)} bricks mapped")
    for label, props in configs:
        hstr.set_properties(dict(props, **BASE))
        m.renderFrame()
        a = gpu_times()
        hstr.set_properties({"storeExact": True})
        m.renderFrame()
        capture(f"{TAG}_{name}_{label.replace(' ', '_')}_A")
        hstr.set_properties(TEST)
        b = gpu_times()
        hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
        m.renderFrame()
        p = hstr.properties
        hstr.set_properties({"compareReference": False, "compareExact": False})
        capture(f"{TAG}_{name}_{label.replace(' ', '_')}_B")
        parts = " ".join(f"{k} {v:.1f}" for k, v in b.items() if v >= 1.0 and k != "HSTRCloud")
        log(f"{name} {label:10s} A {a.get('HSTRCloud', 0):7.2f} ms, B {b.get('HSTRCloud', 0):7.2f} ms ({parts}); B vs A: "
            f"log {float(p['referenceLogError']):.2e}, >0.02 {100 * float(p['referenceNoiseError']):.3f}%, >0.1 {100 * float(p['referenceNoiseLogError']):.3f}%, "
            f"p99.9 {float(p['referenceLogP999']):.2e}, max {float(p['referenceLogMax']):.2e}")
exit()
