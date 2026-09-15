import json
import os
import time
from falcor import *

# Close-ups of one sea cloud: the beam renderer (debugView 9) against the path-traced reference (debugView 6) on the same
# resident fine density, with log error. The cloud was picked from closeup_scout.py (top-down at y 600).
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "closeup")
W, H = int(os.environ.get("HSTR_W", "1280")), int(os.environ.get("HSTR_H", "720"))
SPP = int(os.environ.get("HSTR_SPP", "256"))
BLOCK = 8
target = float3(20.0, -45.0, 65.0)
views = [
    ("near", float3(250.0, -30.0, 65.0)),
    ("detail", float3(135.0, -25.0, 35.0)),
]
if os.environ.get("HSTR_VIEWS"):
    views = [v for v in views if v[0] in os.environ["HSTR_VIEWS"].split(",")]
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(W, H)
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
    return {k: (float(v) if isinstance(v, float) else int(v)) for k, v in hstr.properties.get("cloudStats", {}).items()}


def settle(limit=900):
    for i in range(limit):
        m.renderFrame()
        s = stats()
        if i > 32 and s.get("pendingTiles", 0) == 0 and s.get("pending", 0) == 0:
            return i
    return limit


def shoot(name):
    m.frameCapture.baseFilename = f"{TAG}_{name}"
    m.frameCapture.capture()


def measure(block):
    hstr.set_properties({"debugView": 9, "compareTarget": 15, "compareSubstitute": 0, "compareBlock": block})
    m.renderFrame()
    hstr.set_properties({"compareReference": True})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareBlock": 1})
    return float(p["referenceLogError"]), float(p["referenceNoiseLogError"])


for name, position in views:
    cam.position = position
    cam.target = target
    cam.up = float3(0.0, 1.0, 0.0)
    frames = settle()
    hstr.set_properties({"debugView": 9, "worldCacheUpdates": 1})
    while int(hstr.properties["worldCacheSampleCount"]) < 128:
        m.renderFrame()
    hstr.set_properties({"worldCacheUpdates": 0})
    for i in range(16):
        m.renderFrame()
    log(f"{name}: settled in {frames} frames, {json.dumps(stats())}")
    shoot(f"{name}_beam")
    hstr.set_properties({"debugView": 6, "referenceShow": 15})
    start = time.time()
    m.renderFrame()
    while int(hstr.properties["referenceSampleCount"]) < SPP:
        m.renderFrame()
        n = int(hstr.properties["referenceSampleCount"])
        if n & (n - 1) == 0:
            log(f"{name}: reference {n} spp after {time.time() - start:.0f} s")
    shoot(f"{name}_reference")
    full, full_noise = measure(1)
    block, block_noise = measure(BLOCK)
    log(f"{name}: beam vs {SPP} spp reference: log err {full:.4f} (reference noise {full_noise:.4f}), "
        f"{BLOCK}x{BLOCK} blocks {block:.4f} (noise {block_noise:.4f}), reference {time.time() - start:.0f} s")
exit()
