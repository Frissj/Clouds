import json
import os
import time
from falcor import *

# Cloud sea configurations against the 4K path tracer (debugView 6), which is the ground truth; the per-pixel march at one-voxel
# steps that ab_test.py compares against is only a model of it and shares the beam view's biases. The reference is saved to and
# resumed from the results folder. Run through run_sea.py truth TAG sweep --views farside.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "truth")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
BASE = json.loads(os.environ["HSTR_BASE"])
TESTS = json.loads(os.environ["HSTR_TESTS"])
SPP = int(os.environ.get("HSTR_SPP", "256"))
BLOCK = int(os.environ.get("HSTR_BLOCK", "8"))
SAVE_EVERY = 16
m.script("scripts/HSTR/CloudSea.py")
W, H = 3840, 2160
m.resizeFrameBuffer(W, H)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
views = [("near", (340, -10, 46), (6, -15, 46)), ("farside", (-330, -10, 46), (6, -15, 46)), ("sea", (0, 140, 0), (0, 40, 600))]
views = [v for v in views if v[0] in os.environ.get("HSTR_VIEWS", "farside").split(",")]
lines = []


def log(line):
    lines.append(line)
    print(line)
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def settle():
    quiet = 0
    for i in range(1500):
        m.renderFrame()
        s = hstr.properties.get("cloudStats", {})
        busy = int(s.get("pendingTiles", 0)) + int(s.get("pending", 0)) + int(s.get("sunBakesFrame", 0))
        quiet = quiet + 1 if busy == 0 else 0
        if i > 32 and quiet > 20:
            return i
    return 1500


def reference(name):
    path = f"{OUT}/references/sea_{name}_{W}x{H}"
    os.makedirs(f"{OUT}/references", exist_ok=True)
    # One 4K sample of the sea at once runs past the driver's timeout: trace it in separately submitted bands.
    hstr.set_properties({"debugView": 6, "referenceShow": 15, "referenceBandRows": int(os.environ.get("HSTR_BAND_ROWS", "135"))})
    m.renderFrame()
    if os.path.exists(path + "_s0.exr"):
        hstr.set_properties({"loadReference": path})
        m.renderFrame()
    start = time.time()
    while int(hstr.properties["referenceSampleCount"]) < SPP:
        m.renderFrame()
        samples = int(hstr.properties["referenceSampleCount"])
        if samples % SAVE_EVERY == 0 or samples >= SPP:
            hstr.set_properties({"saveReference": path})
            m.renderFrame()
    return int(hstr.properties["referenceSampleCount"]), time.time() - start


def measure(props, block):
    # compareExact selects the stored exact-march frame, so it must stay false here: the loaded path trace is the authority. The
    # comparison pass still fills its per-pixel histogram for the percentile quality gate.
    hstr.set_properties(dict(props, compareTarget=15, compareSubstitute=0, compareBlock=block, compareExact=False))
    for i in range(8):
        m.renderFrame()
    hstr.set_properties({"compareReference": True})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareBlock": 1})
    return (float(p["referenceLogError"]), float(p["referenceNoiseLogError"]), float(p["referenceLogP999"]),
            float(p["referenceLogMax"]))


for name, position, target in views:
    cam.position = float3(*position)
    cam.target = float3(*target)
    hstr.set_properties(dict(BASE, debugView=9, hstComponents=15, worldCacheUpdates=1))
    frames = settle()
    while int(hstr.properties["worldCacheSampleCount"]) < 64:
        m.renderFrame()
    hstr.set_properties({"worldCacheUpdates": 0, "cloudResidencyFrozen": True})
    samples, seconds = reference(name)
    log(f"{name}: settled in {frames} frames; path-traced reference {samples} spp ({seconds:.0f} s this run)")
    for test, properties in TESTS:
        full, full_noise, p999, maximum = measure(dict(BASE, **properties), 1)
        block, block_noise, _, _ = measure(dict(BASE, **properties), BLOCK)
        with open(f"{OUT}/{TAG}_test.jsonl", "a") as f:
            f.write(json.dumps({"view": name, "test": test, "log": full, "noise": full_noise, "block": block,
                                "blockNoise": block_noise, "p999": p999, "max": maximum, "spp": samples}) + "\n")
        log(f"{name:8s} {test:16s} log err {full:.4f} (reference noise {full_noise:.4f})   {BLOCK}x{BLOCK} blocks {block:.4f} "
            f"(noise {block_noise:.4f})   p99.9 {p999:.4f} max {maximum:.4f}")
    hstr.set_properties({"cloudResidencyFrozen": False})
exit()
