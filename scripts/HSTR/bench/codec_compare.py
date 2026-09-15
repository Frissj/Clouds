import json
import os
import time
from falcor import *

# Cloud codec A/B: the same views rendered from several compilations of the same clouds. The first library is the baseline:
# its path-traced reference is accumulated (and saved), then every library's beam render is measured against that one
# reference, so the difference between libraries is the image cost of their compression alone.
#   HSTR_LIBRARIES   comma-separated library directories (or .hstrlib packages), baseline (near-lossless) first
#   HSTR_VIEWS       JSON list of [name, [px, py, pz], [tx, ty, tz]] with an optional up vector [ux, uy, uz]
#   HSTR_SPP, HSTR_W, HSTR_H, HSTR_TAG, HSTR_LIBRARY_COUNT (first n libraries only), HSTR_PROPS
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "codec")
W, H = int(os.environ.get("HSTR_W", "1280")), int(os.environ.get("HSTR_H", "720"))
SPP = int(os.environ.get("HSTR_SPP", "1024"))
BLOCK = 8
libraries = os.environ["HSTR_LIBRARIES"].split(",")[:int(os.environ.get("HSTR_LIBRARY_COUNT", "99"))]
views = json.loads(os.environ["HSTR_VIEWS"])
os.environ["HSTR_CLOUD_LIBRARY"] = libraries[0]
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(W, H)
hstr = m.activeGraph.getPass("HSTRCloud")
# HSTR_PROPS: JSON of HSTRCloud property overrides for the beam renders (e.g. beam tile settings).
beam_props = json.loads(os.environ.get("HSTR_PROPS", "{}"))
hstr.set_properties(beam_props)
cam = m.scene.camera
m.frameCapture.outputDir = OUT
os.makedirs(f"{OUT}/references", exist_ok=True)
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


def converge_beam():
    hstr.set_properties({"debugView": 9, "worldCacheUpdates": 1})
    while int(hstr.properties["worldCacheSampleCount"]) < 128:
        m.renderFrame()
    hstr.set_properties({"worldCacheUpdates": 0})
    for i in range(16):
        m.renderFrame()


def measure(block):
    hstr.set_properties({"debugView": 9, "compareTarget": 15, "compareSubstitute": 0, "compareBlock": block})
    m.renderFrame()
    hstr.set_properties({"compareReference": True})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareBlock": 1})
    return float(p["referenceLogError"]), float(p["referenceNoiseLogError"])


for name, position, target, *up in views:
    reference = f"{OUT}/references/{TAG}_{name}_{W}x{H}"
    for index, library in enumerate(libraries):
        label = os.path.basename(library) if os.path.isdir(library) else os.path.splitext(os.path.basename(library))[0]
        if hstr.properties["cloudLibrary"] != library:
            hstr.set_properties({"cloudLibrary": library})
        cam.position = float3(*position)
        cam.target = float3(*target)
        cam.up = float3(*(up[0] if up else [0.0, 1.0, 0.0]))
        frames = settle()
        s = stats()
        if index == 0:
            hstr.set_properties({"debugView": 6, "referenceShow": 15})
            start = time.time()
            m.renderFrame()
            while int(hstr.properties["referenceSampleCount"]) < SPP:
                m.renderFrame()
            hstr.set_properties({"saveReference": reference})
            m.renderFrame()
            shoot(f"{name}_reference_{label}")
            log(f"{name}: {SPP} spp reference on {label} in {time.time() - start:.0f} s")
        converge_beam()
        if index > 0:
            hstr.set_properties({"loadReference": reference})
            m.renderFrame()
        shoot(f"{name}_beam_{label}")
        full, noise = measure(1)
        block, block_noise = measure(BLOCK)
        log(f"{name} {label:24s} beam vs reference: log err {full:.4f} (noise {noise:.4f}), {BLOCK}x{BLOCK} blocks {block:.4f} "
            f"(noise {block_noise:.4f}); settled {frames} frames, {s.get('loaded', 0)} bricks, {s.get('residentMB', 0):.1f} MB")
exit()
