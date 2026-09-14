import math
import os
import time
from falcor import *

# Every renderer against the path-traced reference: log error, the reference's own noise, per-component errors of the split
# model, and captures for side-by-side pictures.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "pt")
W, H = int(os.environ.get("HSTR_W", "1920")), int(os.environ.get("HSTR_H", "1080"))
SPP = int(os.environ.get("HSTR_SPP", "256"))
g = RenderGraph("PTCompare")
g.addPass(createPass("HSTRCloud", {"cutTransmittanceTolerance": 0.05}), "HSTRCloud")
g.addPass(createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0}), "ToneMapper")
g.addEdge("HSTRCloud.color", "ToneMapper.src")
g.markOutput("ToneMapper.dst")
m.addGraph(g)
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(W, H)
hstr = g.getPass("HSTRCloud")
cam = m.scene.camera
m.frameCapture.outputDir = OUT
target = float3(-10.0, 73.0, -43.0)
radius = 510.0
base = {"hstComponents": 15, "residualStrength": 1.0, "worldCacheCellVoxels": 2, "worldCacheBands": 2, "worldCacheOrder": 2,
        "worldCacheWindow": 1, "worldCacheEstimator": 1, "worldCacheTextured": 1, "worldCacheSegments": 0, "worldCachePhotons": 65536,
        "worldCacheBakeInterval": 1, "beamSegments": 1, "beamEdgeDepth": 0.0, "compareExact": False}
FULL = {"stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "lightingStride": 1}
CHEAP = {"stepOpticalDepth": 1.0, "minStepVoxels": 2.0, "lightingStride": 2}
default_sun = (0.4319, 0.8639, 0.2699)
back_sun = (-0.5, 0.6, 0.62)
n = math.sqrt(sum(c * c for c in back_sun))
back_sun = tuple(c / n for c in back_sun)
views = [
    ("side", (target.x + radius, 60.0, target.z), default_sun),
    ("sunbehind", (target.x + 0.8 * radius * math.cos(2.2), 180.0, target.z + 0.8 * radius * math.sin(2.2)), back_sun),
]
if os.environ.get("HSTR_VIEWS"):
    views = [v for v in views if v[0] in os.environ["HSTR_VIEWS"].split(",")]
configs = [
    ("split accurate", dict(FULL, debugView=8)),
    ("split cheap", dict(CHEAP, debugView=8)),
    ("beam h16x3 .05", dict(FULL, debugView=9, beamTileSize=16, beamLevels=3, beamTolerance=0.05)),
    ("old HST", dict(CHEAP, debugView=0)),
]
# Split-model components against the same reference components: background, sun single, smooth (sun multiple + sky).
components = [("background", 1), ("sun single", 2), ("smooth", 12)]


BLOCK = int(os.environ.get("HSTR_BLOCK", "8"))  # Block size of the noise-suppressed comparison.
SAVE_EVERY = 32  # Reference samples between saves, so a crash loses at most this many.


def measure(props, mask, block=1):
    hstr.set_properties(dict(props, hstComponents=mask, compareTarget=mask, compareSubstitute=0, compareBlock=block))
    m.renderFrame()
    hstr.set_properties({"compareReference": True})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareBlock": 1})
    return float(p["referenceLogError"]), float(p["referenceNoiseLogError"]), float(p["referenceError"]), float(p["referenceNoiseError"])


def accumulate_reference(name):
    """Path-traced reference of the current view, resumed from and saved to the results folder (per camera, sun and size)."""
    path = f"{OUT}/references/{name}_{W}x{H}"
    os.makedirs(f"{OUT}/references", exist_ok=True)
    hstr.set_properties({"debugView": 6, "referenceShow": 15})
    m.renderFrame()
    if os.path.exists(path + "_s0.exr"):
        hstr.set_properties({"loadReference": path})
        m.renderFrame()
    while int(hstr.properties["referenceSampleCount"]) < SPP:
        m.renderFrame()
        samples = int(hstr.properties["referenceSampleCount"])
        if samples % SAVE_EVERY == 0 or samples >= SPP:
            hstr.set_properties({"saveReference": path})
            m.renderFrame()
    return int(hstr.properties["referenceSampleCount"])


lines = []
current_sun = None
for name, position, sun in views:
    cam.position = float3(*position)
    cam.target = target
    if sun != current_sun:
        hstr.set_properties(dict(base, debugView=8, sunDirection=float3(*sun), worldCacheUpdates=4))
        while int(hstr.properties["worldCacheSampleCount"]) < 32:
            m.renderFrame()
        current_sun = sun
    hstr.set_properties(dict(base, worldCacheUpdates=0))
    start = time.time()
    samples = accumulate_reference(name)
    lines.append(f"{name}: {samples} spp reference ({time.time() - start:.0f} s this run)")
    m.frameCapture.baseFilename = f"{TAG}_{name}_reference"
    m.frameCapture.capture()
    for label, props in configs:
        log_error, noise, linear, linear_noise = measure(dict(base, **props), 15)
        block_error, block_noise, _, _ = measure(dict(base, **props), 15, BLOCK)
        lines.append(f"{name:10s} {label:16s} full frame: log err {log_error:.4f} (reference noise {noise:.4f})  {BLOCK}x{BLOCK} blocks: "
                     f"{block_error:.4f} (noise {block_noise:.4f})  linear {linear:.4f} (noise {linear_noise:.4f})")
        m.frameCapture.baseFilename = f"{TAG}_{name}_{label.replace(' ', '_')}"
        m.frameCapture.capture()
    for cname, mask in components:
        log_error, noise, linear, linear_noise = measure(dict(base, **configs[0][1]), mask)
        block_error, block_noise, _, _ = measure(dict(base, **configs[0][1]), mask, BLOCK)
        lines.append(f"{name:10s} split accurate, {cname:10s}: log err {log_error:.4f} (noise {noise:.4f})  {BLOCK}x{BLOCK} blocks: "
                     f"{block_error:.4f} (noise {block_noise:.4f})")
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
exit()
