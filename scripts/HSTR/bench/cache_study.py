import math
import os
from falcor import *

# What limits the world cache against the path tracer: photon noise, cell size or SH order. Loads the saved 4K reference of each
# view (from pt_compare.py) and measures the split frame's smooth term (and its sun-multiple and sky parts) on 8x8 block means.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "cache_study")
W, H = int(os.environ.get("HSTR_W", "3840")), int(os.environ.get("HSTR_H", "2160"))
BLOCK = int(os.environ.get("HSTR_BLOCK", "8"))
g = RenderGraph("CacheStudy")
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
        "worldCacheBakeInterval": 1, "stepOpticalDepth": 0.5, "minStepVoxels": 1.0, "lightingStride": 1, "debugView": 8,
        "worldCacheModulation": 0.0}  # Every configuration starts from these, so no setting leaks into the next.
SUFFIX = ""  # Reference name suffix for non-default media.
if os.environ.get("HSTR_ANISOTROPY"):
    base["anisotropy"] = float(os.environ["HSTR_ANISOTROPY"])
    SUFFIX = f"_g{base['anisotropy']}"
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
# label, cache properties, photon batches of 65536 paths
configs = [
    ("current (2 vox, L2 win)", {}, 32),
    ("4x photons", {}, 128),
    ("16x photons", {}, 512),
    ("1-voxel cells", {"worldCacheCellVoxels": 1}, 512),
    ("4-voxel cells", {"worldCacheCellVoxels": 4}, 128),
    ("L2 no window", {"worldCacheWindow": 0}, 512),
    ("L1 window", {"worldCacheOrder": 1}, 512),
    # Cache stored relative to exp(-b tau_sun) of the voxel-resolution sun field.
    ("auto", {"worldCacheModulation": -1.0}, 32),
    ("auto 4x photons", {"worldCacheModulation": -1.0}, 128),
    ("auto 16x photons", {"worldCacheModulation": -1.0}, 512),
    ("auto no window 16x", {"worldCacheModulation": -1.0, "worldCacheWindow": 0}, 512),
    ("auto L1 16x", {"worldCacheModulation": -1.0, "worldCacheOrder": 1}, 512),
    ("mod 0.05", {"worldCacheModulation": 0.05}, 512),
    ("mod 0.07", {"worldCacheModulation": 0.07}, 512),
    ("mod 0.1", {"worldCacheModulation": 0.1}, 512),
    ("mod 0.15", {"worldCacheModulation": 0.15}, 512),
    ("mod 0.07 4-voxel", {"worldCacheModulation": 0.07, "worldCacheCellVoxels": 4}, 512),
    ("mod 0.07 1-voxel", {"worldCacheModulation": 0.07, "worldCacheCellVoxels": 1}, 512),
    ("mod 0.25", {"worldCacheModulation": 0.25}, 512),
    ("mod 0.5", {"worldCacheModulation": 0.5}, 512),
    ("mod 1.0", {"worldCacheModulation": 1.0}, 512),
    ("mod 0.25 4-voxel", {"worldCacheModulation": 0.25, "worldCacheCellVoxels": 4}, 512),
    ("mod 0.25 1-voxel", {"worldCacheModulation": 0.25, "worldCacheCellVoxels": 1}, 512),
    ("mod 0.25 no window", {"worldCacheModulation": 0.25, "worldCacheWindow": 0}, 512),
]
if os.environ.get("HSTR_CONFIGS"):
    keep = os.environ["HSTR_CONFIGS"].split(",")
    configs = [c for c in configs if c[0] in keep]


def measure(mask):
    hstr.set_properties({"hstComponents": mask, "compareTarget": mask, "compareSubstitute": 0, "compareBlock": BLOCK})
    m.renderFrame()
    hstr.set_properties({"compareReference": True})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareBlock": 1, "hstComponents": 15})
    return float(p["referenceLogError"]), float(p["referenceNoiseLogError"])


lines = []
for name, position, sun in views:
    path = f"{OUT}/references/{name}{SUFFIX}_{W}x{H}"
    if not os.path.exists(path + "_s0.exr"):
        lines.append(f"{name}: no saved reference at {path}, run pt_compare.py first")
        continue
    cam.position = float3(*position)
    cam.target = target
    hstr.set_properties(dict(base, sunDirection=float3(*sun), worldCacheUpdates=0))
    m.renderFrame()
    hstr.set_properties({"loadReference": path})
    m.renderFrame()
    lines.append(f"{name}: reference {hstr.properties['referenceSampleCount']} spp")
    for label, props, batches in configs:
        # A cell-size change restarts the cache, so every configuration accumulates its batches from zero.
        settings = dict(base, **props)
        hstr.set_properties(dict(settings, worldCacheUpdates=0, worldCacheCellVoxels=settings["worldCacheCellVoxels"] + 1))
        m.renderFrame()
        hstr.set_properties(dict(settings, worldCacheUpdates=8))
        while int(hstr.properties["worldCacheSampleCount"]) < batches:
            m.renderFrame()
        hstr.set_properties({"worldCacheUpdates": 0})
        m.renderFrame()
        # The camera reads the cache's sun and sky fields baked together, so only their sum (the smooth term) is measurable.
        smooth = measure(12)
        full = measure(15)
        lines.append(f"{name:10s} {label:24s} {BLOCK}x{BLOCK}-block log err: full {full[0]:.4f}  smooth {smooth[0]:.4f}  "
                     f"(noise {full[1]:.4f})")
        m.frameCapture.baseFilename = f"{TAG}_{name}_{label.split(' (')[0].replace(' ', '_')}"
        m.frameCapture.capture()
        with open(f"{OUT}/{TAG}_test.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
exit()
