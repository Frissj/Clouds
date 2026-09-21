import os
import sys
from falcor import *

# Does an octahedral beam image reconstruct the same picture as the anchored rectangle?
#
# beamOct makes the beam image the sphere instead of a perspective rectangle anchored to a camera pose. A direction's texel is then
# fixed for the life of the renderer, so there is nothing to re-anchor - which is the point, because a re-anchor discards a cache
# that is entirely correct and costs 4.16 ms of a 6.84 ms fast turn. The lattice, tiles, level map, residual, carry and generated
# dispatch are untouched; only the three functions that map between the image and the world changed.
#
# So the question here is quality, not speed. Against sea_config.REFERENCE - the world-cache march at one-voxel steps, never the
# beam renderer - the octahedral image must score what the rectangle scores. Two known hazards to watch for:
#   - the screen's footprint in octahedral space is curved, so beamScreenBounds is sampled and padded; beamOctFull builds the whole
#     sphere instead, which isolates a bounding bug from a mapping bug. If full is clean and bounded is not, the bound is wrong.
#   - the map wraps at the square's border (the half great circles x = 0 and y = 0 below the horizon), where a tile's 2 x 2
#     neighbourhood is not contiguous. The frame puts +z at the zenith to keep that seam under the horizon, and the yaws below look
#     level and upward; a downward arm would need the wrap-aware apron that is not built yet.
#
# Run at 720p/1080p, not 4K: the sphere at screen angular resolution is 13.7x the screen, so the residual does not fit at 4K until
# it is paged. That is the next step, and it is an allocation question rather than a correctness one.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sea_config import REFERENCE

OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "oct_probe")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
m.script("scripts/HSTR/CloudSea.py")
WIDTH, HEIGHT = [int(v) for v in os.environ.get("HSTR_RES", "1280x720").split("x")]
m.resizeFrameBuffer(WIDTH, HEIGHT)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
START, TARGET = float3(340, -10, 46), float3(6, -15, 46)
lines = []


def log(line):
    lines.append(line)
    print(line)
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


BASE = dict(
    debugView=9, hstComponents=15, beamTemporal=False, cloudSunCache=True, beamTileSize=4, beamLevels=1, beamSegments=1,
    beamTolerance=0.05, beamEdgeContrast=0.5, beamShip=True, beamShipMask=509, beamSparse=False, beamSparseCut=False,
    beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05, beamRefFrame=True,
)


def aim(yaw, pitch=0.0):
    import math
    d = TARGET - START
    c, s = math.cos(yaw), math.sin(yaw)
    turned = float3(c * d.x - s * d.z, d.y, s * d.x + c * d.z)
    reach = math.sqrt(turned.x * turned.x + turned.y * turned.y + turned.z * turned.z)
    cam.position = START
    cam.target = START + turned + float3(0, math.tan(pitch) * reach, 0)


aim(0.0)
hstr.set_properties(dict(BASE, beamRefresh=0, beamRefMargin=0.15, beamOct=False, worldCacheUpdates=1))
for i in range(int(os.environ.get("HSTR_SETTLE", "300"))):
    m.renderFrame()
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0})


def measure(label, props, yaw, pitch=0.0):
    aim(yaw, pitch)
    hstr.set_properties(dict(REFERENCE))  # debugView 8, one-voxel steps: the only truth this script accepts.
    m.renderFrame()
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    hstr.set_properties(dict(BASE, **props))
    for i in range(int(os.environ.get("HSTR_BUILDS", "8"))):
        m.renderFrame()
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    p = hstr.properties
    dim = int(p.get("beamOctDim", 0))
    line = (f"{label:18s} yaw {yaw:5.2f} pitch {pitch:5.2f} | >0.02 {100 * float(p['referenceNoiseError']):7.3f}% "
            f">0.1 {100 * float(p['referenceNoiseLogError']):6.3f}% p99.9 {float(p['referenceLogP999']):.2e} "
            f"max {float(p['referenceLogMax']):.2e} | ms {float(p.get('cloudGpuMs', 0.0)):5.2f} oct {dim}")
    hstr.set_properties({"compareReference": False, "compareExact": False})
    log(line)


for yaw, pitch in ((0.0, 0.0), (0.6, 0.0), (0.0, 0.5), (2.5, 0.0)):
    measure("rect m15", dict(beamRefresh=16, beamRefMargin=0.15, beamOct=False), yaw, pitch)
    measure("oct full", dict(beamRefresh=16, beamOct=True, beamOctFull=True), yaw, pitch)
    measure("oct bounded", dict(beamRefresh=16, beamOct=True, beamOctFull=False), yaw, pitch)
exit()
