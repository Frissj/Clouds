import os
import sys
from falcor import *

# Does the RIGHT EDGE of the frame carry error against per-pixel truth that the rest of the frame does not?
#
# This exists because a whole-frame aggregate cannot answer it. b7f82f00 removed an exact march of the final screen tile from
# resolveBeamPixel after showing it bought no whole-frame quality at 4K and cost 0.4 ms, and that the screen and reference layouts
# score the same against truth. But "the frame is fine" does not prove "the last four columns are fine" - a band that is 0.3% of
# the pixels can be bad while the aggregate moves by less than its own run-to-run noise.
#
# So this reports the SAME statistics twice per arm: over the whole frame, and over the last beamTileSize columns alone, using
# compareColumnLow/High. The truth is sea_config.REFERENCE, the world-cache march at one-voxel steps - never the beam renderer.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sea_config import REFERENCE

OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "edge_probe")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
m.script("scripts/HSTR/CloudSea.py")
WIDTH, HEIGHT = 1280, 720
m.resizeFrameBuffer(WIDTH, HEIGHT)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
START, TARGET = float3(340, -10, 46), float3(6, -15, 46)
TILE = 4
lines = []


def log(line):
    lines.append(line)
    print(line)
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


BASE = dict(
    debugView=9, hstComponents=15, beamTemporal=False, cloudSunCache=True, beamTileSize=TILE, beamLevels=1, beamSegments=1,
    beamTolerance=0.05, beamEdgeContrast=0.5, beamShip=True, beamShipMask=509, beamSparse=False, beamSparseCut=False,
    beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05, beamRefMargin=0.15,
)


def aim(yaw):
    import math
    d = TARGET - START
    c, s = math.cos(yaw), math.sin(yaw)
    cam.position = START
    cam.target = START + float3(c * d.x - s * d.z, d.y, s * d.x + c * d.z)


aim(0.0)
hstr.set_properties(dict(BASE, beamRefresh=0, beamRefFrame=False, worldCacheUpdates=1))
for i in range(int(os.environ.get("HSTR_SETTLE", "300"))):
    m.renderFrame()
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0})


def window(label, low, high):
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1,
                         "compareColumnLow": low, "compareColumnHigh": high})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False, "compareExact": False, "compareColumnLow": 0, "compareColumnHigh": 0})
    return (f"{label:14s} log {float(p['referenceLogError']):.3e} >0.02 {100 * float(p['referenceNoiseError']):7.3f}% "
            f">0.1 {100 * float(p['referenceNoiseLogError']):6.3f}% p99.9 {float(p['referenceLogP999']):.2e} "
            f"max {float(p['referenceLogMax']):.2e}")


def measure(label, props, yaw):
    aim(yaw)
    hstr.set_properties(dict(REFERENCE))  # debugView 8, one-voxel steps: the only truth this script accepts.
    m.renderFrame()
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    hstr.set_properties(dict(BASE, **props))
    for i in range(int(os.environ.get("HSTR_BUILDS", "6"))):
        m.renderFrame()
    log(f"{label:22s} yaw {yaw:5.3f} | whole  {window('', 0, 0)}")
    log(f"{label:22s} yaw {yaw:5.3f} | right  {window('', WIDTH - TILE, WIDTH)}")
    log(f"{label:22s} yaw {yaw:5.3f} | left   {window('', 0, TILE)}")


for yaw in (0.0, 0.05):
    measure("screen, no reuse", dict(beamRefresh=0, beamRefFrame=False), yaw)
    measure("reference, no reuse", dict(beamRefresh=0, beamRefFrame=True), yaw)
    measure("reference, refresh 4", dict(beamRefresh=4, beamRefFrame=True), yaw)
exit()
