import json
import os
from falcor import *

# Fast check that the rotation-invariant beam basis frame renders the same image as the screen-space build, parked and after a turn.
# It answers "is the reference frame correct and does a turn stop rebuilding it", not "is it fast" - use run_sea.py motion for that.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "ref_smoke")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(1280, 720)
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
    beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05, beamRefMargin=0.15,
)
# Diagnostics. HSTR_MARGIN 0 makes the beam image the screen image exactly - same dims, zero offset, one beam unit per pixel - so
# anything that still differs from the screen build is in the residual/resolve plumbing rather than in the frame mapping.
# HSTR_TOL huge accepts every tile, leaving the basis alone with no residual at all.
BASE["beamRefMargin"] = float(os.environ.get("HSTR_MARGIN", "0.15"))
BASE["beamTolerance"] = float(os.environ.get("HSTR_TOL", "0.05"))
BASE["beamEdgeContrast"] = float(os.environ.get("HSTR_EDGE", "0.5"))


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


def measure(label, props, yaw):
    # The exact per-pixel frame of THIS camera is the reference, so a turn is compared against its own truth.
    aim(yaw)
    hstr.set_properties(dict(BASE, beamRefresh=0, beamRefFrame=False))
    m.renderFrame()
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    hstr.set_properties(dict(BASE, **props))
    for i in range(int(os.environ.get("HSTR_BUILDS", "6"))):  # Let the carry reach steady state.
        m.renderFrame()
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    p = hstr.properties
    s = p.get("cloudStats", {})
    hstr.set_properties({"compareReference": False, "compareExact": False})
    log(f"{label:24s} yaw {yaw:5.3f}: log {float(p['referenceLogError']):.3e} >0.02 {100 * float(p['referenceNoiseError']):.3f}% "
        f">0.1 {100 * float(p['referenceNoiseLogError']):.3f}% p99.9 {float(p['referenceLogP999']):.2e} "
        f"max {float(p['referenceLogMax']):.2e}; carried {int(s.get('beamCarriedPoints', 0))} anchors {int(s.get('beamRefAnchors', 0))}")


for yaw in (0.0, 0.05):
    measure("screen, no reuse", dict(beamRefresh=0, beamRefFrame=False), yaw)
    measure("screen, refresh 4", dict(beamRefresh=4, beamRefFrame=False), yaw)
    measure("reference, no reuse", dict(beamRefresh=0, beamRefFrame=True), yaw)
    measure("reference, refresh 4", dict(beamRefresh=4, beamRefFrame=True), yaw)
exit()
