import os
import sys
from falcor import *

# The octahedral beam image scores 30% of pixels over 0.02 where the rectangle scores 0.5%. Is the DIRECTION MAPPING wrong, or the
# TILE RECONSTRUCTION over it?
#
# Two arms separate them, because the renderer has two independent paths from a beam position to a colour:
#   - beamTolerance 0 refines every tile, so every pixel comes from an exact per-pixel march of its own direction and the basis is
#     never consulted. If that is clean, octDecode/octEncode and the marching are right and the fault is in reconstruction.
#   - beamTolerance 10 accepts every tile, so every pixel comes from the bilinear-plus-bubble basis fitted over the tile and the
#     residual never runs. If that is where the error lives, the fault is the basis over a map whose derivative kinks.
# The rectangle runs both as its control, so each arm is read against itself rather than against the default configuration.
#
# The suspicion being tested: the octahedral diamond edge is the great circle perpendicular to the map's z axis, and the frame puts
# +z at the zenith, which makes that edge the HORIZON - where the map is continuous but its derivative is not, and where a cloud
# camera spends all of its time. The first probe already hints at it: looking up 28 degrees scored 4.6% against 30-37% level.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sea_config import REFERENCE

OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "oct_scale")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
m.script("scripts/HSTR/CloudSea.py")
m.frameCapture.outputDir = OUT
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
    beamEdgeContrast=0.5, beamShip=True, beamShipMask=509, beamSparse=False, beamSparseCut=False,
    beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05, beamRefFrame=True, beamRefresh=16,
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
hstr.set_properties(dict(BASE, beamTolerance=0.05, beamRefMargin=0.15, beamOct=False, worldCacheUpdates=1))
for i in range(int(os.environ.get("HSTR_SETTLE", "300"))):
    m.renderFrame()
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0})


def measure(label, props, yaw, pitch=0.0):
    aim(yaw, pitch)
    hstr.set_properties(dict(REFERENCE))
    m.renderFrame()
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    hstr.set_properties(dict(BASE, **props))
    for i in range(int(os.environ.get("HSTR_BUILDS", "8"))):
        m.renderFrame()
    m.frameCapture.baseFilename = f"{TAG}_{label.replace(' ', '_')}_p{int(pitch * 100)}"
    m.frameCapture.capture()
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    p = hstr.properties
    stats = p.get("cloudStats", {})
    log(f"{label:22s} pitch {pitch:5.2f} | >0.02 {100 * float(p['referenceNoiseError']):7.3f}% "
        f">0.1 {100 * float(p['referenceNoiseLogError']):6.3f}% p99.9 {float(p['referenceLogP999']):.2e} "
        f"max {float(p['referenceLogMax']):.2e} | marchTiles {int(stats.get('beamMarchTiles', 0)):8d} oct {int(p.get('beamOctDim', 0))}")
    hstr.set_properties({"compareReference": False, "compareExact": False})


# Is the octahedral image's extra error RESOLUTION, or something structural? A screen pixel never lands on an octahedral texel, so
# every pixel resamples four marched neighbours, and that resampling alone scored 0.511% where the aligned rectangle scored 0.000%.
# If it is sampling, beamOctScale buys it back at the square of the memory; if it plateaus, it is the map's derivative kink at the
# diamond edge - which is the horizon in this frame - and no amount of resolution will fix it.
# Is the octahedral deficit the HORIZON? The map's derivative kinks on the great circle perpendicular to its z axis, and the frame
# puts +z at the zenith, so that circle is the horizon - where a tile straddling it has a ray field the bilinear-plus-bubble basis
# cannot fit. Pitching up walks the horizon out of frame: vfov is about 46 degrees, so pitch 1.0 rad (57 degrees) clears it
# entirely. If the octahedral image reaches the rectangle there and only there, the kink is the cause and the fix is to orient the
# map rather than to spend resolution on it.
# Can MARCHING buy what RESOLUTION was buying? The octahedral deficit is the basis fitting a sheared texel (singular values 2.36
# and 1.20 at 55 degrees, so a texel is about 2:1) plus the resolve resampling four marched units. beamScreenResidual removes the
# resampling outright by marching each failed pixel's own ray; tightening beamTolerance then fails more tiles, moving pixels off the
# basis and onto that exact path. If x1.0 with a tight tolerance reaches what x2.0 reaches, the sphere needs a quarter of the memory
# and pays marching time instead - which is the trade worth knowing before a page pool is sized.
for yaw in (0.0, 0.8):
    measure("rect default", dict(beamTolerance=0.05, beamRefMargin=0.15, beamOct=False), yaw, 0.0)
    measure("oct x1.0", dict(beamTolerance=0.05, beamOct=True, beamOctFull=True, beamOctScale=1.0), yaw, 0.0)
    measure("oct x2.0", dict(beamTolerance=0.05, beamOct=True, beamOctFull=True, beamOctScale=2.0), yaw, 0.0)
    for tol in (0.05, 0.02, 0.005):
        measure(f"oct x1.0 scr t{tol}",
                dict(beamTolerance=tol, beamOct=True, beamOctFull=True, beamOctScale=1.0, beamScreenResidual=True), yaw, 0.0)
exit()
