import os
import sys
from falcor import *

# The octahedral image's residual march costs 2.7x to 5x the rectangle's. Is that because MORE UNITS are dispatched, or because
# more of the units dispatched fail their carry?
#
# The two have completely different fixes. If the dispatch is too broad, the fault is beamScreenBounds: the screen's footprint in
# octahedral space is curved, and it is bounded by sampling a grid of screen directions and padding by the largest gap between
# neighbours, which near the map's wrap is enormous. That is a bounding problem and it is cheap to fix. If instead the dispatch is
# tight and the units inside it are failing, the fault is the basis fitting a sheared cell - the map's singular values are 2.36 and
# 1.20 at the diamond mid-edge - and no bound will help.
#
# beamGridThreads counts the threads the query dispatch launched, cumulative, so it is differenced per frame. marchTiles counts the
# tiles that failed and so the units the residual had to march.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sea_config import mk

OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "oct_units")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
FRAMES = int(os.environ.get("HSTR_FRAMES", "32"))
WARM = int(os.environ.get("HSTR_WARM", "16"))

m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(*[int(v) for v in os.environ.get("HSTR_RES", "1280x720").split("x")])
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
START_POSITION, START_TARGET = float3(0, 140, 0), float3(0, 40, 600)
lines = []


def log(line):
    lines.append(line)
    print(line)
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def stats():
    return hstr.properties.get("cloudStats", {})


def pose(frame, forward, yaw):
    import math
    angle = yaw * frame
    view = START_TARGET - START_POSITION
    turned = float3(view.x * math.cos(angle) - view.z * math.sin(angle), view.y,
                    view.x * math.sin(angle) + view.z * math.cos(angle))
    cam.position = START_POSITION + float3(0, 0, forward * frame)
    cam.target = cam.position + turned


COMMON = dict(beamTileSize=4, beamLevels=1, beamSegments=1, beamTemporal=False, beamSparse=False, beamSparseCut=False,
              beamQueue=False, beamGridDispatch=False, beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05,
              beamRefreshDebug=0, beamRefFrame=True, beamRefresh=256, beamTolerance=0.05, beamGuard=True)

pose(0, 0.0, 0.0)
hstr.set_properties(dict(mk(**COMMON, beamOct=False, beamRefMargin=0.15), worldCacheUpdates=1))
for i in range(int(os.environ.get("HSTR_SETTLE", "900"))):
    m.renderFrame()
    s = stats()
    if i > 64 and int(s.get("pending", 0)) == 0 and int(s.get("sunBakesFrame", 0)) == 0:
        break
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0, "cloudResidencyFrozen": True})
# The beam counters only leave the GPU while a reference comparison is active, so one stored frame is captured to open that path.
# Its error numbers are meaningless here and nothing below reads them.
m.renderFrame()
hstr.set_properties({"storeExact": True})
m.renderFrame()
hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 0})
log(f"sea: settled, mapped {stats().get('mapped', 0)}")

ARMS = [("rect", mk(**COMMON, beamOct=False, beamRefMargin=0.15)),
        ("oct", mk(**COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0))]
MOTIONS = [("park", 0.0, 0.0), ("look", 0.0, 0.01), ("flick", 0.0, 0.05), ("walk", 2.0, 0.01)]

for label, props in ARMS:
    for motion, forward, yaw in MOTIONS:
        hstr.set_properties(dict(props))
        hstr.set_properties({"beamReset": True})
        for i in range(WARM):
            pose(i - WARM, forward, yaw)
            m.renderFrame()
        before = stats()
        priorThreads = int(before.get("beamGridThreads", 0))
        priorUnits = int(before.get("beamUnitThreads", 0))
        threads, units, dirty, unver, cls, sweeps = [], [], [], [], [], 0
        priorSweeps = int(before.get("beamGridSweeps", 0))
        for i in range(FRAMES):
            pose(i, forward, yaw)
            m.renderFrame()
            s = stats()
            now = int(s.get("beamGridThreads", 0))
            threads.append(now - priorThreads)
            priorThreads = now
            nowUnits = int(s.get("beamUnitThreads", 0))
            units.append(nowUnits - priorUnits)
            priorUnits = nowUnits
            dirty.append(int(s.get("beamDirtyBlocks", 0)))
            unver.append(int(s.get("beamDirtyUnverified", 0)))
            cls.append(int(s.get("beamClassifyCells", 0)))
        sweeps = int(stats().get("beamGridSweeps", 0)) - priorSweeps
        mean = lambda v: sum(v) / float(len(v))
        dim = int(hstr.properties.get("beamOctDim", 0))
        log(f"{label:5s} {motion:6s} query {mean(threads):10.0f}  residual {mean(units):10.0f}  dirty blocks {mean(dirty):8.0f}"
            f" unverified {mean(unver):8.0f}  classified {mean(cls):8.0f}  sweeps {sweeps:3d}/{FRAMES}")
exit()
