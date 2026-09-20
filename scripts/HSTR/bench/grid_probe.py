import os
import sys
from falcor import *

# Which way did the query dispatch go, and how many threads did it launch? beamGridSweeps / beamGridGenerated / beamGridStrips
# count the frames that swept the region, generated from the refresh phase alone, or added the strips a rotation exposed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sea_config import mk

OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "grid_probe")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(3840, 2160)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
START, TARGET = float3(0, 140, 0), float3(0, 40, 600)
lines = []


def log(line):
    lines.append(line)
    print(line)
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


BASE = mk(beamTileSize=4, beamLevels=1, beamSegments=1, beamTemporal=False, beamSparse=False, beamSparseCut=False,
          beamQueue=False, beamGridDispatch=False, beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05,
          beamRefreshDebug=0, beamRefFrame=True, beamRefMargin=0.15, beamRefresh=256)


def pose(frame, yaw):
    import math
    a = yaw * frame
    d = TARGET - START
    cam.position = START
    cam.target = START + float3(d.x * math.cos(a) - d.z * math.sin(a), d.y, d.x * math.sin(a) + d.z * math.cos(a))


pose(0, 0.0)
hstr.set_properties(dict(BASE, worldCacheUpdates=1))
for i in range(900):
    m.renderFrame()
    s = hstr.properties.get("cloudStats", {})
    if i > 64 and int(s.get("pending", 0)) == 0:
        break
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0})

for label, yaw, margin, prebuild in (("parked", 0.0, 0.15, False), ("yaw", 0.004, 0.15, False),
                                     ("parked m50", 0.0, 0.5, True), ("yaw m50", 0.004, 0.5, True)):
    hstr.set_properties(dict(BASE, beamRefMargin=margin, beamRefPrebuild=prebuild))
    for i in range(24):  # Settle the frame and its history before counting.
        pose(i, yaw)
        m.renderFrame()
    before = hstr.properties.get("cloudStats", {})
    b = {k: int(before.get(k, 0)) for k in ("beamGridSweeps", "beamGridGenerated", "beamGridStrips", "beamGridThreads", "beamRefAnchors")}
    for i in range(24, 72):
        pose(i, yaw)
        m.renderFrame()
    a = hstr.properties.get("cloudStats", {})
    d = {k: int(a.get(k, 0)) - b[k] for k in b}
    log(f"{label:12s} over 48 frames: sweeps {d['beamGridSweeps']:3d} generated {d['beamGridGenerated']:3d} "
        f"strips {d['beamGridStrips']:3d} anchors {d['beamRefAnchors']:2d} threads/frame {d['beamGridThreads'] // 48}")
exit()
