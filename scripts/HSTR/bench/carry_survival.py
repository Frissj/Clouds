import json
import os
import sys
from falcor import *

# How long does a cached WORLD DIRECTION survive while the camera is moving?
#
# This exists to decide whether a 360-degree directional cache is worth building. Re-anchoring is purely orientational - the test in
# updateBeamReferenceFrame projects the four screen corners and nothing else, so pure translation NEVER re-anchors. Everything a
# flying camera loses, it loses to the parallax test, and that test is a function of displacement over distance: it does not care
# which way a sample faces. So translation invalidates the whole angular domain at once, including the part behind the camera.
#
# That is the entire question for the cylindrical band. A band covering 360 degrees is 5-9x the screen in area (4.7x matching the
# screen's centre angular density, 9.4x matching its edge density). If a direction survives forty frames of flight, that area is a
# cache and the band pays for itself the first time the camera turns. If it survives two, the area is memory being re-marched
# before anything reads it, and the answer is a visible-sector cache with a depth bound instead.
#
# So: carried points and carried residual units per frame, against the parked run as the ceiling. No timings - reading properties
# every frame forces a CPU-GPU sync and inflates them (see live_invalidation.py). Counts are exact and unaffected.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sea_config import mk

OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "carry_survival")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
FRAMES = int(os.environ.get("HSTR_FRAMES", "48"))
WARM = int(os.environ.get("HSTR_WARM", "24"))

m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(*[int(v) for v in os.environ.get("HSTR_RES", "3840x2160").split("x")])
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


BASE = mk(beamTileSize=4, beamLevels=1, beamSegments=1, beamTemporal=False, beamSparse=False, beamSparseCut=False,
          beamQueue=False, beamGridDispatch=False, beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05,
          beamRefreshDebug=0, beamRefFrame=True, beamRefresh=256)

pose(0, 0.0, 0.0)
hstr.set_properties(dict(BASE, beamRefMargin=0.15, worldCacheUpdates=1))
for i in range(int(os.environ.get("HSTR_SETTLE", "900"))):
    m.renderFrame()
    s = stats()
    if i > 64 and int(s.get("pending", 0)) == 0 and int(s.get("sunBakesFrame", 0)) == 0:
        break
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
# The world cache stops updating and residency freezes: a live cut re-bakes the sun and changes which bricks exist, and both move
# the carry counts for reasons that have nothing to do with parallax. This measures parallax.
hstr.set_properties({"worldCacheUpdates": 0, "cloudResidencyFrozen": True})
# The beam counters are only read back off the GPU while a reference comparison is active (HSTRCloud.cpp, the mCompareReference
# block), so a stored exact frame is captured once purely to open that path. Its error numbers are meaningless here - the stored
# frame is of one pose and the camera then flies away from it - and nothing below reads them.
m.renderFrame()
hstr.set_properties({"storeExact": True})
m.renderFrame()
hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 0})
log(f"sea: settled, mapped {stats().get('mapped', 0)}")

# forward units per frame, yaw radians per frame. The parked arm is the ceiling: with beamRefresh 256 a parked build still refreshes
# 1/256 of its points by policy, so survival is measured against what parked carries, not against the lattice size.
MOTIONS = [("parked", 0.0, 0.0), ("fly 1", 1.0, 0.0), ("fly 2", 2.0, 0.0), ("fly 5", 5.0, 0.0), ("fly 10", 10.0, 0.0),
           ("fly 20", 20.0, 0.0), ("yaw slow", 0.0, 0.004), ("yaw fast", 0.0, 0.05), ("fly 10 turn", 10.0, 0.03)]
ARMS = [("m15", 0.15), ("m50", 0.5)]

rows = []
for armLabel, margin in ARMS:
    ceiling = {}
    for label, forward, yaw in MOTIONS:
        hstr.set_properties(dict(BASE, beamRefMargin=margin))
        for i in range(WARM):
            pose(i - WARM, forward, yaw)
            m.renderFrame()
        before = stats()
        anchors = int(before.get("beamRefAnchors", 0))
        points, pixels, threads, anchored = [], [], [], 0
        priorThreads = int(before.get("beamGridThreads", 0))  # Cumulative over the run, so it is differenced per frame.
        for i in range(FRAMES):
            pose(i, forward, yaw)
            m.renderFrame()
            s = stats()
            now = int(s.get("beamRefAnchors", 0))
            anchored += now - anchors
            anchors = now
            nowThreads = int(s.get("beamGridThreads", 0))
            threads.append(nowThreads - priorThreads)
            priorThreads = nowThreads
            points.append(int(s.get("beamCarriedPoints", 0)))
            pixels.append(int(s.get("beamCarriedPixels", 0)))
        mean = lambda v: sum(v) / float(len(v))
        row = {"arm": armLabel, "motion": label, "points": mean(points), "pixels": mean(pixels),
               "threads": mean(threads), "anchors": anchored, "pointsMin": min(points), "pixelsMin": min(pixels)}
        if label == "parked":
            ceiling = {"points": max(row["points"], 1.0), "pixels": max(row["pixels"], 1.0)}
        row["pointSurvival"] = row["points"] / ceiling["points"]
        row["pixelSurvival"] = row["pixels"] / ceiling["pixels"]
        rows.append(row)
        log(f"{armLabel:4s} {label:12s} points {row['points']:12.0f} ({100 * row['pointSurvival']:6.2f}%)  "
            f"units {row['pixels']:12.0f} ({100 * row['pixelSurvival']:6.2f}%)  "
            f"threads {row['threads']:12.0f}  anchors {anchored:3d}/{FRAMES}")

with open(f"{OUT}/{TAG}_rows.json", "w") as f:
    json.dump(rows, f, indent=1)
log("")
log("survival is the carried count against the same arm's PARKED carried count. anchors are orientation-only: a pure fly arm")
log("cannot re-anchor, so its loss is parallax alone.")
exit()
