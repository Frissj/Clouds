import json
import os
import sys
from falcor import *

# HOW OFTEN does each thing that can invalidate a carried beam basis actually happen, with residency live? Frequency only.
#
# READ THIS BEFORE ADDING A TIMING OR QUALITY NUMBER HERE. This script reads hstr.properties every frame to classify the frame,
# and that read forces a CPU-GPU sync. Its GPU times were inflated by roughly 2x because of it - it reported 4.47 ms parked where
# sea_motion.py, which reads nothing inside its timed loop, measures 2.09 ms for the same configuration. Its per-frame quality
# loop was worse than useless: scoring a carried basis needs an oracle capture, and switching debugView to take one rebuilt the
# basis, so two arms 256x apart in refresh rate scored bit-identically; under motion it scored 30% of pixels over 0.02 in every
# arm, which was the measurement collapsing rather than the renderer.
#
# So timings and quality come from sea_motion.py (run_sea.py motion ... --live). This answers only what that harness cannot: the
# event frequencies that decide whether a fixed refresh rate can become event-driven.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sea_config import mk

OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "live_invalidation")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
FRAMES = int(os.environ.get("HSTR_FRAMES", "300"))
WARM = int(os.environ.get("HSTR_WARM", "16"))
FORWARD = float(os.environ.get("HSTR_FORWARD", "2"))
YAW = float(os.environ.get("HSTR_YAW", "0"))
SUN_RATE = float(os.environ.get("HSTR_SUN_RATE", "0"))  # Radians per frame, as in sea_motion.

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


def pose(frame):
    import math
    if SUN_RATE != 0.0:
        a = SUN_RATE * frame
        hstr.set_properties({"sunDirection": float3(math.cos(a) * 0.6, 0.75, math.sin(a) * 0.6)})
    angle = YAW * frame
    view = START_TARGET - START_POSITION
    turned = float3(view.x * math.cos(angle) - view.z * math.sin(angle), view.y,
                    view.x * math.sin(angle) + view.z * math.cos(angle))
    cam.position = START_POSITION + float3(0, 0, FORWARD * frame)
    cam.target = cam.position + turned


BASE = mk(beamTileSize=4, beamLevels=1, beamSegments=1, beamTemporal=False, beamSparse=False, beamSparseCut=False,
          beamQueue=False, beamGridDispatch=False, beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05,
          beamRefreshDebug=0, beamRefFrame=True, beamRefMargin=0.15, beamRefresh=16)

pose(0)
hstr.set_properties(dict(BASE, worldCacheUpdates=1))
for i in range(int(os.environ.get("HSTR_SETTLE", "900"))):
    m.renderFrame()
    s = stats()
    if i > 64 and int(s.get("pending", 0)) == 0 and int(s.get("sunBakesFrame", 0)) == 0:
        break
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
log(f"sea: settled, mapped {stats().get('mapped', 0)}")
# RESIDENCY stays live; the world cache does not. Leaving worldCacheUpdates on keeps the cache accumulating samples and re-baking
# the sun every frame, which is the startup transient rather than a running sea. sea_motion --live draws the line in the same place.
hstr.set_properties({"worldCacheUpdates": 0, "cloudResidencyFrozen": os.environ.get("HSTR_FREEZE", "0") != "0"})

for i in range(WARM):
    pose(i - WARM)
    m.renderFrame()
before = stats()
prior = {"anchors": int(before.get("beamRefAnchors", 0)), "density": int(before.get("densityChangedFrames", 0)),
         "sun": int(before.get("sunBakeFrames", 0))}
frames = []
for i in range(FRAMES):
    pose(i)
    m.renderFrame()
    s = stats()
    now = {"anchors": int(s.get("beamRefAnchors", 0)), "density": int(s.get("densityChangedFrames", 0)),
           "sun": int(s.get("sunBakeFrames", 0))}
    frames.append({"frame": i, "density": now["density"] - prior["density"], "sun": now["sun"] - prior["sun"],
                   "anchor": now["anchors"] - prior["anchors"], "cutMs": float(s.get("cutMs", 0.0))})
    prior = now

with open(f"{OUT}/{TAG}_frames.json", "w") as f:
    json.dump(frames, f)


def klass(f):
    if f["anchor"]:
        return "re-anchor"
    if f["density"] and f["sun"]:
        return "density+sun"
    if f["density"]:
        return "density"
    if f["sun"]:
        return "sun"
    return "quiet"


log(f"forward {FORWARD}, yaw {YAW}, sun {SUN_RATE} rad/frame, {FRAMES} frames")
for event in ("quiet", "density", "sun", "density+sun", "re-anchor"):
    n = sum(1 for f in frames if klass(f) == event)
    if n:
        log(f"  {event:12s} {n:5d} frames  {100.0 * n / len(frames):5.1f}%")
log(f"  cut over 8 ms: {sum(1 for f in frames if f['cutMs'] >= 8.0)} frames")
exit()
