import json
import os
from falcor import *

# Fast check that the cloud sea's projected transport cut is built and evaluated: a short settle at a small frame size, then the
# beam view with the cut evaluator against the same view with the volume marcher, compared per pixel against the exact frame. It
# answers "is the cut carrying the image at all", not "is it fast" - use run_sea.py ab for timings.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "cut_smoke")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(1280, 720)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
cam.position = float3(340, -10, 46)
cam.target = float3(6, -15, 46)
lines = []


def log(line):
    lines.append(line)
    print(line)
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


BASE = dict(
    debugView=9, hstComponents=15, beamTemporal=False, cloudSunCache=True, beamTileSize=32, beamLevels=4, beamSegments=1,
    beamTolerance=0.05, beamEdgeContrast=0.5, beamRefresh=0, beamShip=True, beamShipMask=509, beamSparse=True,
    beamSparseMinLevel=2, cutTransmittanceTolerance=0.01,
)
hstr.set_properties(dict(BASE, beamSparseCut=False, worldCacheUpdates=1))
for i in range(int(os.environ.get("HSTR_SETTLE", "300"))):
    m.renderFrame()
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0})
m.renderFrame()
hstr.set_properties({"storeExact": True})
m.renderFrame()

for label, props in [("volume", {"beamSparseCut": False}), ("cut", {"beamSparseCut": True})]:
    hstr.set_properties(dict(BASE, **props))
    m.renderFrame()
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    p = hstr.properties
    s = p.get("cloudStats", {})
    hstr.set_properties({"compareReference": False, "compareExact": False})
    log(f"{label:7s} log {float(p['referenceLogError']):.3e} >0.02 {100 * float(p['referenceNoiseError']):.3f}% "
        f">0.1 {100 * float(p['referenceNoiseLogError']):.3f}% p99.9 {float(p['referenceLogP999']):.2e} "
        f"max {float(p['referenceLogMax']):.2e}; queries {int(s.get('beamUniqueQueries', 0))} "
        f"final {int(s.get('beamFinalTiles', 0))}; cut paged/marched {int(s.get('beamCutPaged', 0))}/"
        f"{int(s.get('beamCutMarched', 0))} residual {int(s.get('beamCutResidualQueries', 0))} q "
        f"{int(s.get('beamCutResidualSteps', 0))} steps")
exit()
