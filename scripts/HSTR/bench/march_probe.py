import json
import os
from falcor import *

# Per-pixel camera march counters at 4K on settled views (marchProbe): which path each camera sun query takes (baked, ancestor's
# bake, proxy, live march) and where the density samples go (step limits, empty, thin, working).
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "march_probe")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
PROPS = json.loads(os.environ.get("HSTR_PROPS", "{}"))  # HSTRCloud properties to probe with.
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(3840, 2160)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
views = [("near", (340, -10, 46), (6, -15, 46)), ("farside", (-330, -10, 46), (6, -15, 46)), ("sea", (0, 140, 0), (0, 40, 600))]
views = [v for v in views if v[0] in os.environ.get("HSTR_VIEWS", "near,farside,sea").split(",")]
lines = []


def log(line):
    lines.append(line)
    print(line)
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def settle():
    quiet = 0
    for i in range(1500):
        m.renderFrame()
        s = hstr.properties.get("cloudStats", {})
        busy = int(s.get("pendingTiles", 0)) + int(s.get("pending", 0)) + int(s.get("sunBakesFrame", 0))
        quiet = quiet + 1 if busy == 0 else 0
        if i > 32 and quiet > 20:
            return i
    return 1500


def mean_of(mode):
    # The probe frame against the stored black frame: the mean absolute difference is the mean count.
    hstr.set_properties({"marchProbe": mode})
    m.renderFrame()
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    value = float(hstr.properties["referenceError"])
    hstr.set_properties({"compareReference": False, "compareExact": False, "marchProbe": 0})
    return value


for name, position, target in views:
    cam.position = float3(*position)
    cam.target = float3(*target)
    hstr.set_properties(dict(PROPS, debugView=8, hstComponents=0, worldCacheUpdates=0))
    frames = settle()
    s = hstr.properties.get("cloudStats", {})
    log(f"{name}: settled in {frames} frames; bakes {s.get('sunBaked')}, waiting {s.get('sunWaiting')}, sun slots free {s.get('sunSlotsFree')}")
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    hstr.set_properties({"hstComponents": 3})
    k = {mode: mean_of(mode) for mode in range(1, 24) if mode != 19}
    q = max(k[1], 1e-9)
    log(f"{name}: {k[1]:6.2f} sun queries per pixel: own bake {100 * k[2] / q:5.1f}%, ancestor's bake {100 * k[18] / q:5.1f}%, proxy {100 * k[3] / q:5.1f}%, "
        f"live: finer than asked {100 * k[4] / q:5.1f}%, no bake {100 * k[5] / q:5.1f}%, empty {100 * k[6] / q:5.1f}%")
    d = max(k[7], 1e-9)
    log(f"{name}: steps {k[7] + k[17]:6.2f}/px (majorant zero {k[17]:5.2f}); density samples {k[7]:6.2f}/px: at the minimum step {100 * k[10] / d:5.1f}%, "
        f"at the maximum step {100 * k[11] / d:5.1f}%, exactly empty {100 * k[13] / d:5.1f}%, non-empty alpha <= 1e-6 {100 * k[12] / d:5.1f}%, "
        f"alpha > 0.5 {100 * k[14] / d:5.1f}%; mean step {k[15] / d:5.2f} sampled voxels; pixels terminated {100 * k[16]:5.1f}%")
    log(f"{name}: brick samples {k[21]:6.2f}/px in {k[20]:5.2f} brick visits/px ({k[21] / max(k[20], 1e-9):5.2f} samples per visit); "
        f"pixels still marching after 32 steps {100 * k[22]:5.1f}%, after 64 steps {100 * k[23]:5.1f}%")
exit()
