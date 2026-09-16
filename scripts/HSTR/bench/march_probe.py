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
    k = {mode: mean_of(mode) for mode in range(1, 30) if mode != 19}
    q = max(k[1], 1e-9)
    log(f"{name}: {k[1]:6.2f} sun queries per pixel: own bake {100 * k[2] / q:5.1f}%, ancestor's bake {100 * k[18] / q:5.1f}%, proxy {100 * k[3] / q:5.1f}%, "
        f"live: finer than asked {100 * k[4] / q:5.1f}%, no bake {100 * k[5] / q:5.1f}%, empty {100 * k[6] / q:5.1f}%")
    d = max(k[7], 1e-9)
    log(f"{name}: steps {k[7] + k[17]:6.2f}/px (majorant zero {k[17]:5.2f}); density samples {k[7]:6.2f}/px: at the minimum step {100 * k[10] / d:5.1f}%, "
        f"at the maximum step {100 * k[11] / d:5.1f}%, exactly empty {100 * k[13] / d:5.1f}%, non-empty alpha <= 1e-6 {100 * k[12] / d:5.1f}%, "
        f"alpha > 0.5 {100 * k[14] / d:5.1f}%; mean step {k[15] / d:5.2f} sampled voxels; pixels terminated {100 * k[16]:5.1f}%")
    log(f"{name}: brick samples {k[21]:6.2f}/px in {k[20]:5.2f} brick visits/px ({k[21] / max(k[20], 1e-9):5.2f} samples per visit); "
        f"pixels still marching after 32 steps {100 * k[22]:5.1f}%, after 64 steps {100 * k[23]:5.1f}%")
    # The price of dilating the majorant: samples the march takes because a neighbouring block holds density, in a block that holds
    # none itself. Mode 25 is the share of those a conservative tight test could reject without a brick walk - the rest sit within
    # half a voxel of a block face, where the trilinear support crosses into the neighbour and the dilation is doing real work.
    log(f"{name}: tight-zero samples {k[24]:6.2f}/px ({100 * k[24] / d:5.1f}% of density samples), of which rejectable "
        f"{k[25]:6.2f}/px ({100 * k[25] / d:5.1f}%); exactly empty {k[13]:6.2f}/px for comparison")
    # What an ancestor table keyed on (entry, desired level) would replace: the parent climb in cloudAssetBrickAt walks .parent and
    # .info per iteration, each address coming out of the previous load. If it barely runs, the table is not worth building.
    log(f"{name}: parent-climb iterations {k[28]:6.2f}/px over {k[29]:5.2f} samples that climb ({100 * k[29] / max(k[21], 1e-9):4.1f}% of "
        f"brick samples, {k[28] / max(k[29], 1e-9):4.2f} steps each)")
    # The run a compact interval representation would stand for. Samples per brick visit counts the empty and faint ones too, and
    # those are nearly free, so the length that decides whether an interval can pay is the contributing one. Near 1 means it cannot.
    log(f"{name}: contributing brick samples {k[26]:6.2f}/px over {k[27]:5.2f} contributing entries/px = "
        f"{k[26] / max(k[27], 1e-9):5.2f} per entry (against {k[21] / max(k[20], 1e-9):5.2f} counting every sample in the visit)")
    # What a transfer cache would face on the path that ships: brick crossings it could serve against the distinct (asset brick,
    # source-space direction class, entry cell) entries it would have to produce. Break-even is Z / (Z - 1) with Z the events per
    # crossing above, so about 1.45. The beam view traces far fewer rays than there are pixels, which is the whole question.
    hstr.set_properties(dict(PROPS, debugView=9, hstComponents=15))
    for edge in (4, 8, 16):
        hstr.set_properties({"cloudTransferClasses": edge})
        m.renderFrame()
        m.renderFrame()
        t = hstr.properties.get("cloudStats", {})
        crossings, entries = int(t.get("transferCrossings", 0)), int(t.get("transferEntries", 0))
        log(f"{name}: transfer reuse at {edge * edge:4d} direction classes: {crossings} crossings over {entries} entries "
            f"= {crossings / max(entries, 1):5.2f} per entry")
    hstr.set_properties({"cloudTransferClasses": 0})
exit()
