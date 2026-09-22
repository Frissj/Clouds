import os
import sys
from falcor import *

# Under translation, what does the dirty unit march actually march? It dispatches (edge + 2 apron)^2 threads per listed block - 24^2
# = 576 for 8 x 8 own units at tile 4 - forcing the block's own units and asking the apron's as usual. If most threads march, the
# work is real and only fewer invalidations can cut it (reprojection); if most return early, a dense one-thread-per-unit pass or a
# deduplicated unit list can. beamDirtyOwnMarched / beamDirtyApronMarched count the marched units, per build.
#
# Headless: HSTR_TAG=probe Mogwai.exe --headless --script=scripts/HSTR/bench/dirty_march_probe.py (HSTR_RES defaults to 4K).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runpy  # noqa: E402

OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "dirty_march_probe")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
FRAMES = int(os.environ.get("HSTR_FRAMES", "16"))
WARM = int(os.environ.get("HSTR_WARM", "16"))
ARM = dict(runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweeps", "persistent_residual.py"))["OCT_T001"],
           beamGuardParallax=1.0, beamRepairProbe=True)

m.script("scripts/HSTR/CloudSea.py")
W, H = [int(v) for v in os.environ.get("HSTR_RES", "3840x2160").split("x")]
m.resizeFrameBuffer(W, H)
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


pose(0, 0.0, 0.0)
hstr.set_properties(dict(ARM, worldCacheUpdates=1))
for i in range(int(os.environ.get("HSTR_SETTLE", "1500"))):
    m.renderFrame()
    s = stats()
    if i > 64 and int(s.get("pending", 0)) == 0 and int(s.get("sunBakesFrame", 0)) == 0 and int(s.get("pendingTiles", 0)) == 0:
        break
hstr.set_properties({"worldCacheUpdates": 0, "cloudResidencyFrozen": True})
# The beam counters only leave the GPU while a comparison is active; its error numbers are not read here.
m.renderFrame()
hstr.set_properties({"storeExact": True})
m.renderFrame()
hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 0})
log(f"sea: settled, mapped {stats().get('mapped', 0)}, {W}x{H}")

MOTIONS = [("walk", 2.0, 0.004), ("sprint", 20.0, 0.0)]
for motion, forward, yaw in MOTIONS:
    hstr.set_properties(dict(ARM))
    hstr.set_properties({"beamReset": True})
    for i in range(WARM):
        pose(i - WARM, forward, yaw)
        m.renderFrame()
    rows = []
    for i in range(FRAMES):
        pose(i, forward, yaw)
        m.renderFrame()
        s = stats()
        rows.append([int(s.get("beamDirtyBlocks", 0)), int(s.get("beamDirtyOwnMarched", 0)), int(s.get("beamDirtyApronMarched", 0)),
                     float(hstr.properties.get("beamMarchedFraction", 0.0))] +
                    [int(s.get(f"beamProbe{kind}{k}", 0)) for kind in ("Units", "Rays")
                     for k in ("Scored", "InPlace", "Reprojected", "Either")])
    mean = lambda k: sum(r[k] for r in rows) / float(len(rows))
    blocks, own, apron, fraction = mean(0), mean(1), mean(2), mean(3)
    threads = blocks * 576
    log(f"{motion:6s} dirty blocks {blocks:9.0f}  threads {threads:11.0f}  own slots {blocks * 64:10.0f}  own marched {own:10.0f}"
        f" ({100 * own / max(blocks * 64, 1):5.1f}%)  apron marched {apron:9.0f}  marched/threads {100 * (own + apron) / max(threads, 1):5.1f}%"
        f"  pixels in marched tiles {100 * fraction:5.1f}%  frame units {W * H}")
    # beamRepairProbe: of the samples this build re-marched, how many a repair could have predicted within 0.02 (log(1 + x)).
    for i, kind in enumerate(("units", "rays")):
        scored, inPlace, reprojected, either = (mean(4 + 4 * i + k) for k in range(4))
        pct = lambda v: 100 * v / max(scored, 1)
        log(f"{motion:6s}   {kind:5s} re-marched {scored:9.0f}: kept in place ok {pct(inPlace):5.1f}%, reprojected ok {pct(reprojected):5.1f}%,"
            f" either {pct(either):5.1f}%, neither (true repair) {100 - pct(either):5.1f}%")
exit()
