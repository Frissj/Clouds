import os
import sys
from falcor import *

# Under translation, what does the dirty march actually march? beamDirtyOwnMarched / beamDirtyApronMarched count the units the dirty
# tile pass listed, per build (a listed block's own, and those of the tiles around it); beamRepairProbe scores what a prediction
# could have kept of every re-marched sample, by count and by march steps, and how much of each wave's march its lanes spent idle.
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
           beamGuardParallax=1.0, beamRepairProbe=True, beamShadowCarry=int(os.environ.get("HSTR_SHADOW", "0")))

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
# The same idle test as sea_motion.py: a single frame without bakes is not a settled sea (it once stopped with 350k bakes waiting).
quiet = 0
for i in range(int(os.environ.get("HSTR_SETTLE", "1500"))):
    m.renderFrame()
    s = stats()
    busy = sum(int(s.get(k, 0)) for k in ("pendingTiles", "pending", "sunBakesFrame", "activeFades", "mapBacklog", "undesiredFadingOut"))
    quiet = quiet + 1 if busy == 0 else 0
    if i > 32 and quiet > 20:
        break
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0, "cloudResidencyFrozen": True})
# The beam counters only leave the GPU while a comparison is active; its error numbers are not read here.
m.renderFrame()
hstr.set_properties({"storeExact": True})
m.renderFrame()
hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 0})
log(f"sea: settled, mapped {stats().get('mapped', 0)}, {W}x{H}; sun baked {stats().get('sunBaked', 0)} waiting "
    f"{stats().get('sunWaiting', 0)} stale {stats().get('sunStale', 0)} slots free {stats().get('sunSlotsFree', 0)}")

MOTIONS = [("walk", 2.0, 0.004), ("sprint", 20.0, 0.0)]
# HSTR_SHADOWS: beamShadowCarry values to run the motions under, one settle for all (0: the plain probe).
SHADOWS = [int(v) for v in os.environ.get("HSTR_SHADOWS", str(ARM["beamShadowCarry"])).split(",")]
for shadow, (motion, forward, yaw) in [(sh, mo) for sh in SHADOWS for mo in MOTIONS]:
    ARM["beamShadowCarry"] = shadow
    log(f"--- beamShadowCarry {shadow}")
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
                     for k in ("Scored", "InPlace", "Reprojected", "Either")] +
                    [int(s.get(f"beamProbe{kind}Steps{k}", 0)) for kind in ("Unit", "Ray")
                     for k in ("Scored", "InPlace", "Reprojected", "Either")] +
                    [int(s.get(f"beamProbe{k}", 0)) for k in ("UnitStepsTaken", "UnitStepsPaid", "RayStepsTaken", "RayStepsPaid")] +
                    [int(s.get(f"beamProbe{k}", 0)) for k in ("Blocks", "BlocksOk", "BlockOkSteps", "BlockSteps", "TightSteps",
                                                              "TightBadSteps", "LooseSteps", "LooseBadSteps")] +
                    [int(s.get(f"beamProbeSlot{k}", 0)) for k in range(60, 240)])
    mean = lambda k: sum(r[k] for r in rows) / float(len(rows))
    blocks, own, apron, fraction = mean(0), mean(1), mean(2), mean(3)
    log(f"{motion:6s} dirty blocks {blocks:9.0f}  own units {blocks * 64:10.0f}  own marched {own:10.0f} ({100 * own / max(blocks * 64, 1):5.1f}%)"
        f"  apron marched {apron:9.0f}  pixels in marched tiles {100 * fraction:5.1f}%  frame units {W * H}")
    # beamRepairProbe: of the samples this build re-marched, how many a repair could have predicted within 0.02 (log(1 + x)).
    for i, kind in enumerate(("units", "rays")):
        scored, inPlace, reprojected, either = (mean(4 + 4 * i + k) for k in range(4))
        pct = lambda v: 100 * v / max(scored, 1)
        log(f"{motion:6s}   {kind:5s} re-marched {scored:9.0f}: kept in place ok {pct(inPlace):5.1f}%, reprojected ok {pct(reprojected):5.1f}%,"
            f" either {pct(either):5.1f}%, neither (true repair) {100 - pct(either):5.1f}%")
        # The same, weighted by the march steps each sample cost.
        steps, sIn, sRe, sEither = (mean(12 + 4 * i + k) for k in range(4))
        sp = lambda v: 100 * v / max(steps, 1)
        log(f"{motion:6s}   {kind:5s} by steps {steps:11.0f} ({steps / max(scored, 1):6.1f}/sample): in place ok {sp(sIn):5.1f}%, reprojected ok"
            f" {sp(sRe):5.1f}%, either {sp(sEither):5.1f}%")
        # Wave efficiency of the march: steps the lanes took over what their waves paid (the longest lane, times the lanes).
        taken, paid = mean(20 + 2 * i), mean(21 + 2 * i)
        log(f"{motion:6s}   {kind:5s} wave efficiency {100 * taken / max(paid, 1):5.1f}% ({taken:11.0f} steps taken, {paid:11.0f} paid)")
    # Whole blocks: the ceiling of a coherent block carry, and what a current-view geometry signature would have accepted of it.
    nb, nok, okSteps, allSteps, tight, tightBad, loose, looseBad = (mean(24 + k) for k in range(8))
    sp = lambda v: 100 * v / max(allSteps, 1)
    log(f"{motion:6s}   blocks {nb:9.0f}: every ray ok in place {100 * nok / max(nb, 1):5.1f}% of blocks, {sp(okSteps):5.1f}% of their steps")
    log(f"{motion:6s}   geometry tight (opacity 0.02, distance 2%): accepts {sp(tight):5.1f}% of steps, wrongly {sp(tightBad):5.1f}%")
    log(f"{motion:6s}   geometry loose (opacity 0.05, distance 10%): accepts {sp(loose):5.1f}% of steps, wrongly {sp(looseBad):5.1f}%")
    # Witness matrix (beamProbeWitnesses): per signature and policy, query steps a carry would save (accepted and right, as a share
    # of the scored blocks' steps), steps accepted wrongly, and the witnesses' own extinction steps against the same total.
    slot = lambda k: mean(32 + k - 60)
    signatures = ["opacity", "+centroid", "+quantiles", "+sun d50", "+sun+cache", "quant 5%", "sun+c 5%", "radiance",
                  "radiance.01", "rad+sun+c"]
    policies = ["centre", "2 centres", "4 centres", "all 8", "nearest", "nearest 2"]
    cost = [slot(180 + p) for p in range(6)]
    log(f"{motion:6s}   witness cost (extinction steps / query steps): " +
        ", ".join(f"{policies[p]} {100 * cost[p] / max(allSteps, 1):4.1f}%" for p in range(6)))
    log(f"{motion:6s}   {'signature':11s} " + " ".join(f"{p:>17s}" for p in policies) + "   (right% / wrong% of query steps)")
    for sig in range(10):
        cells = []
        for p in range(6):
            accepted, bad = slot(60 + 2 * (sig * 6 + p)), slot(61 + 2 * (sig * 6 + p))
            cells.append(f"{sp(accepted - bad):7.1f} /{sp(bad):5.1f}   ")
        log(f"{motion:6s}   {signatures[sig]:11s} " + " ".join(f"{c:>17s}" for c in cells))
    # The residual of accepted blocks: share of ALL re-marched unit steps that lie in accepted blocks, and of those, the share within
    # 0.02 in place and kept-or-reprojected.
    unitSteps = mean(12)
    log(f"{motion:6s}   raw unit steps {unitSteps:.0f}, first accepted cell {slot(186):.0f} / {slot(187):.0f} / {slot(188):.0f}")
    for j, sig in enumerate((4, 7, 9)):
        cells = []
        for p in range(6):
            b = j * 6 + p
            total, kept, either = slot(186 + 3 * b), slot(187 + 3 * b), slot(188 + 3 * b)
            cells.append(f"{100 * total / max(unitSteps, 1):4.1f}: {100 * kept / max(total, 1):4.1f}/{100 * either / max(total, 1):4.1f}")
        log(f"{motion:6s}   units {signatures[sig]:11s} " + " ".join(f"{c:>17s}" for c in cells) + "   (% of unit steps: kept/either ok)")
exit()
