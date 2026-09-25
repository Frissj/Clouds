import json
import os
from falcor import *

# Beam view of the cloud sea with a moving camera at 4K: GPU ms while flying, and the error of every step of a flight against the
# exact per-pixel frame of the same camera (the stored frame of HSTR_BASE, as in ab_test). A parked camera hides everything a
# temporal beam gets wrong, so this is the benchmark for beamRefresh. Every test flies the same path from the same start: a warm-up
# that builds its history, a timed flight, and a flight of compared steps. HSTR_MOTIONS lists the camera motions as
# [label, forward units per frame, yaw radians per frame]. Results go to <tag>_test.txt and <tag>_test.jsonl.
OUT = "C:/Users/Friss/Documents/HSTR_results"
TAG = os.environ.get("HSTR_TAG", "sea_motion")
os.environ["HSTR_CLOUD_LIBRARY"] = os.environ.get("HSTR_CLOUD_LIBRARY", "C:/Users/Friss/Downloads/clouds_hr/codec/v6_default")
BASE = json.loads(os.environ["HSTR_BASE"])
TESTS = json.loads(os.environ["HSTR_TESTS"])
MOTIONS = json.loads(os.environ.get("HSTR_MOTIONS", '[["fly 2", 2.0, 0.0], ["fly 20", 20.0, 0.0], ["yaw", 0.0, 0.004]]'))
# READ THIS BEFORE COMPARING A WORLD-PERSISTENT CACHE ACROSS ARMS. WARM is how many frames an arm runs before its timed window,
# and for the anchored rectangle it barely matters: the rectangle re-anchors and forgets, so it reaches the same state from any
# history, and its numbers here are stable to 0.03 ms across every run of this session. The octahedral image does not forget. Its
# cost depends on how much of the sphere is already resident, so it depends on everything the camera did first, and WARM decides
# that. Measured at 720p: the same octahedral arm at 0.01 rad/frame reported 0.64, 0.54, 0.08, 0.08 ms as it was simply repeated
# within one run at WARM 16 - the first arms were still paying the fresh build that beamReset forces, which for the sphere clears
# roughly 300 MB of texture and marches a whole first footprint. Raising WARM to 96 moved 0.01 rad/frame from 0.92 to 0.70 and
# 0.05 rad/frame from 0.61 to 1.43, in opposite directions, because 96 frames of turning at 0.05 fills most of the sphere.
#
# So a single number for a persistent directional cache under a fixed-length synthetic motion is not well defined, and none of the
# octahedral timings taken this way should be quoted as ITS cost. What they support is the comparison of one mechanism against
# another inside one arm - query threads, residual threads, which pass moved - and the rectangle's own numbers. Characterising the
# octahedral image needs a protocol that states the residency it starts from, which this harness does not yet express.
WARM, TIMED, STEPS = int(os.environ.get("HSTR_WARM", "16")), 48, int(os.environ.get("HSTR_STEPS", "12"))
TRACE_SUN = os.environ.get("HSTR_TRACE_SUN", "0") != "0"  # Sums the sun scheduler's per-frame counts over the timed flight.
m.script("scripts/HSTR/CloudSea.py")
# HSTR_RES=1920x1080 settles in a fraction of the time, for residency diagnostics (the cut and the timings are not the 4K ones).
m.resizeFrameBuffer(*[int(v) for v in os.environ.get("HSTR_RES", "3840x2160").split("x")])
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
START_POSITION, START_TARGET = float3(0, 140, 0), float3(0, 40, 600)
ORIGIN = START_POSITION  # The pose frame 0 passes through; motions labelled "near ..." move it to NEAR_POSITION (see below).
lines = []


def log(line):
    lines.append(line)
    print(line)
    with open(f"{OUT}/{TAG}_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def stats():
    return hstr.properties.get("cloudStats", {})


def settle():
    quiet = 0
    for i in range(1500):
        m.renderFrame()
        s = stats()
        # Idle means the residency has nothing left to change, not only nothing left to load: freezing it (below) also stops fades
        # and map/unmap work mid-way. Checked on the sea (2026-09-22): nothing was mid-fade where the streaming-only test ended, so
        # this is hygiene, not the cause of the sprint flakiness - that was sea tiles landing between a step's stored exact frame
        # and its scored frame, fixed by the sea taking tiles on synchronously while residency is frozen (updateCloudDomain).
        busy = sum(int(s.get(k, 0)) for k in ("pendingTiles", "pending", "sunBakesFrame", "activeFades", "mapBacklog",
                                                "undesiredFadingOut"))
        quiet = quiet + 1 if busy == 0 else 0
        if i > 32 and quiet > 20:
            return i + 1
    print("sea_motion: the residency never went idle in 1500 frames; runs are not comparable")
    return 1500


# Radians of sun rotation per frame, as a motion's optional fourth element ["label", forward, yaw, sun]; HSTR_SUN_RATE is the
# default for motions that do not give one. A carried beam basis holds LIGHTING as well as transport, so a moving sun is the one
# hazard a fixed refresh rate is genuinely insurance against, and no other motion here exercises it. The per-step oracle is
# captured at the same sun as the frame it scores, so correctness needs no special handling.
SUN_RATE = float(os.environ.get("HSTR_SUN_RATE", "0"))


def pose(frame, forward, yaw, sun=0.0):
    # Frame 0 is the settled view; the path runs through it, so every flight starts WARM frames before it.
    import math
    if sun != 0.0:
        a = sun * frame
        hstr.set_properties({"sunDirection": float3(math.cos(a) * 0.6, 0.75, math.sin(a) * 0.6)})
    offset = float3(0, 0, forward * frame)
    angle = yaw * frame
    view = START_TARGET - START_POSITION
    turned = float3(view.x * math.cos(angle) - view.z * math.sin(angle), view.y, view.x * math.sin(angle) + view.z * math.cos(angle))
    cam.position = ORIGIN + offset
    cam.target = ORIGIN + offset + turned


NSYS = os.environ.get("HSTR_NSYS")  # run_sea.py --nsys: capture each timed flight, and only it.


def nsys(verb, *options):
    import subprocess
    subprocess.run([NSYS, verb, f"--session={os.environ['HSTR_NSYS_SESSION']}", *options], creationflags=subprocess.CREATE_NO_WINDOW)


def timed(frames, first, forward, yaw, sun=0.0, report=None):
    import time
    if NSYS and report:
        metric_set = [f"--gpu-metrics-set={os.environ['HSTR_NSYS_SET']}"] if os.environ.get("HSTR_NSYS_SET") else []
        nsys("start", "--sample=none", "--gpu-metrics-devices=all", "--gpu-metrics-frequency=20000", "--force-overwrite=true",
             *metric_set, f"--output={os.environ['HSTR_NSYS_OUTPUT']}_{report}")
    m.profiler.enabled = True
    m.profiler.start_capture()
    before = stats()
    cuts = before.get("cuts", 0)
    wall = time.perf_counter()
    trace = {"sunStale": 0, "sunBakesFrame": 0}
    peak = {"unmapBacklog": 0, "mapBacklog": 0, "activeFades": 0, "undesiredFadingOut": 0}
    for i in range(frames):
        pose(first + i, forward, yaw, sun)
        m.renderFrame()
        if TRACE_SUN:  # Reads the counts back every frame: the timings are not the shipping ones.
            s = stats()
            for k in trace:
                trace[k] += int(s.get(k, 0))
            for k in peak:
                peak[k] = max(peak[k], int(s.get(k, 0)))
    wall = (time.perf_counter() - wall) / frames * 1000.0
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    if NSYS and report:
        nsys("stop")  # After end_capture, which has waited for the flight's GPU work.
    t = {name[:-len("/gpu_time")].split("HSTRCloud", 1)[-1].strip("/") or "HSTRCloud": lane["stats"]["mean"]
         for name, lane in capture["events"].items() if name.endswith("gpu_time") and "HSTRCloud" in name}
    t["wall"] = wall
    # The tone mapper reads HSTRCloud's colour output, so the output format moves its cost too (outside the HSTRCloud scope).
    t["toneMapper"] = next((lane["stats"]["mean"] for name, lane in capture["events"].items()
                            if name.endswith("ToneMapper/gpu_time")), 0.0)
    # Per frame of the flight (a scope's own mean covers only the frames it ran in, so nested means do not add up).
    import math
    def per_frame(lane):
        return round(sum(v for v in lane["records"] if isinstance(v, (int, float)) and math.isfinite(v)) / frames, 3)
    t["cpu"] = {name[:-len("/cpu_time")].split("cloudSea", 1)[-1].strip("/") or "cloudSea": per_frame(lane)
                for name, lane in capture["events"].items() if name.endswith("cpu_time") and "cloudSea" in name}
    t["cpuMax"] = {name[:-len("/cpu_time")].split("cloudSea", 1)[-1].strip("/") or "cloudSea": round(lane["stats"]["max"], 2)
                   for name, lane in capture["events"].items() if name.endswith("cpu_time") and "cloudSea" in name}
    # The frames of any scope that spiked past 1 ms, to place its spikes (the jsonl only).
    t["cpuFrames"] = {name[:-len("/cpu_time")].split("cloudSea", 1)[-1].strip("/") or "cloudSea": [round(v, 2) for v in lane["records"]]
                      for name, lane in capture["events"].items()
                      if name.endswith("cpu_time") and "cloudSea" in name and lane["stats"]["max"] >= 1.0}
    s = stats()
    t["residency"] = {k: s.get(k, 0) for k in ("loaded", "mapped", "desired", "pending", "cutMs", "sunWaiting", "sunBakesFrame", "sunStale", "cutPops",
                                               "cutOrdered", "cutTotalMs", "cutMargin")}
    t["residency"]["cuts"] = s.get("cuts", 0) - cuts  # Cuts run during the timed frames.
    # Page table edits during the timed frames (per frame), and what waited at the end.
    for k in ("maps", "unmaps", "fadeStarts", "fadeEnds", "fadeVoid"):
        t["residency"][k] = round((s.get(k, 0) - before.get(k, 0)) / frames, 3)
    for k in ("mapBacklog", "undesiredFadingOut", "undesiredHeld", "undesiredIdle", "staleFades"):
        t["residency"][k] = s.get(k, 0)
    if TRACE_SUN:
        t["residency"]["flight"] = trace
        t["residency"]["peak"] = peak
    return t


pose(0, 0.0, 0.0)
hstr.set_properties(dict(BASE, debugView=9, hstComponents=15, worldCacheUpdates=1))
frames = settle()
while int(hstr.properties["worldCacheSampleCount"]) < 64:
    m.renderFrame()
# The resident set is frozen once settled: otherwise every camera change re-runs the CPU residency cut (100+ ms), the GPU idles into a
# lower power state and the timings measure that instead of the frame.
hstr.set_properties({"worldCacheUpdates": 0, "cloudResidencyFrozen": os.environ.get("HSTR_FREEZE", "1") != "0"})
settled = stats()
log(f"sea: settled in {frames} frames (mapped {settled.get('mapped', 0)}, sun baked {settled.get('sunBaked', 0)}, waiting "
    f"{settled.get('sunWaiting', 0)}, slots free {settled.get('sunSlotsFree', 0)}, world cache samples "
    f"{hstr.properties.get('worldCacheSampleCount', 0)}, fades {settled.get('activeFades', 0)}, map backlog "
    f"{settled.get('mapBacklog', 0)}, fading out {settled.get('undesiredFadingOut', 0)})")
# Near cloud: the start pose looks at cloud from clear air (beamPushProbe, 2026-09-23: no visible cell within 8 voxels), which never
# exercises cells covering much of the screen. Motions labelled "near ..." fly through the pose, among a few around the start with
# the same view direction, whose cells nearer than 8 voxels overlap the most tiles (the probe's pushNearOverlaps).
NEAR_POSITION = START_POSITION
if any(motion.startswith("near") for motion, *_ in MOTIONS):
    # The probe only runs in the octahedral beam view, so the first arm's configuration, not BASE (the reference's).
    hstr.set_properties(dict(BASE, **dict(TESTS[0][1], beamPushProbe=True, pushTileSize=16)))
    best = None
    for dy in (-120.0, -100.0, -60.0, -20.0):
        for dx in (-150.0, 0.0, 150.0):
            for dz in (0.0, 150.0, 300.0, 450.0, 600.0):
                ORIGIN = START_POSITION + float3(dx, dy, dz)
                pose(0, 0.0, 0.0)
                m.renderFrame()
                m.renderFrame()
                s = stats()
                nearest = float(s.get("pushMinDistance", -1))
                score = (int(s.get("pushNearOverlaps", 0)), -(nearest if nearest >= 0 else 1e9))
                log(f"near search {dx:+.0f} {dy:+.0f} {dz:+.0f}: nearest {float(s.get('pushMinDistance', -1)):.1f} voxels, near "
                    f"overlaps {s.get('pushNearOverlaps', 0)}, mid {s.get('pushMidOverlaps', 0)}, visible {s.get('pushVisible', 0)}, "
                    f"camera in a cell {s.get('pushNear', 0)}")
                if best is None or score > best[0]:
                    best = (score, ORIGIN)
    NEAR_POSITION = best[1]
    ORIGIN = START_POSITION
    log(f"near pose {NEAR_POSITION}")
# Every key any arm sets, at the value it had before the first arm: an arm that does not set a key gets this back, not the last
# arm's. Without it a key set by one arm stayed on for every arm after it - budgetwarp1 / budgetwarp2 scored walk "par 8" with
# the previous arm's beamWarp still on, which read as run-to-run variation (1.439 / 0.960% against 1.985%) and got a working
# warp removed.
PRIOR = {k: hstr.properties[k] for test, properties in TESTS for k in properties if k not in BASE and k in hstr.properties}
for motion, forward, yaw, *rest in MOTIONS:
    ORIGIN = NEAR_POSITION if motion.startswith("near") else START_POSITION
    sun = float(rest[0]) if rest else SUN_RATE
    for test, properties in TESTS:
        hstr.set_properties(dict(BASE, **dict(PRIOR, **properties)))
        # Every arm starts from an empty beam image. The reference frame re-anchors on its own so it hardly noticed, but the
        # octahedral image is fixed to the world and persists until its dimensions change: without this an arm inherits the
        # directions its predecessors marched, and one configuration measured 0.36 ms and then 2.41 ms in consecutive runs while
        # the rectangle control reproduced to 0.07. The warm-up below then fills it from this arm's own motion.
        hstr.set_properties({"beamReset": True})
        # Sea tiles replaced during the arm: the sea streams even with residency frozen, and a tile landing between a direction's
        # march and a scored step changes the cloud under the persistent beam image.
        tilesAtStart = int(stats().get("seaTilesChanged", 0))
        invalidatedAtStart = int(stats().get("beamInvalidatedBuilds", 0))
        for i in range(WARM):
            pose(i - WARM - TIMED, forward, yaw, sun)
            m.renderFrame()
        t = timed(TIMED, -TIMED, forward, yaw, sun, report=f"{motion}_{test}".replace(" ", "_"))
        tilesBeforeSteps = int(stats().get("seaTilesChanged", 0)) - tilesAtStart
        tilesPerStep = []
        errors = []
        for step in range(STEPS):
            # The scored frame is the arm's own frame in flight: the pose after the last one it rendered, with its guard history
            # intact. It used to be rendered right after the reference frame, and switching views drops the beam history, so every
            # scored frame re-marched the whole screen (every guard block dirty and unverified) and no reuse - the guard budget,
            # carried blocks - was ever scored: budgetparallax1 gave beamGuardParallax 1 / 2 / 4 / 8 identical errors while the
            # flight's query fell 0.70 -> 0.56 ms at 8. So the arm frame is stored and the reference frame compared against it
            # (the log error is symmetric). Later steps re-fly WARM frames first, since the previous reference frame broke it.
            if step > 0:
                for i in range(WARM):
                    pose(step - WARM + i, forward, yaw, sun)
                    m.renderFrame()
            pose(step, forward, yaw, sun)
            # compare* too, only so the beam counters (dirty blocks, marched fraction) are read back from this frame: it
            # compares the frame it has just stored with itself.
            hstr.set_properties({"storeExact": True, "compareReference": True, "compareExact": True, "compareBlock": 1})
            m.renderFrame()
            s = dict(stats())
            marched = float(hstr.properties["beamMarchedFraction"])
            hstr.set_properties(dict(BASE, compareReference=False, compareExact=False))
            m.renderFrame()  # As before, the reference view renders once before the frame that is compared.
            hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
            m.renderFrame()
            p = hstr.properties
            hstr.set_properties(dict(BASE, **dict(properties, compareReference=False, compareExact=False)))
            tilesPerStep.append(int(s.get("seaTilesChanged", 0)) - tilesAtStart)
            errors.append({"over02": float(p["referenceNoiseError"]), "p999": float(p["referenceLogP999"]),
                           "max": float(p["referenceLogMax"]), "marched": marched,
                           "carriedPoints": int(s.get("beamCarriedPoints", 0)), "carriedPixels": int(s.get("beamCarriedPixels", 0)),
                           "marchTiles": int(s.get("beamMarchTiles", 0)), "debug": int(s.get("beamRefreshDebugCount", 0)),
                           "marchedSteps": int(s.get("beamMarchedSteps", 0)), "carriedSteps": int(s.get("beamCarriedSteps", 0)),
                           # The dirty passes' work: blocks listed, units marched (own tile / apron), cells classified.
                           "dirty": {k: int(s.get("beam" + k, 0)) for k in ("DirtyBlocks", "DirtyUnverified", "DirtyOwnMarched",
                                                                           "DirtyApronMarched", "ClassifyCells", "FrameDim", "RefAnchors",
                                                                           "WarpHeld", "WarpListed", "WarpOn")},
                           # cellViews: the scored frame's composition (0 with it off).
                           "cell": {k: int(s.get("cellView" + k, 0)) for k in ("Rays", "Hits", "EmptyHits", "Exact", "Cells", "Requests",
                                                                                 "Built", "Steps")},
                           # beamPushProbe: the scored frame's cell x tile work count (absent with it off); spanProbe's too.
                           "push": {k: s[k] for k in s if k.startswith(("push", "span")) or k == "beamFrameDim"}})
        if not errors:  # HSTR_STEPS=0: timings and residency only.
            errors = [{"over02": 0.0, "p999": 0.0, "max": 0.0, "marched": 0.0, "carriedPoints": 0, "carriedPixels": 0, "marchTiles": 0,
                       "debug": 0, "marchedSteps": 0, "carriedSteps": 0}]
        mean = {k: sum(e[k] for e in errors) / len(errors) for k in errors[0] if k not in ("cell", "push", "dirty")}
        cell = errors[-1].get("cell", {})
        push = errors[-1].get("push", {})
        worst = max(e["over02"] for e in errors)
        with open(f"{OUT}/{TAG}_test.jsonl", "a") as f:
            f.write(json.dumps({"motion": motion, "test": test, "gpu": t, "errors": errors}) + "\n")
        parts = " ".join(f"{k} {v:.2f}" for k, v in t.items() if k not in ("HSTRCloud", "residency", "cpu", "cpuMax", "cpuFrames") and v >= 0.3)
        parts += " " + json.dumps(t["residency"]) + " cpu per frame " + json.dumps({k: v for k, v in t["cpu"].items() if v >= 0.05})
        parts += " cpu max " + json.dumps({k: v for k, v in t["cpuMax"].items() if v >= 1.0})
        log(f"{motion:7s} {test:16s} {t.get('HSTRCloud', 0):6.2f} ms ({parts}); marched {100 * mean['marched']:4.1f}%; "
            f"carried {mean['carriedPoints']:.0f} points {mean['carriedPixels']:.0f} pixels (debug {mean['debug']:.0f}, steps marched {mean['marchedSteps']:.0f} carried {mean['carriedSteps']:.0f}); "
            f">0.02 mean {100 * mean['over02']:.3f}% worst {100 * worst:.3f}%, p99.9 {mean['p999']:.2e}, max {max(e['max'] for e in errors):.2e}"
            f"; sea tiles changed before steps {tilesBeforeSteps}, by step {tilesPerStep}"
            f", invalidating builds {int(stats().get('beamInvalidatedBuilds', 0)) - invalidatedAtStart}"
            f", per-step >0.02 {[round(100 * e['over02'], 3) for e in errors]}" + (f"; cell views (last step) {cell}" if cell.get("Rays") else "")
            + (f"; push (last step) {json.dumps(push)}; push ms " + json.dumps({k: round(v, 3) for k, v in t.items() if k.startswith("push")})
               if push else ""))
exit()
