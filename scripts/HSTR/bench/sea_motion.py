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
        busy = int(s.get("pendingTiles", 0)) + int(s.get("pending", 0)) + int(s.get("sunBakesFrame", 0))
        quiet = quiet + 1 if busy == 0 else 0
        if i > 32 and quiet > 20:
            return i
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
    cam.position = START_POSITION + offset
    cam.target = START_POSITION + offset + turned


def timed(frames, first, forward, yaw, sun=0.0):
    import time
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
    t = {name[:-len("/gpu_time")].split("HSTRCloud", 1)[-1].strip("/") or "HSTRCloud": lane["stats"]["mean"]
         for name, lane in capture["events"].items() if name.endswith("gpu_time") and "HSTRCloud" in name}
    t["wall"] = wall
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
    f"{settled.get('sunWaiting', 0)}, slots free {settled.get('sunSlotsFree', 0)})")
for motion, forward, yaw, *rest in MOTIONS:
    sun = float(rest[0]) if rest else SUN_RATE
    for test, properties in TESTS:
        hstr.set_properties(dict(BASE, **properties))
        # Every arm starts from an empty beam image. The reference frame re-anchors on its own so it hardly noticed, but the
        # octahedral image is fixed to the world and persists until its dimensions change: without this an arm inherits the
        # directions its predecessors marched, and one configuration measured 0.36 ms and then 2.41 ms in consecutive runs while
        # the rectangle control reproduced to 0.07. The warm-up below then fills it from this arm's own motion.
        hstr.set_properties({"beamReset": True})
        for i in range(WARM):
            pose(i - WARM - TIMED, forward, yaw, sun)
            m.renderFrame()
        t = timed(TIMED, -TIMED, forward, yaw, sun)
        errors = []
        for step in range(STEPS):
            pose(step, forward, yaw, sun)
            hstr.set_properties(BASE)
            m.renderFrame()
            hstr.set_properties({"storeExact": True})
            m.renderFrame()
            hstr.set_properties(dict(properties, compareReference=True, compareExact=True, compareBlock=1))
            m.renderFrame()
            p = hstr.properties
            hstr.set_properties({"compareReference": False, "compareExact": False})
            s = p.get("cloudStats", {})
            errors.append({"over02": float(p["referenceNoiseError"]), "p999": float(p["referenceLogP999"]),
                           "max": float(p["referenceLogMax"]), "marched": float(p["beamMarchedFraction"]),
                           "carriedPoints": int(s.get("beamCarriedPoints", 0)), "carriedPixels": int(s.get("beamCarriedPixels", 0)),
                           "marchTiles": int(s.get("beamMarchTiles", 0)), "debug": int(s.get("beamRefreshDebugCount", 0)),
                           "marchedSteps": int(s.get("beamMarchedSteps", 0)), "carriedSteps": int(s.get("beamCarriedSteps", 0))})
        if not errors:  # HSTR_STEPS=0: timings and residency only.
            errors = [{"over02": 0.0, "p999": 0.0, "max": 0.0, "marched": 0.0, "carriedPoints": 0, "carriedPixels": 0, "marchTiles": 0,
                       "debug": 0, "marchedSteps": 0, "carriedSteps": 0}]
        mean = {k: sum(e[k] for e in errors) / len(errors) for k in errors[0]}
        worst = max(e["over02"] for e in errors)
        with open(f"{OUT}/{TAG}_test.jsonl", "a") as f:
            f.write(json.dumps({"motion": motion, "test": test, "gpu": t, "errors": errors}) + "\n")
        parts = " ".join(f"{k} {v:.2f}" for k, v in t.items() if k not in ("HSTRCloud", "residency", "cpu", "cpuMax", "cpuFrames") and v >= 0.3)
        parts += " " + json.dumps(t["residency"]) + " cpu per frame " + json.dumps({k: v for k, v in t["cpu"].items() if v >= 0.05})
        parts += " cpu max " + json.dumps({k: v for k, v in t["cpuMax"].items() if v >= 1.0})
        log(f"{motion:7s} {test:16s} {t.get('HSTRCloud', 0):6.2f} ms ({parts}); marched {100 * mean['marched']:4.1f}%; "
            f"carried {mean['carriedPoints']:.0f} points {mean['carriedPixels']:.0f} pixels (debug {mean['debug']:.0f}, steps marched {mean['marchedSteps']:.0f} carried {mean['carriedSteps']:.0f}); "
            f">0.02 mean {100 * mean['over02']:.3f}% worst {100 * worst:.3f}%, p99.9 {mean['p999']:.2e}, max {max(e['max'] for e in errors):.2e}")
exit()
