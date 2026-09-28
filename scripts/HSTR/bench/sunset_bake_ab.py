# Ancestral sun bakes (cloudSunAncestral: a bake reads only its brick and its ancestors, so no mapping ever outdates it) against
# the classic ones (the finest mapped bricks, rebaked whenever the neighbourhood maps or unmaps), on SunsetCloudSea.py's graph,
# headless, one process. Order classic -> ancestral -> classic (the anchor). Each mode rebakes everything (cloudSunRebake) and
# settles parked at the start pose; its parked frame is compared with the first classic one (same pose; the anchor gives the
# floor) and captured; then walk and sprint fly from the start pose: wall per frame, bake counters, GPU leaf scopes.
# Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/sunset_bake_ab.py   (HSTR_BAKE_OUT: capture directory)
import os
import time
from pathlib import Path
from falcor import *

root = Path(__file__).resolve().parents[3]
launcher = root / "scripts" / "HSTR" / "SunsetCloudSea.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
m.resizeFrameBuffer(3840, 2160)
hstr = m.activeGraph.getPass("HSTRCloud")
hstr.set_properties({"cloudSunBakesMoving": 0})  # The bake change alone.
cam = m.scene.camera
START = float3(cam.position.x, cam.position.y, cam.position.z)
VIEW = float3(cam.target.x, cam.target.y, cam.target.z) - START
OUT = os.environ.get("HSTR_BAKE_OUT", "C:/Users/Friss/Documents/HSTR_results/sunset_bake_ab")
LEVELS = ("desired", "mapped", "pending", "sunWaiting", "sunStale", "sunBakesFrame", "sunSlotsFree")


def stats():
    s = hstr.properties.get("cloudStats", {})
    return {k: s.get(k) for k in LEVELS}


def frame_at(p):
    cam.position = p
    cam.target = p + VIEW
    t0 = time.perf_counter()
    m.renderFrame()
    return (time.perf_counter() - t0) * 1000.0


def park(frames):
    return [frame_at(START) for _ in range(frames)]


def summary(walls):
    v = sorted(walls)
    return f"mean {sum(v) / len(v):.2f} ms, p50 {v[len(v) // 2]:.2f}, p95 {v[int(0.95 * (len(v) - 1))]:.2f}, max {v[-1]:.2f}"


def settle_bakes(limit):
    for n in range(0, limit, 30):
        park(30)
        s = stats()
        if not s["sunWaiting"] and (s["pending"] or 0) <= 256:
            return n + 30
    return limit


def profile(tag, speed, frames):
    m.profiler.enabled = True
    m.profiler.start_capture()
    for k in range(frames):
        frame_at(START + float3(0, 0, speed * (k + 1)))
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    lanes = {n: l for n, l in capture["events"].items() if n.endswith("/gpu_time")}
    paths = [n.rsplit("/", 1)[0] for n in lanes]
    leaves = {n: l for n, l in lanes.items() if not any(p.startswith(n.rsplit("/", 1)[0] + "/") for p in paths)}
    means = sorted(((sum(l["records"]) / max(len(l["records"]), 1), n) for n, l in leaves.items()), reverse=True)
    print(f"BAKE {tag} gpu leaves: " + ", ".join(f"{n.rsplit('/', 2)[-2]} {v:.3f}" for v, n in means[:8]), flush=True)


def fly(tag, speed):
    walls = [frame_at(START + float3(0, 0, speed * (k + 1))) for k in range(120)]
    print(f"BAKE {tag} v{speed:g}: {summary(walls)} {stats()}", flush=True)
    park(240)
    profile(f"{tag} v{speed:g}", speed, 30)
    park(240)
    print(f"BAKE {tag} parkedAfter v{speed:g}: {summary(park(60))} {stats()}", flush=True)


def compare(tag):
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    p = hstr.properties
    print(f"BAKE {tag} parked vs classic: >0.02 {100 * float(p['referenceNoiseError']):.3f}%, p99.9 {float(p['referenceLogP999']):.3g}, "
          f"max {float(p['referenceLogMax']):.3g}", flush=True)
    hstr.set_properties({"compareReference": False, "compareExact": False})


def capture(tag):
    m.frameCapture.outputDir = OUT
    m.frameCapture.baseFilename = f"sunset_bake_{tag}"
    m.frameCapture.capture()


os.makedirs(OUT, exist_ok=True)
for index, (tag, ancestral) in enumerate([("classic", False), ("ancestral", True), ("classicAgain", False)]):
    hstr.set_properties({"cloudSunAncestral": ancestral, "cloudSunRebake": True})
    frames = park(1800) if index == 0 else None
    settled = settle_bakes(1800)
    print(f"BAKE {tag}: settled in {settled} frames after the rebake {stats()}", flush=True)
    print(f"BAKE {tag} parked: {summary(park(60))}", flush=True)
    if index == 0:
        hstr.set_properties({"storeExact": True})
        m.renderFrame()
    else:
        compare(tag)
    capture(tag)
    # HSTR_BAKE_FLY=0: the parked comparison alone. Flights between the modes grow the resident set (desired 71k -> 243k in
    # sunset_bake_ab1), and the later modes' parked frames then differ from the first by that LOD change, not by their bakes
    # (the classic anchor scored 10.4% of pixels over 0.02 against the first classic frame).
    if os.environ.get("HSTR_BAKE_FLY", "1") != "0":
        fly(tag, 2.0)
        fly(tag, 20.0)
        park(300)
exit()
