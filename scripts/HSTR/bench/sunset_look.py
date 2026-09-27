# The sunset sea's look and cost: SunsetCloudSea.py's graph, headless, residency live. Per arm (a cloudSeaLayers value; the sea is
# rebuilt between arms, in the same process): settles, captures the start view, turns 60 degrees right and captures again, and prints
# the profiler-off wall time per frame over each window.
# Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/sunset_look.py
#   HSTR_RES=WxH, HSTR_LOOK_OUT=capture directory, HSTR_LOOK_LAYERS=comma list of layer counts (default "2,1"),
#   HSTR_LOOK_PROPS=a Python dict literal of HSTRCloud properties applied before the arms.
import math
import os
import time
from pathlib import Path
from falcor import *

root = Path(__file__).resolve().parents[3]
launcher = root / "scripts" / "HSTR" / "SunsetCloudSea.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
m.resizeFrameBuffer(*[int(v) for v in os.environ.get("HSTR_RES", "3840x2160").split("x")])
hstr = m.activeGraph.getPass("HSTRCloud")
if os.environ.get("HSTR_LOOK_PROPS"):
    hstr.set_properties(eval(os.environ["HSTR_LOOK_PROPS"]))
    print("LOOK props " + os.environ["HSTR_LOOK_PROPS"], flush=True)
cam = m.scene.camera
START_POSITION = float3(cam.position.x, cam.position.y, cam.position.z)
START_VIEW = float3(cam.target.x, cam.target.y, cam.target.z) - START_POSITION
yaw = [0.0]


def fly(frames, dyaw=0.0):
    walls = []
    for _ in range(frames):
        yaw[0] += dyaw
        c, s = math.cos(yaw[0]), math.sin(yaw[0])
        cam.position = START_POSITION
        cam.target = START_POSITION + float3(START_VIEW.x * c + START_VIEW.z * s, START_VIEW.y, -START_VIEW.x * s + START_VIEW.z * c)
        t0 = time.perf_counter()
        m.renderFrame()
        walls.append((time.perf_counter() - t0) * 1000.0)
    return walls


def report(label, walls):
    v = sorted(walls)
    print(f"LOOK {label}: mean {sum(v) / len(v):.2f} ms, p95 {v[int(0.95 * (len(v) - 1))]:.2f}, max {v[-1]:.2f}", flush=True)


def capture(label):
    m.frameCapture.outputDir = os.environ.get("HSTR_LOOK_OUT", os.environ.get("TEMP", "."))
    m.frameCapture.baseFilename = f"sunset_{label}"
    m.frameCapture.capture()


SUN_KEYS = ("desired", "mapped", "pending", "sunBaked", "sunWaiting", "sunSlotsFree", "sunStale", "sunBakesFrame")


def stats(tag):
    s = hstr.properties.get("cloudStats", {})
    print(f"LOOK {tag} " + str({k: s.get(k) for k in SUN_KEYS}), flush=True)


def profile(tag, frames):
    # GPU time per leaf scope over a parked window (the profiler adds its own overhead: compare arms, not against the wall times).
    m.profiler.enabled = True
    m.profiler.start_capture()
    fly(frames)
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    lanes = {name: lane for name, lane in capture["events"].items() if name.endswith("/gpu_time")}
    paths = [name.rsplit("/", 1)[0] for name in lanes]
    leaves = {name: lane for name, lane in lanes.items() if not any(p.startswith(name.rsplit("/", 1)[0] + "/") for p in paths)}
    means = sorted(((sum(lane["records"]) / max(len(lane["records"]), 1), name) for name, lane in leaves.items()), reverse=True)
    frame = next((sum(l["records"]) / len(l["records"]) for n, l in lanes.items() if n.endswith("/onFrameRender/gpu_time")), 0.0)
    print(f"LOOK {tag} profiled frame gpu {frame:.2f} ms; top leaves:", flush=True)
    for value, name in means[:14]:
        print(f"LOOK {tag}   {value:6.3f} ms  {name.rsplit('/', 1)[0].split('/onFrameRender/')[-1]}", flush=True)


# HSTR_LOOK_ARMS: a Python list literal of property dicts, one arm each (the sea or residency rebuilds between arms as needed).
ARMS = eval(os.environ.get("HSTR_LOOK_ARMS", "[{'cloudSeaLayers': 2}, {'cloudSeaLayers': 1}]"))
for index, arm in enumerate(ARMS):
    yaw[0] = 0.0
    hstr.set_properties(arm)
    tag = f"A{index}"
    print(f"LOOK {tag} = {arm}", flush=True)
    for checkpoint in range(6):
        report(f"{tag} settle{checkpoint}", fly(300))
        stats(f"{tag} settle{checkpoint}")
    report(f"{tag} parked", fly(60))
    profile(f"{tag} parked", 60)
    capture(f"{tag}_start")
    report(f"{tag} turn", fly(30, math.radians(60) / 30))
    stats(f"{tag} afterTurn")
    profile(f"{tag} afterTurn", 30)
    report(f"{tag} parkedAfterTurn", fly(90))
    stats(f"{tag} parkedAfterTurn")
    capture(f"{tag}_turned")
    s = hstr.properties.get("cloudStats", {})
    print(f"LOOK {tag} stats " + str({k: s.get(k) for k in ("desired", "mapped", "pending", "mapBacklog", "sunBaked", "sunWaiting")}), flush=True)
exit()
