# Why the Intel sea takes minutes to settle: the interactive launcher's graph, headless, with GPU/CPU scope timings over the first
# frames. Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/intel_load_probe.py   (HSTR_INTEL_SET=half|full, HSTR_RES=WxH)
import math
import os
import time
from pathlib import Path
from falcor import *

root = Path(__file__).resolve().parents[3]
launcher = root / "scripts" / "HSTR" / f"IntelCloudSea{os.environ.get('HSTR_INTEL_SET', 'half').capitalize()}.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
m.resizeFrameBuffer(*[int(v) for v in os.environ.get("HSTR_RES", "1920x1080").split("x")])
hstr = m.activeGraph.getPass("HSTRCloud")


cam = m.scene.camera
START_POSITION = float3(cam.position.x, cam.position.y, cam.position.z)  # A copy: cam.position is a live reference.
START_VIEW = float3(cam.target.x, cam.target.y, cam.target.z) - START_POSITION
travel = [0.0, 0.0]  # Distance flown along +z and yaw turned so far: each window continues from the last.


def window(label, frames, forward=0.0, yaw=0.0):
    m.profiler.enabled = True
    m.profiler.start_capture()
    wall = time.perf_counter()
    for _ in range(frames):
        travel[0] += forward
        travel[1] += yaw
        c, s = math.cos(travel[1]), math.sin(travel[1])
        cam.position = START_POSITION + float3(0, 0, travel[0])
        cam.target = cam.position + float3(START_VIEW.x * c - START_VIEW.z * s, START_VIEW.y, START_VIEW.x * s + START_VIEW.z * c)
        m.renderFrame()
    wall = (time.perf_counter() - wall) / frames * 1000.0
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    lanes = []
    for name, lane in capture["events"].items():
        if not (name.endswith("gpu_time") or name.endswith("cpu_time")):
            continue
        total = sum(v for v in lane["records"] if isinstance(v, (int, float)) and math.isfinite(v)) / frames
        lanes.append((total, name))
    lanes.sort(reverse=True)
    s = hstr.properties.get("cloudStats", {})
    print(f"PROBE {label}: wall {wall:.1f} ms/frame; desired {s.get('desired')} loaded {s.get('loaded')} mapped {s.get('mapped')} "
          f"pending {s.get('pending')} committed {s.get('committed')} sunBakes {s.get('sunBakesFrame')}", flush=True)
    for total, name in lanes[:6]:
        print(f"PROBE   {total:9.2f} ms  {name}", flush=True)
    keys = ("desired", "loaded", "mapped", "pending", "mapBacklog", "unmaps", "cuts", "cutTotalMs", "pendingTiles", "residentMB",
            "sunBaked", "sunWaiting", "activeFades")
    print(f"PROBE   camera {cam.position} -> {cam.target}; {({k: s.get(k) for k in keys})}", flush=True)
    m.frameCapture.outputDir = os.environ.get("TEMP", ".")
    m.frameCapture.baseFilename = "intel_probe_" + label.split()[0] + "_" + str(int(travel[0]))
    m.frameCapture.capture()


window("startup 60", 60)
window("parked 240", 240)
window("walk 2/frame 120", 120, forward=2.0)
window("parked after walk 120", 120)
window("turn 180 over 60", 60, yaw=math.pi / 60)
window("parked after turn 120", 120)
exit()
