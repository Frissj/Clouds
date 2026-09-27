# The Intel sea's interactive start and turns: the launcher's graph, headless, with residency live (as the launcher runs it, unlike
# sea_motion.py, which freezes it). Phase 1 flies startup, park, walk, a 180 degree turn and park with the defaults and captures the
# frame after each (the stale beam image showed there as black blocks and holes). Phase 2 continues the flight alternating
# beamChangeCellVoxels 4 / 0 (the GPU change list on / off) every 20 frames, for their cost on one path and residency state.
# Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/intel_load_probe.py   (HSTR_INTEL_SET=half|full, HSTR_RES=WxH)
import math
import os
import time
from pathlib import Path
from falcor import *

root = Path(__file__).resolve().parents[3]
launcher = root / "scripts" / "HSTR" / f"IntelCloudSea{os.environ.get('HSTR_INTEL_SET', 'half').capitalize()}.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
m.resizeFrameBuffer(*[int(v) for v in os.environ.get("HSTR_RES", "3840x2160").split("x")])
hstr = m.activeGraph.getPass("HSTRCloud")
OUT = os.environ.get("TEMP", ".")

cam = m.scene.camera
START_POSITION = float3(cam.position.x, cam.position.y, cam.position.z)  # A copy: cam.position is a live reference.
START_VIEW = float3(cam.target.x, cam.target.y, cam.target.z) - START_POSITION
travel = [0.0, 0.0]  # Distance flown along +z and yaw turned so far: each window continues from the last.
SCOPES = ("/onFrameRender/gpu_time", "markBeamChanges/gpu_time", "invalidate/gpu_time", "beamQueries/gpu_time", "march/gpu_time",
          "resolve/gpu_time")


def fly(frames, forward, yaw):
    for _ in range(frames):
        travel[0] += forward
        travel[1] += yaw
        c, s = math.cos(travel[1]), math.sin(travel[1])
        cam.position = START_POSITION + float3(0, 0, travel[0])
        cam.target = cam.position + float3(START_VIEW.x * c - START_VIEW.z * s, START_VIEW.y, START_VIEW.x * s + START_VIEW.z * c)
        m.renderFrame()


def timed(frames, forward=0.0, yaw=0.0):
    """Per-frame ms of SCOPES (summed over the frames a scope ran, divided by all frames) and wall ms."""
    m.profiler.enabled = True
    m.profiler.start_capture()
    wall = time.perf_counter()
    fly(frames, forward, yaw)
    wall = (time.perf_counter() - wall) / frames * 1000.0
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    t = {"wall": wall}
    for scope in SCOPES:
        t[scope] = sum(sum(v for v in lane["records"] if isinstance(v, (int, float)) and math.isfinite(v))
                       for name, lane in capture["events"].items() if name.endswith(scope)) / frames
    return t


def stats_line():
    s = hstr.properties.get("cloudStats", {})
    keys = ("desired", "mapped", "pending", "mapBacklog", "activeFades", "sunBaked", "sunWaiting", "sunBakesFrame", "beamInvalidatedBuilds")
    return str({k: s.get(k) for k in keys})


def capture(label):
    m.frameCapture.outputDir = OUT
    m.frameCapture.baseFilename = f"intel_ab_{label}"
    m.frameCapture.capture()


PHASES = os.environ.get("HSTR_PHASES", "12")

if "3" in PHASES:
    # Phase 3: the change list's cost parked while a turn's new view streams in, within one window: the frames that listed changes
    # (the invalidate scope ran) against those that did not. Park, turn 180 degrees in 20 frames, park 240 frames while it streams.
    # The capture opens 10 parked frames before the turn: its first frames spike whatever the camera does.
    fly(290, 0.0, 0.0)
    m.profiler.enabled = True
    m.profiler.start_capture()
    fly(10, 0.0, 0.0)
    fly(20, 0.0, math.pi / 20)
    fly(240, 0.0, 0.0)
    capture3 = m.profiler.end_capture()
    m.profiler.enabled = False
    def records(suffix):
        lane = next((lane for name, lane in capture3["events"].items() if name.endswith(suffix)), None)
        return lane["records"] if lane else []
    frame = records("/onFrameRender/gpu_time")
    listed = records("invalidate/gpu_time")
    ok = lambda v: isinstance(v, (int, float)) and math.isfinite(v)
    marks = [ok(v) and v > 0.0 for v in listed] + [False] * (len(frame) - len(listed))
    for lo, hi in ((0, 10), (10, 30), (30, 110), (110, 190), (190, 270)):  # Parked, turning, then parked while it streams in.
        hi = min(hi, len(frame))  # The capture may hold fewer records than frames flown.
        on =[frame[i] for i in range(lo, hi) if marks[i] and ok(frame[i])]
        off = [frame[i] for i in range(lo, hi) if not marks[i] and ok(frame[i])]
        allv = sorted(on + off)
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        p95 = allv[min(len(allv) - 1, int(0.95 * len(allv)))] if allv else float("nan")
        print(f"PROBE phase3 frames {lo}-{hi}: mean {mean(allv):.3f} ms, p95 {p95:.3f}, max {max(allv, default=float('nan')):.3f}; "
              f"listing frames {mean(on):.3f} (n {len(on)}), others {mean(off):.3f} (n {len(off)})", flush=True)
    # Where the worst frames go: their largest leaf scopes (no other scope nested under them), GPU and CPU.
    paths = {name.rsplit("/", 1)[0] for name in capture3["events"]}
    leaves = {p for p in paths if not any(q.startswith(p + "/") for q in paths)}
    ok_frames = [i for i in range(len(frame)) if ok(frame[i])]
    for i in sorted(ok_frames, key=lambda i: frame[i], reverse=True)[:4]:
        print(f"PROBE   worst frame {i}: {frame[i]:.2f} ms listing {marks[i]}", flush=True)
        for kind in ("gpu_time", "cpu_time"):
            scopes = sorted(((lane["records"][i], name.rsplit("/", 1)[0].split("HSTRCloud/", 1)[-1])
                             for name, lane in capture3["events"].items()
                             if name.endswith(kind) and name.rsplit("/", 1)[0] in leaves and i < len(lane["records"])
                             and ok(lane["records"][i])), reverse=True)[:6]
            print(f"PROBE     {kind[:3]}: " + ", ".join(f"{n} {v:.2f}" for v, n in scopes), flush=True)
    print(f"PROBE   {stats_line()}", flush=True)

# Phase 1: defaults (the change list on), captures.
for label, frames, forward, yaw in ([] if "1" not in PHASES else (("startup", 60, 0.0, 0.0), ("parked", 240, 0.0, 0.0), ("walk", 120, 2.0, 0.0),
                                     ("turn", 60, 0.0, math.pi / 60), ("parkedAfterTurn", 120, 0.0, 0.0))):
    t = timed(frames, forward, yaw)
    print(f"PROBE phase1 {label}: " + " ".join(f"{k.split('/')[-2] if '/' in k else k} {v:.2f}" for k, v in t.items()), flush=True)
    print(f"PROBE   {stats_line()}", flush=True)
    capture(label)

# Phase 2: arms alternating every 20 frames on one path.
CHUNK = 24
ARMS = {"batched8": {"beamChangeCellVoxels": 4, "beamChangeInterval": 8}, "off": {"beamChangeCellVoxels": 0},
        "everyFrame": {"beamChangeCellVoxels": 4, "beamChangeInterval": 1}}
for label, frames, forward, yaw in ([] if "2" not in PHASES else (("parked", 216, 0.0, 0.0), ("walk", 216, 2.0, 0.0),
                                     ("turn", 216, 0.0, math.pi / 108), ("sprint", 216, 20.0, 0.0))):
    sums = {arm: {} for arm in ARMS}
    chunks = frames // CHUNK
    for chunk in range(chunks):
        arm = list(ARMS)[chunk % len(ARMS)]
        hstr.set_properties(ARMS[arm])
        t = timed(CHUNK, forward, yaw)
        for k, v in t.items():
            sums[arm][k] = sums[arm].get(k, 0.0) + v / (chunks // len(ARMS))
    for arm in ARMS:
        print(f"PROBE phase2 {label} {arm}: " + " ".join(f"{k.split('/')[-2] if '/' in k else k} {v:.3f}" for k, v in sums[arm].items()),
              flush=True)
    print(f"PROBE   {stats_line()}", flush=True)
hstr.set_properties(ARMS["batched8"])
exit()
