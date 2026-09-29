# Are the sunset sea's box artifacts brick level seams? The path tracer (footprint 0) and the exact march read the finest resident
# brick, so where the desired level changes from brick to brick the density jumps at the brick border. Per arm (same process): settle
# with residency live, freeze, capture the exact per-pixel march (sea_config.REFERENCE) at the start view, print the residency.
# Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/sunset_levels.py
#   HSTR_LEVELS_ARMS=a Python list literal of (name, property dict), HSTR_LEVELS_SETTLE=frames per arm (900).
import os
import sys
from pathlib import Path
from falcor import *

root = Path(__file__).resolve().parents[3]
launcher = root / "scripts" / "HSTR" / "SunsetCloudSea.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
m.resizeFrameBuffer(*[int(v) for v in os.environ.get("HSTR_RES", "3840x2160").split("x")])
hstr = m.activeGraph.getPass("HSTRCloud")
hstr.set_properties({"atmosphereAerialDistance": 0.0})
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sea_config import REFERENCE

OUT = os.environ.get("HSTR_LEVELS_OUT", "C:/Users/Friss/Documents/HSTR_results/sunset_levels")
LEVELS = ("desired", "mapped", "pending", "pendingQueued", "pendingInFlight", "pendingCompletions", "pendingWaiting", "arrivalsStored", "arrivalsDropped", "arrivalsFailed", "storesReleased", "payloadAllocFailed", "releaseNowFrames", "mapBacklog", "sunWaiting", "sunBakesFrame", "payloadMB", "cuts", "cutTotalMs", "cutPops")
ARMS = eval(os.environ.get("HSTR_LEVELS_ARMS", "None")) or [
    ("default", {"cloudLodBias": 0.0}),
    ("level0", {"cloudLodBias": -8.0, "cloudBrickPoolMB": 2048})]
SETTLE = int(os.environ.get("HSTR_LEVELS_SETTLE", "900"))


def stats():
    s = hstr.properties.get("cloudStats", {})
    return {k: s.get(k) for k in LEVELS}


beam = {k: hstr.properties[k] for k in REFERENCE if k in hstr.properties}
os.makedirs(OUT, exist_ok=True)
m.frameCapture.outputDir = OUT
for name, props in ARMS:
    hstr.set_properties(dict(beam, cloudResidencyFrozen=False, cloudTraceSlot=int(os.environ.get("HSTR_LEVELS_TRACE", "-1")), **props))
    for i in range(0, SETTLE, 300):
        for _ in range(300):
            m.renderFrame()
        print(f"LEVELS {name} settle{i // 300}: {stats()}", flush=True)
    # HSTR_LEVELS_TRACE: an instance slot whose cut walk expansions are written to OUT/trace_<arm>.txt (cloudTraceSlot).
    if os.environ.get("HSTR_LEVELS_TRACE"):
        with open(f"{OUT}/trace_{name}.txt", "w") as f:
            f.write(hstr.properties.get("cutTrace", ""))
    hstr.set_properties({"cloudResidencyFrozen": True})
    # HSTR_LEVELS_VIEWS: a Python list literal of (name, property dict) exact-march variants captured on this arm's settle.
    for view, extra in eval(os.environ.get("HSTR_LEVELS_VIEWS", "[('exact', {})]")):
        extra = dict(extra)
        # 'exposure': the tone mapper's exposure for this view (marchProbe count maps need it far below 0 to stay under white).
        m.activeGraph.getPass("ToneMapper").set_properties({"exposureCompensation": float(extra.pop("exposure", 0.0))})
        # 'beam': True starts from the launcher's beam view instead of REFERENCE (warmed 240 frames, the beam's own settle).
        shipped = extra.pop("beam", False)
        hstr.set_properties(dict(beam if shipped else REFERENCE, **extra))
        for _ in range(240 if shipped else 8):
            m.renderFrame()
        m.frameCapture.baseFilename = f"sunset_levels_{name}_{view}"
        m.frameCapture.capture()
        print(f"LEVELS {name} {view} {extra}", flush=True)
        # The first arm's first view is kept as the exact frame; every later view is scored against it (the gate's measures).
        if name == ARMS[0][0] and view == "exact":
            hstr.set_properties({"storeExact": True})
            m.renderFrame()
        else:
            hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
            m.renderFrame()
            p = hstr.properties
            print(f"LEVELS {name} {view} vs {ARMS[0][0]} exact: >0.02 {100 * float(p['referenceNoiseError']):.3f}%  p99.9 "
                  f"{float(p['referenceLogP999']):.4f}  max {float(p['referenceLogMax']):.3f}", flush=True)
            hstr.set_properties({"compareReference": False, "compareExact": False})
    # HSTR_LEVELS_PROBES: a Python list literal of (x, y) pixels whose exact-march steps are dumped (probePixel), sun-single view.
    # HSTR_LEVELS_PROBE_PROPS: the probed view's properties over REFERENCE. The live sun (cloudSunCache False) at REFERENCE's
    # source-voxel steps outlasts the GPU timeout at 4K (device removed), so keep the cache there.
    for x, y in eval(os.environ.get("HSTR_LEVELS_PROBES", "[]")):
        probeProps = eval(os.environ.get("HSTR_LEVELS_PROBE_PROPS", "{'hstComponents': 2, 'cloudSunCache': False}"))
        hstr.set_properties(dict(REFERENCE, probeX=x, probeY=y, **probeProps))
        m.renderFrame()
        values = [float(v) for v in hstr.properties.get("probeRecords", "").split(",") if v]
        print(f"PROBE {name} pixel ({x}, {y}): {len(values) // 12} steps", flush=True)
        # layers: per layer, 1 empty square, 2 proxy footprint, 3 outside the asset, 4 no brick, 5 brick (gProbeLayers).
        print("PROBE   dist     sigma     sunDepth  light      brick empty cell lvl des map  entry  inst layers  x y z", flush=True)
        for i in range(0, len(values) - 11, 12):
            d, sigma, sun, light, flags, entry, inst, layers, sx, sy, sz, empty = values[i:i + 12]
            f = int(flags)
            # empty: 1 missing child octant, 2 empty cell, 4 skirt asked, 8 skirt said yes (gProbeEmpty).
            print(f"PROBE {d:8.2f} {sigma:9.5f} {sun:9.4f} {light:10.6f}   {f & 1} {(f >> 1) & 1} {(f >> 2) & 1}  {(f >> 4) & 15} "
                  f"{(f >> 8) & 15} {(f >> 12) & 15} {int(entry):7d} {int(inst):5d}   {int(layers) & 15}{(int(layers) >> 4) & 15} "
                  f"e{int(empty) & 15:<2d} {sx:.1f} {sy:.1f} {sz:.1f}  skirt side {(int(empty) >> 4) & 3}{(int(empty) >> 6) & 3}"
                  f"{(int(empty) >> 8) & 3} looked {(int(empty) >> 10) & 255:08b} nlevel {(int(empty) >> 20) & 15} "
                  f"nchild {(int(empty) >> 24) & 1}", flush=True)
        hstr.set_properties({"probeX": -1, "probeY": -1})
    # HSTR_LEVELS_PHASES: a Python list literal of (name, property dict, frames) run in turn with residency frozen; after each, the
    # exact view is captured and scored against the first arm's exact frame - which phase changes the resident medium.
    for phase, extra, count in eval(os.environ.get("HSTR_LEVELS_PHASES", "[]")):
        hstr.set_properties(dict(beam, **extra))
        for _ in range(count):
            m.renderFrame()
        hstr.set_properties(dict(REFERENCE, debugView=REFERENCE.get("debugView", 8)))
        for _ in range(8):
            m.renderFrame()
        m.frameCapture.baseFilename = f"sunset_levels_{name}_after_{phase}"
        m.frameCapture.capture()
        hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
        m.renderFrame()
        p = hstr.properties
        print(f"LEVELS {name} after {phase} ({count} frames) vs {ARMS[0][0]} exact: >0.02 {100 * float(p['referenceNoiseError']):.3f}%  "
              f"p99.9 {float(p['referenceLogP999']):.4f}  max {float(p['referenceLogMax']):.3f}  {stats()}", flush=True)
        hstr.set_properties({"compareReference": False, "compareExact": False})
    print(f"LEVELS {name} frozen: {stats()} props {props}", flush=True)
exit()
