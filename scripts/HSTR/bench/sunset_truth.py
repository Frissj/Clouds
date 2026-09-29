# The sunset sea against the path tracer: RunSunsetCloudSea.bat's graph (SunsetCloudSea.py) at its start view, headless, 4K. Settles
# with residency live, freezes it (the path tracer reads the resident fine bricks), accumulates the path-traced reference (debugView
# 6) into HSTR_results/references/sunset_start_v2_WxH, resumed and saved every 16 samples, then scores the launcher's beam view and the
# exact per-pixel march (sea_config.REFERENCE) against it: share of pixels over 0.02 log error (the gate: under 1%), p99.9, max, and
# the mean log error beside the reference's own noise.
# The path tracer has no air, so the whole run sets atmosphereAerialDistance 0: the atmosphere's sun, sky and background stay, the
# aerial haze over the clouds goes (hazedBeamRadiance).
# Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/sunset_truth.py
#   HSTR_RES=WxH (default 3840x2160), HSTR_TRUTH_SETTLE=settle frames (1800), HSTR_SPP=reference samples (256),
#   HSTR_TRUTH_SECONDS=path-trace time budget this run (600; resumes next run), HSTR_TRUTH_OUT=capture directory,
#   HSTR_TRUTH_PROPS=a Python dict literal of HSTRCloud properties for the beam view (an arm to score).
import os
import sys
import time
from pathlib import Path
from falcor import *

root = Path(__file__).resolve().parents[3]
launcher = root / "scripts" / "HSTR" / "SunsetCloudSea.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
W, H = [int(v) for v in os.environ.get("HSTR_RES", "3840x2160").split("x")]
m.resizeFrameBuffer(W, H)
hstr = m.activeGraph.getPass("HSTRCloud")
hstr.set_properties({"atmosphereAerialDistance": 0.0})
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sea_config import REFERENCE

OUT = "C:/Users/Friss/Documents/HSTR_results"
# v2: sunset_start_ (98 spp) is invalid - traced while residency livelocked (most of the sea on blurred level-3/4 bricks) and the
# layered lookup answered the proxy outside every layer (thin sheets past the asset boxes). v2 is the whole library resident, fully
# mapped, proxy gone there.
PATH = f"{OUT}/references/sunset_start_v2_{W}x{H}"
SPP = int(os.environ.get("HSTR_SPP", "256"))
BUDGET = float(os.environ.get("HSTR_TRUTH_SECONDS", "600"))
SAVE_EVERY = 16
CAPTURES = os.environ.get("HSTR_TRUTH_OUT", f"{OUT}/sunset_truth")
LEVELS = ("desired", "mapped", "pending", "mapBacklog", "sunWaiting", "sunStale", "sunBakesFrame")


def stats():
    s = hstr.properties.get("cloudStats", {})
    return {k: s.get(k) for k in LEVELS}


def frames(n):
    for _ in range(n):
        m.renderFrame()


def capture(label):
    os.makedirs(CAPTURES, exist_ok=True)
    m.frameCapture.outputDir = CAPTURES
    m.frameCapture.baseFilename = f"sunset_truth_{label}"
    m.frameCapture.capture()


settle = int(os.environ.get("HSTR_TRUTH_SETTLE", "1800"))
for checkpoint in range(0, settle, 300):
    frames(300)
    print(f"TRUTH settle{checkpoint // 300}: {stats()}", flush=True)
# The resident set is the medium the reference traces; a resumed reference is only valid if a new run settles to the same set.
hstr.set_properties({"cloudResidencyFrozen": True})
frames(60)
print(f"TRUTH frozen: {stats()}", flush=True)
beam = {k: hstr.properties[k] for k in set(REFERENCE) | {"referenceShow", "referenceBandRows"} if k in hstr.properties}

# Reference: resumed if saved, accumulated for the budget, saved every SAVE_EVERY samples.
os.makedirs(f"{OUT}/references", exist_ok=True)
hstr.set_properties({"debugView": 6, "referenceShow": 15, "referenceBandRows": int(os.environ.get("HSTR_BAND_ROWS", "135"))})
m.renderFrame()
if os.path.exists(PATH + "_s0.exr"):
    hstr.set_properties({"loadReference": PATH})
    m.renderFrame()
    print(f"TRUTH resumed {int(hstr.properties['referenceSampleCount'])} spp from {PATH}", flush=True)
start = time.time()
first = int(hstr.properties["referenceSampleCount"])
while int(hstr.properties["referenceSampleCount"]) < SPP and time.time() - start < BUDGET:
    m.renderFrame()
    if int(hstr.properties["referenceSampleCount"]) % SAVE_EVERY == 0:
        hstr.set_properties({"saveReference": PATH})
        m.renderFrame()
# One more sample keeps the average as the exact frame (storeExact in the reference view), and the state is saved.
hstr.set_properties({"storeExact": True, "saveReference": PATH})
m.renderFrame()
samples = int(hstr.properties["referenceSampleCount"])
seconds = time.time() - start
print(f"TRUTH reference {samples} spp ({samples - first} this run in {seconds:.0f} s, "
      f"{seconds / max(samples - first, 1):.2f} s/spp)", flush=True)
capture("reference")


def score(label, props, warm):
    # compareExact against the stored path-traced average: zw are the shares over 0.02 and 0.1. Then the reference compare (the
    # accumulated average itself): mean log error beside the reference's own noise, which the threshold share does not remove.
    hstr.set_properties(dict(props, compareReference=False, compareExact=False))
    frames(warm)
    capture(label)
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    p = hstr.properties
    result = {"over02": float(p["referenceNoiseError"]), "p999": float(p["referenceLogP999"]), "max": float(p["referenceLogMax"])}
    hstr.set_properties({"compareExact": False, "compareTarget": 15, "compareSubstitute": 0})
    m.renderFrame()
    p = hstr.properties
    result.update(meanLog=float(p["referenceLogError"]), refNoiseLog=float(p["referenceNoiseLogError"]))
    hstr.set_properties({"compareReference": False})
    print(f"TRUTH {label} ({samples} spp): >0.02 {100 * result['over02']:.3f}%  p99.9 {result['p999']:.4f}  max "
          f"{result['max']:.3f}  mean log {result['meanLog']:.4f} (reference noise {result['refNoiseLog']:.4f})", flush=True)
    return result


arm = eval(os.environ.get("HSTR_TRUTH_PROPS", "{}"))
score("beam", dict(beam, **arm), 240)
print(f"TRUTH after beam: {stats()}", flush=True)
score("exact", dict(REFERENCE), 8)
exit()
