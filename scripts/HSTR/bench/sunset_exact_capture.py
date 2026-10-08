# The per-pixel march of the sunset start view at the shipping step settings (debugView 8, everything else as the launcher sets it):
# the image the beam's units would produce at every pixel, captured as HDR (HSTRCloud.color, EXR) beside the tone-mapped frame.
# Input for offline tiling oracles (tile_oracle.py). Settles with residency live, freezes, renders a few exact frames, captures.
# Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/sunset_exact_capture.py
#   HSTR_RES (3840x2160), HSTR_EXACT_SETTLE (1200), HSTR_EXACT_OUT (HSTR_results/sunset_exact), HSTR_EXACT_TAG (exact).
import os
from pathlib import Path
from falcor import *

root = Path(__file__).resolve().parents[3]
launcher = root / "scripts" / "HSTR" / "SunsetCloudSea.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
m.resizeFrameBuffer(*[int(v) for v in os.environ.get("HSTR_RES", "3840x2160").split("x")])
hstr = m.activeGraph.getPass("HSTRCloud")
m.activeGraph.markOutput("HSTRCloud.color")

OUT = os.environ.get("HSTR_EXACT_OUT", "C:/Users/Friss/Documents/HSTR_results/sunset_exact")
os.makedirs(OUT, exist_ok=True)
m.frameCapture.outputDir = OUT

settle = int(os.environ.get("HSTR_EXACT_SETTLE", "1200"))
for i in range(settle):
    m.renderFrame()
hstr.set_properties({"cloudResidencyFrozen": True})
for _ in range(60):
    m.renderFrame()
s = hstr.properties.get("cloudStats", {})
print(f"EXACT settled: mapped {s.get('mapped')} pending {s.get('pending')} sunWaiting {s.get('sunWaiting')}", flush=True)
hstr.set_properties({"debugView": 8})
for _ in range(4):
    m.renderFrame()
tag = os.environ.get("HSTR_EXACT_TAG", "exact")
m.frameCapture.baseFilename = tag
m.frameCapture.capture()
# The EXR is PIZ-compressed half (no reader in plain Python, no numpy in Mogwai's): the same frame raw for the offline oracles.
hstr.set_properties({"saveColorRaw": os.path.join(OUT, f"{tag}_color.raw")})
m.renderFrame()
print("EXACT captured", flush=True)
exit()
