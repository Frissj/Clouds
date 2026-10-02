# A high-sample path-traced crop of the sunset start view, magnified, without tracing the rest of the frame: the camera stays where
# the launcher puts it, turns to the crop's centre and narrows its field of view to the crop, rendered HSTR_HILL_WIDTH (3840) wide. Settles with residency live (the narrower frustum spends the brick budget on the crop), freezes,
# accumulates the path tracer (debugView 6), then captures the exact march and the beam of the same crop beside it.
# Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/sunset_hill.py
#   HSTR_HILL_CROP=x,y,w,h in the 3840x2160 frame (400,850,700,400), HSTR_SPP (2048), HSTR_TRUTH_SECONDS (900),
#   HSTR_TRUTH_SETTLE (600), HSTR_HILL_OUT (HSTR_results/sunset_hill).
import math
import os
import sys
import time
from pathlib import Path
from falcor import *

root = Path(__file__).resolve().parents[3]
launcher = root / "scripts" / "HSTR" / "SunsetCloudSea.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
hstr = m.activeGraph.getPass("HSTRCloud")
# beamOct off: the octahedral beam image's resolution follows the pixel angle, and at the crop's narrow field of view it asked for
# a 12.7 GB buffer (3840-wide crop of a 700x400 region). The settle only needs residency; the beam is not captured.
hstr.set_properties({"atmosphereAerialDistance": 0.0, "beamOct": False})
# HSTR_HILL_PRE: properties (a dict literal) set before the settle, e.g. residency settings every arm shares.
hstr.set_properties(eval(os.environ.get("HSTR_HILL_PRE", "{}")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sea_config import REFERENCE

W, H = 3840, 2160
cx, cy, cw, ch = [int(v) for v in os.environ.get("HSTR_HILL_CROP", "400,850,700,400").split(",")]
cam = m.scene.camera
pos = [cam.position.x, cam.position.y, cam.position.z]
tgt = [cam.target.x, cam.target.y, cam.target.z]
up = [cam.up.x, cam.up.y, cam.up.z]


def norm(v):
    length = math.sqrt(sum(c * c for c in v))
    return [c / length for c in v]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


forward = norm([t - p for t, p in zip(tgt, pos)])
right = norm(cross(forward, up))
upv = cross(right, forward)
tanY = cam.frameHeight / (2.0 * cam.focalLength)
tanX = tanY * W / H
nx = (cx + 0.5 * cw) / W * 2.0 - 1.0
ny = 1.0 - (cy + 0.5 * ch) / H * 2.0
d = norm([f + r * nx * tanX + u * ny * tanY for f, r, u in zip(forward, right, upv)])
cam.target = float3(*[p + c for p, c in zip(pos, d)])
# The field of view is the crop's; the frame is HSTR_HILL_WIDTH wide (3840) at the crop's aspect, a magnified crop.
cam.focalLength = cam.focalLength * H / ch
OW = int(os.environ.get("HSTR_HILL_WIDTH", "3840"))
OH = int(round(OW * ch / cw))
m.resizeFrameBuffer(OW, OH)
print(f"HILL crop {cx},{cy} {cw}x{ch} rendered at {OW}x{OH}: focal {cam.focalLength:.2f} mm, direction {d}", flush=True)

OUT = os.environ.get("HSTR_HILL_OUT", "C:/Users/Friss/Documents/HSTR_results/sunset_hill")
os.makedirs(OUT, exist_ok=True)
m.frameCapture.outputDir = OUT
LEVELS = ("desired", "mapped", "pending", "mapBacklog", "sunWaiting", "sunBaked", "sunSlotsFree", "sunStale", "sunBakesFrame")


def stats():
    s = hstr.properties.get("cloudStats", {})
    return {k: s.get(k) for k in LEVELS}


def capture(label):
    m.frameCapture.baseFilename = f"sunset_hill_{label}"
    m.frameCapture.capture()


settle = int(os.environ.get("HSTR_TRUTH_SETTLE", "600"))
for i in range(0, settle, 300):
    for _ in range(min(300, settle - i)):
        m.renderFrame()
    print(f"HILL settle{i // 300}: {stats()}", flush=True)
beam = {k: hstr.properties[k] for k in REFERENCE if k in hstr.properties}
hstr.set_properties({"cloudResidencyFrozen": True})
for _ in range(60):
    m.renderFrame()
print(f"HILL frozen: {stats()}", flush=True)
if int(hstr.properties.get("cloudTraceSlot", -1)) >= 0:
    print("HILL cut trace begin", flush=True)
    print(hstr.properties.get("cutTrace", ""), flush=True)
    print("HILL cut trace end", flush=True)

SPP = int(os.environ.get("HSTR_SPP", "2048"))
BUDGET = float(os.environ.get("HSTR_TRUTH_SECONDS", "900"))
# HSTR_HILL_TAPS=1,4: no path trace; the beam at each beamRimTaps (captured, and timed over 48 frames of a small camera sway so
# every frame re-marches), then the exact march.
TAPS = [int(v) for v in os.environ.get("HSTR_HILL_TAPS", "").split(",") if v]
POOLS = [int(v) for v in os.environ.get("HSTR_HILL_POOLS", "").split(",") if v]
# HSTR_HILL_PROBES: (x, y) pixels whose exact-march steps are dumped (probePixel; the light column holds the step's majorant).
for x, y in eval(os.environ.get("HSTR_HILL_PROBES", "[]")):
    hstr.set_properties(dict(REFERENCE, probeX=x, probeY=y))
    m.renderFrame()
    values = [float(v) for v in hstr.properties.get("probeRecords", "").split(",") if v]
    print(f"PROBE pixel ({x}, {y}): {len(values) // 12} steps", flush=True)
    print("PROBE   dist     sigma     sunDepth  majorant   brick empty cell lvl des map  entry  inst layers  x y z", flush=True)
    for i in range(0, len(values) - 11, 12):
        d, sigma, sun, major, flags, entry, inst, layers, sx, sy, sz, empty = values[i:i + 12]
        # Only the steps that read density, unless within HSTR_HILL_PROBE_RANGE (near,far), where every step is shown.
        near, far = eval(os.environ.get("HSTR_HILL_PROBE_RANGE", "(0, 0)"))
        if sigma <= 0.0 and not near <= d < far:
            continue
        f = int(flags)
        print(f"PROBE {d:8.2f} {sigma:9.5f} {sun:9.4f} {major:10.6f}   {f & 1} {(f >> 1) & 1} {(f >> 2) & 1}  {(f >> 4) & 15} "
              f"{(f >> 8) & 15} {(f >> 12) & 15} {int(entry):7d} children {int(inst) & 255:08b} octant {int(inst) >> 8} "
              f"{'HAS' if (int(inst) >> (int(inst) >> 8)) & 1 else 'NO '} child   {int(layers) & 15}{(int(layers) >> 4) & 15} "
              f"e{int(empty) & 15:<2d} s{int(empty) >> 4:<7d} {sx:.1f} {sy:.1f} {sz:.1f}", flush=True)
    hstr.set_properties({"probeX": -1, "probeY": -1})
SUN_ARMS = eval(os.environ.get("HSTR_HILL_SUN_ARMS", "None"))
# HSTR_HILL_PT_ARMS: a Python list of (label, properties) over REFERENCE, each scored against the saved path-traced crop; properties
# may hold "frames" (default 8; a beam arm rebuilds over several).
PT_ARMS = eval(os.environ.get("HSTR_HILL_PT_ARMS", "None"))
if SUN_ARMS or PT_ARMS:
    # The exact view scored against the saved path-traced crop (HSTR_TRUTH_SETTLE / width must match its name) under each sun atlas
    # (label, cloudSunPoolScale, cloudSunAtlas8); a change rebuilds residency and settles again.
    ref = f"C:/Users/Friss/Documents/HSTR_results/references/sunset_hill_{cx}_{cy}_{cw}x{ch}_{OW}x{OH}"
    hstr.set_properties({"debugView": 6, "referenceShow": 15, "referenceBandRows": 135})
    m.renderFrame()
    hstr.set_properties({"loadReference": ref})
    m.renderFrame()
    # The reference view writes referenceBandRows rows a frame and storeExact copies the displayed frame: every band is drawn from
    # the loaded average before the store, or the rows not yet drawn are stale.
    for _ in range((OH + 134) // 135):
        m.renderFrame()
    capture("reference")
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    print(f"HILL reference {int(hstr.properties['referenceSampleCount'])} spp stored", flush=True)
    touched = {}  # Each property a PT arm set, at its value before the first: restored before every arm, so none leaks into the next.
    for label, *arm in (SUN_ARMS or PT_ARMS):
        props = dict(arm[0]) if PT_ARMS else {}
        frames = props.pop("frames", 8)
        for k in props:
            touched.setdefault(k, hstr.properties.get(k))
        hstr.set_properties({k: v for k, v in touched.items() if v is not None})
        if SUN_ARMS and (arm[0] != hstr.properties["cloudSunPoolScale"] or arm[1] != hstr.properties["cloudSunAtlas8"]):
            hstr.set_properties(dict(REFERENCE, cloudSunPoolScale=arm[0], cloudSunAtlas8=arm[1], cloudResidencyFrozen=False))
            for _ in range(settle):
                m.renderFrame()
            hstr.set_properties({"cloudResidencyFrozen": True})
        hstr.set_properties(dict(REFERENCE, compareReference=False, compareExact=False, **props))
        # The units the arm's rebuild lists (beamDirtyStats, when an arm sets it): the work side of its error.
        units_before = int(hstr.properties.get("cloudStats", {}).get("dirtyStatUnits", 0))
        for _ in range(frames):
            m.renderFrame()
        units = int(hstr.properties.get("cloudStats", {}).get("dirtyStatUnits", 0)) - units_before
        capture(label)
        # Path-traced arms over 8 x 8 blocks, as the gate scores (per pixel, the trace's own noise passes 0.02 almost everywhere).
        hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 8 if PT_ARMS else 1})
        m.renderFrame()
        p = hstr.properties
        print(f"HILL {label}: >0.02 {100 * float(p['referenceNoiseError']):.3f}%  p99 {float(p['referenceLogP99']):.4f}  "
              f"p99.9 {float(p['referenceLogP999']):.4f}  max {float(p['referenceLogMax']):.3f}  error {float(p['referenceError']):.4f} log {float(p['referenceLogError']):.4f}  "
              f"{stats()}", flush=True)
        if PT_ARMS:
            # Per pixel against the trace's halves, squared: the trace's noise variance is common to every arm and subtracted, so
            # "excess" is the arm's mean squared log error against the converged trace, tile-scale interpolation error included.
            hstr.set_properties({"compareExact": False, "compareBlock": 1, "compareTarget": 15, "compareSubstitute": 0,
                                 "compareSquared": True})
            m.renderFrame()
            p = hstr.properties
            mse, noise = float(p["referenceNoiseError"]), float(p["referenceNoiseLogError"])
            print(f"HILL {label} squared: mse {mse:.6f} noise {noise:.6f} excess {mse - noise:.6f} "
                  f"(rms {math.sqrt(max(mse - noise, 0.0)):.4f}) units {units}", flush=True)
            hstr.set_properties({"compareSquared": False})
        hstr.set_properties({"compareReference": False, "compareExact": False})
elif POOLS:
    # The exact view at each cloudSunPoolScale (sun atlas slots per density slot): a pool too small for the view's bakes leaves
    # bricks answered by an ancestor's bake. A change rebuilds residency, so each arm after the first settles again.
    for pool in POOLS:
        if pool != hstr.properties["cloudSunPoolScale"]:
            hstr.set_properties({"cloudSunPoolScale": pool, "cloudResidencyFrozen": False})
            for _ in range(settle):
                m.renderFrame()
            hstr.set_properties({"cloudResidencyFrozen": True})
        hstr.set_properties(dict(REFERENCE))
        for _ in range(8):
            m.renderFrame()
        capture(f"exact_pool{pool}")
        print(f"HILL captured pool {pool}: {stats()}", flush=True)
elif os.environ.get("HSTR_HILL_EXACT"):
    # Where the exact view's hair and brick grid come from: all components, no sun (background + sky), then the live sun (no bakes)
    # last and at half size, since the live sun at source-voxel steps has removed the device before.
    # HSTR_HILL_ARMS: a Python list of (label, properties) over REFERENCE instead; labels ending "_half" render at half size.
    arms = eval(os.environ.get("HSTR_HILL_ARMS", "None")) or [
        ("exact_all", {}), ("exact_nosun", {"hstComponents": 1 | 8}), ("exact_livesun_half", {"cloudSunCache": False})]
    compare_arms = bool(int(os.environ.get("HSTR_HILL_COMPARE", "0")))
    for arm_index, (label, props) in enumerate(arms):
        props = dict(props)
        if label.endswith("_half") and m.frameCapture is not None and OW > 0:
            m.resizeFrameBuffer(OW // 2, OH // 2)
        if props.pop("resettle", False):
            # A residency property: thawed and settled again under it, then frozen for the capture.
            hstr.set_properties(dict(props, cloudResidencyFrozen=False))
            for _ in range(settle):
                m.renderFrame()
            hstr.set_properties({"cloudResidencyFrozen": True})
        frames = props.pop("frames", 8)  # A beam arm rebuilds over several frames.
        hstr.set_properties(dict(REFERENCE, **props))
        for _ in range(frames):
            m.renderFrame()
        capture(label)
        print(f"HILL captured {label}: {stats()}", flush=True)
        print(f"HILL cut levels {label}: {hstr.properties.get('cutLevels', '')}", flush=True)
        if compare_arms:
            if arm_index == 0:
                hstr.set_properties({"storeExact": True})
                m.renderFrame()
            else:
                hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
                m.renderFrame()
                p = hstr.properties
                print(f"HILL {label} vs {arms[0][0]}: >0.02 {100 * float(p['referenceNoiseError']):.3f}%  "
                      f"p99.9 {float(p['referenceLogP999']):.4f}  max {float(p['referenceLogMax']):.3f}", flush=True)
                hstr.set_properties({"compareReference": False, "compareExact": False})
elif TAPS:
    from sea_config import BEAM
    base = [cam.target.x, cam.target.y, cam.target.z]
    for taps in TAPS:
        hstr.set_properties(dict(BEAM, beamOct=False, beamRimTaps=taps))
        for _ in range(60):
            m.renderFrame()
        capture(f"beam_taps{taps}")
        m.profiler.enabled = True
        m.profiler.start_capture()
        for i in range(48):
            a = 0.002 * math.sin(i * 0.5)
            cam.target = float3(base[0] + a, base[1], base[2] - a)
            m.renderFrame()
        events = m.profiler.end_capture()["events"]
        m.profiler.enabled = False
        cam.target = float3(*base)
        ms = next((lane["stats"]["mean"] for name, lane in events.items() if name.endswith("HSTRCloud/gpu_time")), 0.0)
        print(f"HILL beam taps {taps}: HSTRCloud {ms:.3f} ms per frame (swaying)", flush=True)
    hstr.set_properties(dict(REFERENCE))
    for _ in range(8):
        m.renderFrame()
    capture("exact")
else:
    # Mogwai's exit() only requests shutdown and the script runs on, so the trace is this branch rather than after an exit().
    hstr.set_properties({"debugView": 6, "referenceShow": 15, "referenceBandRows": 135})
    m.renderFrame()
    # Resumed and saved every 16 samples (HSTR_results/references/sunset_hill_<crop>_<W>x<H>), so a longer trace is more runs.
    PATH = f"C:/Users/Friss/Documents/HSTR_results/references/sunset_hill_{cx}_{cy}_{cw}x{ch}_{OW}x{OH}"
    if os.path.exists(PATH + "_s0.exr"):
        hstr.set_properties({"loadReference": PATH})
        m.renderFrame()
        print(f"HILL resumed {int(hstr.properties['referenceSampleCount'])} spp", flush=True)
    start = time.time()
    while int(hstr.properties["referenceSampleCount"]) < SPP and time.time() - start < BUDGET:
        m.renderFrame()
        if int(hstr.properties["referenceSampleCount"]) % 16 == 0:
            hstr.set_properties({"saveReference": PATH})
            m.renderFrame()
    hstr.set_properties({"saveReference": PATH})
    m.renderFrame()
    samples = int(hstr.properties["referenceSampleCount"])
    print(f"HILL reference {samples} spp in {time.time() - start:.0f} s", flush=True)
    capture(f"pt_{samples}spp")
    # HSTR_HILL_SPLIT=3,5: the path trace and then the exact march restricted to each component set (kComponent* bits).
    SPLIT = [int(v) for v in os.environ.get("HSTR_HILL_SPLIT", "").split(",") if v]
    for bits in SPLIT:
        hstr.set_properties({"referenceShow": bits})
        for _ in range((OH + 134) // 135 + 1):
            m.renderFrame()
        capture(f"pt_c{bits}")

    hstr.set_properties(dict(REFERENCE))
    for _ in range(8):
        m.renderFrame()
    capture("exact")
    for bits in SPLIT:
        hstr.set_properties({"hstComponents": bits})
        for _ in range(8):
            m.renderFrame()
        capture(f"exact_c{bits}")
    # HSTR_HILL_SCORES=[(label, components)]: the exact march restricted to the components against the same components of the
    # path trace just accumulated, over 8x8 blocks (the trace's own noise from its two halves beside it), and the signed error
    # map (red: the march brighter, blue: darker; full scale 0.2 in log radiance). The march's cache holds the multiply scattered
    # sun and the scattered sky together, so those two bits only make sense as 12.
    # An optional third element is a dict of properties for that arm and the ones after it (a cache setting: the cache is given
    # 300 frames to rebuild under it). A cache that has met worldCacheTarget traces nothing more, and a changed setting only
    # zeroes its sample count: raise worldCacheTarget past the last arm's with each change, or the old deposits are scored.
    # "frames" in the dict overrides the 300 (a setting that needs no rebuild, e.g. cacheOracle).
    for label, bits, *extra in eval(os.environ.get("HSTR_HILL_SCORES", "[]")):
        props = dict(extra[0]) if extra else {}
        frames = props.pop("frames", 300 if extra else 8)
        hstr.set_properties(dict(REFERENCE, hstComponents=bits, compareTarget=bits, compareSubstitute=0, compareReference=True,
                                 compareExact=False, compareBlock=8, compareMapScale=0.2, **props))
        for _ in range(frames):
            m.renderFrame()
        p = hstr.properties
        print(f"HILL score {label}: error {float(p['referenceError']):.4f} log {float(p['referenceLogError']):.4f}  "
              f"noise {float(p['referenceNoiseError']):.4f} log {float(p['referenceNoiseLogError']):.4f}  "
              f"cache samples {int(p['worldCacheSampleCount'])}", flush=True)
        capture(f"map_{label}")
        hstr.set_properties({"compareReference": False, "compareMapScale": 0.0})
exit()
