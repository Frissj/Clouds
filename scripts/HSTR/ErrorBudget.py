import math
import os
from falcor import *

# Splits the HST error against the path-traced reference by radiance component, on the views the sun model was
# fitted to and on hold-out camera/sun cases. Writes %TEMP%/hstr_bench/<tag>_budget.txt.
#
# Components (bits of hstComponents / referenceShow / compareSubstitute / compareTarget):
#   1 background   sky seen without scattering            HST: transmittance * sky
#   2 sun single   sun scattered once                     HST: sun residual octave 0
#   4 sun multiple sun scattered two or more times        HST: sun residual octaves 1-3
#   8 sky          sky scattered one or more times        HST: HST camera source

OUT = "C:/Users/Friss/AppData/Local/Temp/hstr_bench"
TAG = os.environ.get("HSTR_TAG", "budget")
# Reference noise depends on samples per pixel, not resolution, and falls slowly (forward-peaked sun NEE): measure at a
# reduced resolution with many samples rather than at 1080p with few.
SPP = int(os.environ.get("HSTR_SPP", "1024"))
W = int(os.environ.get("HSTR_W", "960"))
H = int(os.environ.get("HSTR_H", "540"))
ONLY = [c for c in os.environ.get("HSTR_CASES", "").split(",") if c]

target = float3(-10.0, 73.0, -43.0)
radius = 510.0
FIT_SUN = (0.4319, 0.8639, 0.2699)

BACKGROUND, SUN1, SUNN, SKY, ALL = 1, 2, 4, 8, 15
NAMES = {BACKGROUND: "background", SUN1: "sun single", SUNN: "sun multiple", SKY: "sky scattered"}


def normalized(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def orbit(azimuth_deg, height, distance=0.8):
    a = math.radians(azimuth_deg)
    return (target.x + distance * radius * math.cos(a), height, target.z + distance * radius * math.sin(a))


def opposite_sun(sun, distance=1.1):
    return (target.x - distance * radius * sun[0], target.y - distance * radius * sun[1], target.z - distance * radius * sun[2])


# (name, split, camera position, sun direction). The first three are the views the sun octaves were fitted on.
low_sun = normalized((0.9, 0.25, 0.35))
side_sun = normalized((-0.3, 0.9, -0.3))
back_sun = normalized((-0.5, 0.6, 0.62))
CASES = [
    ("static", "fit", (target.x + radius, 60.0, target.z), FIT_SUN),
    ("backlit", "fit", opposite_sun(FIT_SUN), FIT_SUN),
    ("oblique", "fit", orbit(math.degrees(2.2), 180.0), FIT_SUN),
    ("orbit60", "holdout", orbit(60.0, 60.0), FIT_SUN),
    ("orbit200", "holdout", orbit(200.0, 120.0), FIT_SUN),
    ("static_lowsun", "holdout", (target.x + radius, 60.0, target.z), low_sun),
    ("static_sidesun", "holdout", (target.x + radius, 60.0, target.z), side_sun),
    ("oblique_backsun", "holdout", orbit(math.degrees(2.2), 180.0), back_sun),
    ("backlit_backsun", "holdout", opposite_sun(back_sun), back_sun),
]
if ONLY:
    CASES = [c for c in CASES if c[0] in ONLY]

os.makedirs(OUT, exist_ok=True)


def trace(message):
    with open(f"{OUT}/{TAG}_trace.txt", "a") as f:
        f.write(message + "\n")


g = RenderGraph("ErrorBudget")
g.addPass(createPass("HSTRCloud", {"cutTransmittanceTolerance": 0.05}), "HSTRCloud")
g.addPass(createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0}), "ToneMapper")
g.addEdge("HSTRCloud.color", "ToneMapper.src")
g.markOutput("ToneMapper.dst")
m.addGraph(g)
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(W, H)
hstr = g.getPass("HSTRCloud")
cam = m.scene.camera
m.frameCapture.outputDir = OUT
trace("scene loaded")


def shoot(name):
    m.frameCapture.baseFilename = f"{TAG}_{name}"
    m.frameCapture.capture()


SKY_SCALES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]


def measure(hst, substitute, target_mask, debug_view=0, extra=None):
    """Error of (HST components `hst` + reference components `substitute`) against reference components `target_mask`.
    debug_view 4 integrates every pixel exactly through the cut instead of using the tile camera basis."""
    props = {"debugView": debug_view, "compareReference": True, "hstComponents": hst, "compareSubstitute": substitute, "compareTarget": target_mask}
    props.update(extra or {})
    hstr.set_properties(props)
    for i in range(2):
        m.renderFrame()
    p = hstr.properties
    return (float(p["referenceError"]), float(p["referenceLogError"]), float(p["referenceNoiseError"]), float(p["referenceNoiseLogError"]))


rows = {}
for name, split, position, sun in CASES:
    hstr.set_properties({"sunDirection": float3(*sun), "debugView": 6, "compareReference": False, "referenceShow": ALL})
    cam.position = float3(*position)
    cam.target = target
    for i in range(SPP):
        m.renderFrame()
    samples = int(hstr.properties["referenceSampleCount"])
    shoot(f"{name}_pt")
    for bit in (SUN1, SUNN, SKY):
        hstr.set_properties({"referenceShow": bit})
        m.renderFrame()
        shoot(f"{name}_pt_{NAMES[bit].replace(' ', '_')}")
    trace(f"reference {name} done ({samples} spp)")

    r = {"split": split, "spp": samples}
    # Mean radiance and mean log(1 + radiance) of each reference component: an empty frame against it.
    for bit in NAMES:
        r[f"energy {bit}"] = measure(0, 0, bit)
    r["total"] = measure(ALL, 0, ALL)
    # Each HST component against its reference counterpart, and its mean (against an empty target) for the signed bias.
    for bit in NAMES:
        r[f"component {bit}"] = measure(bit, 0, bit)
        r[f"hst mean {bit}"] = measure(bit, 0, 0)
    # The HST sky and the total with exact per-pixel integration: separates transport error from camera-basis error.
    r["component sky exact"] = measure(SKY, 0, SKY, 4)
    r["hst mean sky exact"] = measure(SKY, 0, 0, 4)
    r["total exact"] = measure(ALL, 0, ALL, 4)
    # Oracle substitution: the full error once one HST component is replaced by the reference.
    for bit in NAMES:
        r[f"oracle {bit}"] = measure(ALL & ~bit, bit, ALL)
    # Exact single scattering and exact sun: what remains for the HST sky, and for the HST sky plus fitted multiple sun.
    r["oracle sun"] = measure(ALL & ~(SUN1 | SUNN), SUN1 | SUNN, ALL)
    r["oracle all but sky"] = measure(SKY, BACKGROUND | SUN1 | SUNN, ALL)
    # Does the HST sky carry structure beyond opacity? Scaled HST sky against a scaled opacity-weighted ambient baseline.
    r["sky hst"] = {k: measure(SKY, 0, SKY, extra={"skyScale": k, "skyAmbient": 0.0}) for k in SKY_SCALES}
    r["sky ambient"] = {k: measure(SKY, 0, SKY, extra={"skyScale": 1.0, "skyAmbient": k}) for k in SKY_SCALES}
    rows[name] = r

    hstr.set_properties({"hstComponents": ALL, "compareSubstitute": 0, "compareTarget": ALL, "compareReference": False, "skyScale": 1.0, "skyAmbient": 0.0})
    for i in range(3):
        m.renderFrame()
    shoot(f"{name}_hst")
    trace(f"measured {name}")


def fmt(e):
    return f"L1 {e[0]:.4f} log {e[1]:.4f}"


with open(f"{OUT}/{TAG}_budget.txt", "w") as f:
    f.write("HST error budget against the path-traced reference (mean per pixel; 'noise' is the reference's own expected error).\n")
    f.write("Components: 1 background, 2 sun single, 4 sun multiple, 8 sky scattered.\n\n")
    for name, r in rows.items():
        total = r["total"]
        f.write(f"== {name} [{r['split']}] {r['spp']} spp; reference noise L1 {total[2]:.4f} log {total[3]:.4f}\n")
        f.write(f"  total HST error            {fmt(total)}   (exact per-pixel integration: {fmt(r['total exact'])})\n")
        for bit, label in NAMES.items():
            energy = r[f"energy {bit}"]
            comp = r[f"component {bit}"]
            oracle = r[f"oracle {bit}"]
            hst_mean = r[f"hst mean {bit}"][0]
            f.write(
                f"  {label:14s} ref mean {energy[0]:.4f} HST mean {hst_mean:.4f} | HST vs ref {fmt(comp)} (noise L1 {comp[2]:.4f} log {comp[3]:.4f})"
                f" | total if exact: {fmt(oracle)}\n"
            )
        f.write(f"  sky scattered with exact per-pixel integration: {fmt(r['component sky exact'])}\n")
        f.write(f"  total with exact sun (single + multiple): {fmt(r['oracle sun'])}\n")
        f.write(f"  total with only the HST sky left:         {fmt(r['oracle all but sky'])}\n\n")

    f.write("== Sums of log error\n")
    for split in ("fit", "holdout"):
        subset = [r for r in rows.values() if r["split"] == split]
        if not subset:
            continue
        f.write(f"  {split} ({len(subset)} cases): total {sum(r['total'][1] for r in subset):.4f}")
        for bit, label in NAMES.items():
            f.write(f" | exact {label} {sum(r[f'oracle {bit}'][1] for r in subset):.4f}")
        f.write(f" | exact sun {sum(r['oracle sun'][1] for r in subset):.4f}")
        f.write(f" | noise {sum(r['total'][3] for r in subset):.4f}\n")

    # Sky model test: one scale per model, chosen on the fitted views, scored on every split (sky component, log error).
    f.write("\n== Scattered sky: HST field vs opacity-weighted ambient, one scale per model chosen on the fitted views\n")
    fit_rows = [r for r in rows.values() if r["split"] == "fit"] or list(rows.values())
    for model in ("sky hst", "sky ambient"):
        best = min(SKY_SCALES, key=lambda k: sum(r[model][k][1] for r in fit_rows))
        f.write(f"  {model:12s} scale {best:.2f}:")
        for split in ("fit", "holdout"):
            subset = [r for r in rows.values() if r["split"] == split]
            if subset:
                f.write(f"  {split} log {sum(r[model][best][1] for r in subset):.4f} L1 {sum(r[model][best][0] for r in subset):.4f}")
        if model == "sky hst":
            f.write(f"  (unscaled: fit log {sum(r[model][1.0][1] for r in fit_rows):.4f})")
        f.write("\n")
    for name, r in rows.items():
        f.write(f"  {name:16s}")
        for model in ("sky hst", "sky ambient"):
            k = min(SKY_SCALES, key=lambda s: r[model][s][1])
            f.write(f"  {model} best scale {k:.2f} log {r[model][k][1]:.4f}")
        f.write(f"  (sky noise log {r['component 8'][3]:.4f}; HST sky mean basis {r['hst mean 8'][0]:.6f} exact {r['hst mean sky exact'][0]:.6f})\n")
trace("done")
exit()
