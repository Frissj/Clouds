import itertools
import json
import math
import os
from falcor import *

OUT = "C:/Users/Friss/AppData/Local/Temp/hstr_bench"
TAG = os.environ.get("HSTR_TAG", "fit")
GRID = json.loads(os.environ.get("HSTR_GRID", "{}"))
VIEWS = os.environ.get("HSTR_VIEWS", "static,backlit,oblique").split(",")
SPP = int(os.environ.get("HSTR_SPP", "256"))
W = int(os.environ.get("HSTR_W", "1920"))
H = int(os.environ.get("HSTR_H", "1080"))

target = float3(-10.0, 73.0, -43.0)
radius = 510.0
sun = float3(0.4319, 0.8639, 0.2699)


def place(cam, view):
    if view == "static":
        cam.position = float3(target.x + radius, 60.0, target.z)
    elif view == "close":
        cam.position = float3(target.x + radius * 0.45, 70.0, target.z)
    elif view == "oblique":
        a = 2.2
        cam.position = float3(target.x + 0.8 * radius * math.cos(a), 180.0, target.z + 0.8 * radius * math.sin(a))
    elif view == "backlit":
        cam.position = float3(target.x - 1.1 * radius * sun.x, target.y - 1.1 * radius * sun.y, target.z - 1.1 * radius * sun.z)
    cam.target = target


def trace(message):
    with open(f"{OUT}/{TAG}_trace.txt", "a") as f:
        f.write(message + "\n")


g = RenderGraph("Fit")
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


keys = list(GRID.keys())
combos = list(itertools.product(*[GRID[k] for k in keys])) if keys else [()]
results = {}
for view in VIEWS:
    place(cam, view)
    hstr.set_properties({"debugView": 6, "compareReference": False})
    for i in range(SPP):
        m.renderFrame()
    shoot(f"pt_{view}")
    trace(f"reference {view} done")
    for combo in combos:
        props = dict(zip(keys, combo))
        props["debugView"] = 0
        props["compareReference"] = True
        hstr.set_properties(props)
        for i in range(2):
            m.renderFrame()
        p = hstr.properties
        results.setdefault(combo, {})[view] = (float(p["referenceError"]), float(p["referenceLogError"]))

lines = []
for combo, errs in results.items():
    lines.append((sum(e[1] for e in errs.values()), combo, errs))
lines.sort()
with open(f"{OUT}/{TAG}_results.txt", "w") as f:
    f.write("sorted by summed log error; keys: " + ",".join(keys) + "\n")
    for total, combo, errs in lines:
        f.write(f"{total:.5f}  {combo}  " + "  ".join(f"{v}: L1 {e[0]:.4f} log {e[1]:.4f}" for v, e in errs.items()) + "\n")

best = dict(zip(keys, lines[0][1]))
best["debugView"] = 0
best["compareReference"] = False
hstr.set_properties(best)
for view in VIEWS:
    place(cam, view)
    for i in range(3):
        m.renderFrame()
    shoot(f"best_{view}")
exit()
