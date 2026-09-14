import json
import math
import os
from falcor import *

OUT = os.environ.get("HSTR_OUT", "C:/Users/Friss/AppData/Local/Temp/hstr_bench")
TAG = os.environ.get("HSTR_TAG", "run")
W = int(os.environ.get("HSTR_W", "3840"))
H = int(os.environ.get("HSTR_H", "2160"))
# JSON: {"base": {props applied before timing}, "variants": [[name, {props}], ...]}
CONFIG = json.loads(os.environ.get("HSTR_CONFIG", "{}"))
os.makedirs(OUT, exist_ok=True)

m.script("scripts/HSTR/HSTRCloud.py")
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(W, H)
hstr = m.activeGraph.getPass("HSTRCloud")

cam = m.scene.camera
target = float3(-10.0, 73.0, -43.0)
start = float3(500.0, 60.0, -43.0)
radius = math.sqrt((start.x - target.x) ** 2 + (start.z - target.z) ** 2)


def set_orbit(angle, r=radius, h=60.0):
    cam.position = float3(target.x + r * math.cos(angle), h, target.z + r * math.sin(angle))
    cam.target = target


def report(label, capture):
    with open(f"{OUT}/{TAG}_timing.txt", "a") as f:
        f.write(f"== {label} ({capture['frame_count']} frames, {W}x{H})\n")
        for name, lane in capture["events"].items():
            if name.endswith("gpu_time") and "HSTRCloud" in name:
                f.write(f"{lane['stats']['mean']:9.3f} ms (max {lane['stats']['max']:9.3f})  {name}\n")


def shoot(name):
    m.frameCapture.baseFilename = f"{TAG}_{name}"
    m.frameCapture.capture()


if CONFIG.get("base"):
    hstr.set_properties(CONFIG["base"])

for i in range(8):
    m.renderFrame()

m.profiler.enabled = True
m.profiler.start_capture()
for i in range(40):
    m.renderFrame()
report("static", m.profiler.end_capture())
m.frameCapture.outputDir = OUT
shoot("static")

m.profiler.start_capture()
for i in range(40):
    set_orbit(0.02 * (i + 1))
    m.renderFrame()
report("orbit", m.profiler.end_capture())

set_orbit(2.2, radius * 0.8, 180.0)
m.renderFrame()
shoot("oblique")

set_orbit(0.0, radius * 0.45, 70.0)
m.renderFrame()
shoot("close")

# Backlit: the camera looks along the sun direction at the cloud (forward scattering, silver linings).
sun = float3(0.4319, 0.8639, 0.2699)
cam.position = float3(target.x - 1.1 * radius * sun.x, target.y - 1.1 * radius * sun.y, target.z - 1.1 * radius * sun.z)
cam.target = target
m.renderFrame()
shoot("backlit")

for name, props in CONFIG.get("variants", []):
    hstr.set_properties(props)
    for view in ["static", "close"]:
        if view == "static":
            set_orbit(0.0)
        else:
            set_orbit(0.0, radius * 0.45, 70.0)
        m.renderFrame()
        m.renderFrame()
        shoot(f"{name}_{view}")

exit()
