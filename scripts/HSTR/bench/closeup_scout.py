import os
from falcor import *

# Top-down scout of the cloud sea around the origin, to pick a cloud for close-ups (pixel -> world: see closeup_compare.py).
OUT = "C:/Users/Friss/Documents/HSTR_results"
HEIGHT = float(os.environ.get("HSTR_SCOUT_HEIGHT", "600"))
m.script("scripts/HSTR/CloudSea.py")
m.resizeFrameBuffer(1280, 720)
hstr = m.activeGraph.getPass("HSTRCloud")
cam = m.scene.camera
cam.position = float3(0.0, HEIGHT, 0.0)
cam.up = float3(0.0, 0.0, 1.0)
cam.target = float3(0.0, 0.0, 0.0)
m.frameCapture.outputDir = OUT
for i in range(400):
    m.renderFrame()
    s = hstr.properties.get("cloudStats", {})
    if i > 32 and int(s.get("pendingTiles", 0)) == 0 and int(s.get("pending", 0)) == 0:
        break
for i in range(64):
    m.renderFrame()
m.frameCapture.baseFilename = "closeup_scout"
m.frameCapture.capture()
print("frame height mm", cam.frameHeight, "focal", cam.focalLength)
exit()
