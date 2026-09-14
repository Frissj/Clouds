import math
from falcor import *

OUT = "C:/Users/Friss/Documents/HSTR_results"
g = RenderGraph("SplitCapture")
g.addPass(createPass("HSTRCloud", {"cutTransmittanceTolerance": 0.05}), "HSTRCloud")
g.addPass(createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0}), "ToneMapper")
g.addEdge("HSTRCloud.color", "ToneMapper.src")
g.markOutput("ToneMapper.dst")
m.addGraph(g)
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(960, 540)
hstr = g.getPass("HSTRCloud")
cam = m.scene.camera
m.frameCapture.outputDir = OUT
target = float3(-10.0, 73.0, -43.0)
radius = 510.0
back_sun = (-0.5 / 1.0, 0.6, 0.62)
n = math.sqrt(sum(c * c for c in back_sun))
back_sun = tuple(c / n for c in back_sun)
cases = [
    ("static", (target.x + radius, 60.0, target.z), (0.4319, 0.8639, 0.2699)),
    ("oblique_backsun", (target.x + 0.8 * radius * math.cos(2.2), 180.0, target.z + 0.8 * radius * math.sin(2.2)), back_sun),
]
for name, position, sun in cases:
    cam.position = float3(*position)
    cam.target = target
    hstr.set_properties({"sunDirection": float3(*sun), "debugView": 8, "hstComponents": 15, "residualStrength": 1.0, "worldCacheCellVoxels": 2,
                         "worldCacheBands": 2, "worldCacheOrder": 2, "worldCacheWindow": 1, "worldCacheEstimator": 1, "worldCacheUpdates": 4})
    while int(hstr.properties["worldCacheSampleCount"]) < 32:
        m.renderFrame()
    hstr.set_properties({"worldCacheUpdates": 0})
    m.renderFrame()
    m.frameCapture.baseFilename = f"capture_{name}_split"
    m.frameCapture.capture()
exit()
