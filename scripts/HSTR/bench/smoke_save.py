from falcor import *

# Smoke test: save a small reference, reload it, compare per pixel and in blocks; temporal beam frames.
OUT = "C:/Users/Friss/Documents/HSTR_results"
g = RenderGraph("Smoke")
g.addPass(createPass("HSTRCloud", {"cutTransmittanceTolerance": 0.05}), "HSTRCloud")
g.markOutput("HSTRCloud.color")
m.addGraph(g)
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(480, 272)
hstr = g.getPass("HSTRCloud")
base = {"hstComponents": 15, "residualStrength": 1.0, "worldCacheCellVoxels": 2, "worldCacheBands": 2, "worldCacheOrder": 2,
        "worldCacheWindow": 1, "worldCacheEstimator": 1, "worldCacheSegments": 0, "worldCachePhotons": 65536}
hstr.set_properties(dict(base, debugView=8, worldCacheUpdates=4))
while int(hstr.properties["worldCacheSampleCount"]) < 16:
    m.renderFrame()
hstr.set_properties({"worldCacheUpdates": 0, "debugView": 6})
while int(hstr.properties["referenceSampleCount"]) < 16:
    m.renderFrame()
hstr.set_properties({"saveReference": OUT + "/smoke_ref"})
m.renderFrame()
lines = [f"accumulated {hstr.properties['referenceSampleCount']}"]


def compare(view, block):
    hstr.set_properties({"debugView": view, "compareBlock": block, "compareReference": True})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False})
    return f"view {view} block {block}: log err {float(p['referenceLogError']):.4f} noise {float(p['referenceNoiseLogError']):.4f}"


lines.append("before reload: " + compare(8, 1))
# Wipe by resizing, then reload.
m.resizeFrameBuffer(480, 270)
m.renderFrame()
m.resizeFrameBuffer(480, 272)
m.renderFrame()
lines.append(f"after resize {hstr.properties['referenceSampleCount']}")
hstr.set_properties({"loadReference": OUT + "/smoke_ref"})
m.renderFrame()
lines.append(f"reloaded {hstr.properties['referenceSampleCount']}")
lines.append("after reload: " + compare(8, 1))
lines.append("after reload: " + compare(8, 8))
hstr.set_properties({"beamTemporal": True, "beamTileSize": 16, "beamLevels": 3})
for i in range(4):
    m.renderFrame()
lines.append("temporal beam: " + compare(9, 1))
with open(OUT + "/smoke_save.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
exit()
