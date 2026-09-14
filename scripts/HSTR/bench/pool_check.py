from falcor import *

# Persistent photon pool vs whole-path batches: smooth-term error against a short reference, by frames of updates.
OUT = "C:/Users/Friss/Documents/HSTR_results/pool_check.txt"
g = RenderGraph("Pool")
g.addPass(createPass("HSTRCloud", {"cutTransmittanceTolerance": 0.05}), "HSTRCloud")
g.markOutput("HSTRCloud.color")
m.addGraph(g)
m.loadScene("data/HSTR/wdas_cloud.pyscene")
m.resizeFrameBuffer(480, 270)
hstr = g.getPass("HSTRCloud")
cam = m.scene.camera
cam.position = float3(500.0, 60.0, -43.0)
cam.target = float3(-10.0, 73.0, -43.0)
hstr.set_properties({"debugView": 6})
for i in range(256):
    m.renderFrame()
common = {"debugView": 8, "worldCacheCellVoxels": 2, "worldCacheBands": 2, "worldCacheEstimator": 1, "worldCacheTextured": 1,
          "worldCacheOrder": 2, "worldCacheWindow": 1, "residualStrength": 1.0}


def error():
    hstr.set_properties({"compareReference": True, "hstComponents": 12, "compareSubstitute": 0, "compareTarget": 12})
    m.renderFrame()
    p = hstr.properties
    hstr.set_properties({"compareReference": False})
    return float(p["referenceLogError"])


lines = []
for label, props, frames in (
    ("batches of 65k whole paths", {"worldCacheSegments": 0, "worldCachePhotons": 65536}, [4, 16, 32]),
    ("pool 65k x 16 segments", {"worldCacheSegments": 16, "worldCachePhotons": 65536}, [16, 64, 256, 1024]),
):
    hstr.set_properties(dict(common, worldCacheUpdates=0, worldCacheSegments=1 - min(1, props["worldCacheSegments"])))
    m.renderFrame()  # Restart the cache.
    hstr.set_properties(dict(common, worldCacheUpdates=1, **props))
    done = 0
    for target in frames:
        while done < target:
            m.renderFrame()
            done += 1
        lines.append(f"{label}: {done} updates, smooth-term log error {error():.5f}")
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
exit()
