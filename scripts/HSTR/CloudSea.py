import os
from pathlib import Path
from falcor import *

# Interactive cloud sea: an endless procedural layer of a compiled cloud library, rendered through the beam view with fine density
# paged in around the camera. The package is compiled offline from the VDBs:
#   HSTRCloudCompiler.exe <clouds_hr/vdb> <clouds_hr/clouds_1gb.hstrlib> --budget-mb 953
# Environment overrides: HSTR_CLOUD_LIBRARY (the .hstrlib), HSTR_CLOUD_POOL_MB, HSTR_CLOUD_LOADS, HSTR_SEA_TILES.
library = Path(os.environ.get("HSTR_CLOUD_LIBRARY", str(Path.home() / "Downloads" / "clouds_hr" / "clouds_1gb.hstrlib")))
tiles = int(os.environ.get("HSTR_SEA_TILES", "8"))

g = RenderGraph("CloudSea")
g.addPass(createPass("HSTRCloud", {
    "debugView": 9,
    "cloudLibrary": str(library),
    "cloudProxyResolution": int(os.environ.get("HSTR_SEA_VOXELS", "64")),
    "cloudSeaTiles": tiles,
    "cloudSeaCoverage": float(os.environ.get("HSTR_SEA_COVERAGE", "0.85")),
    "cloudBrickPoolMB": int(os.environ.get("HSTR_CLOUD_POOL_MB", "256")),
    "cloudBrickLoadsPerFrame": int(os.environ.get("HSTR_CLOUD_LOADS", "1024")),
    # Clamped to the resident window of tiles around the camera.
    "seaViewDistance": 20000.0,
    "worldCacheCellVoxels": int(os.environ.get("HSTR_SEA_CELL", "4")),
    "worldCachePhotons": 65536,
    "worldCacheUpdates": 1,
    "worldCacheBakeInterval": 8,
    "sunNearVoxels": 2.0,
    "beamTileSize": 16,
    "beamLevels": 3,
    "beamTolerance": 0.05,
    "beamTemporal": True,
}), "HSTRCloud")
g.addPass(createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0}), "ToneMapper")
g.addEdge("HSTRCloud.color", "ToneMapper.src")
g.markOutput("ToneMapper.dst")
m.addGraph(g)
m.loadScene("data/HSTR/cloud_sea.pyscene")
